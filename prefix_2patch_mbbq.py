#!/usr/bin/env python
"""prefix_patch_cli.py — Bias × Alignment Neuron Scanner (CrowS‑Pairs & MBBQ)
Revision 2025‑07‑31 g — stack size mismatch fixed for alignment scan
────────────────────────────────────────────────────────────────────────
*   Bias‑patch scan (old de‑bias)
*   Alignment‑specific scan (new safety neurons)
*   Bias‑probe / pair‑gap post analysis
All in one CLI.  TXT files are optional thanks to --auto_pair_align.
"""
from __future__ import annotations
import argparse, collections, importlib, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

# ファイル冒頭に追加
from tqdm.auto import tqdm


import torch, torch.nn.functional as F
import pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ───────────────────────────────────────────── bitsandbytes hot‑patch (Colab)
if not hasattr(torch._utils, "try_import"):

    def _try_import(name):
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    torch._utils.try_import = _try_import  # type: ignore

CROWS_URL_DEFAULT = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
    "data/crows_pairs_anonymized.csv"
)

# ╭───────────────────────────── CLI parsing ╮


def parse_layers(arg: str, n_layers: int) -> Sequence[int] | str:
    if arg.strip().lower() == "all":
        return "all"
    idxs = sorted({int(x) for x in arg.split(",") if x})
    if any(i < 0 or i >= n_layers for i in idxs):
        raise argparse.ArgumentTypeError(
            f"layer index {idxs} out of range 0..{n_layers-1}"
        )
    return idxs


def get_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neuron‑level bias & alignment scanner")
    # benchmark + basic I/O
    p.add_argument("--benchmark", choices=["crows", "mbbq"], default="crows")
    p.add_argument("--crows_url", default=CROWS_URL_DEFAULT)
    p.add_argument("--mbbq_path")
    p.add_argument("--bias_type")
    p.add_argument(
        "--num_prompts", default="all", help="使うプロンプト数（数値 or 'all'）"
    )
    # neuron scanning
    p.add_argument("--layers", default="all")
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--report_top_layers", type=int, default=10)
    p.add_argument("--report_top_neurons", type=int, default=10)
    p.add_argument("--save_csv", action="store_true")
    # post analysis
    p.add_argument(
        "--pair_gap", help="bias_type prefix whose *_more/_less rows will be paired"
    )
    # alignment scan parameters
    p.add_argument("--harmful_txt")
    p.add_argument("--normal_txt")
    p.add_argument(
        "--auto_pair_align",
        action="store_true",
        help="derive harmful/normal from *_more/less or sent_more/less when TXT omitted",
    )
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--min_precision", type=float, default=0.9)
    p.add_argument("--min_support", type=int, default=5)
    # model loading
    p.add_argument("--device", default="cuda")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--base_id", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--aligned_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    return p.parse_args()


# ╭──────────────────────────── Hooks ╮


def cap(name: str, cache: Dict[str, torch.Tensor]):
    def _hook(_, __, out):
        cache[name] = (out[0] if isinstance(out, tuple) else out).detach()

    return _hook


def patch(name: str, src: Dict[str, torch.Tensor]):
    def _hook(_, __, out):
        tgt = src[name].to(out[0].device if isinstance(out, tuple) else out.device)
        return (tgt,) + out[1:] if isinstance(out, tuple) else tgt

    return _hook


def zero_many(idxs: List[int]):
    idxs_t = torch.as_tensor(idxs)

    def _hook(_, __, out):
        x = out[0] if isinstance(out, tuple) else out
        x[..., idxs_t] = 0.0
        return (x,) if isinstance(out, tuple) else x

    return _hook


# ╭──────────────────────── Utility ╮


def kld(p: torch.Tensor, q: torch.Tensor) -> float:
    p = F.softmax(p, -1)
    ql = F.log_softmax(q, -1)
    return F.kl_div(ql, p, reduction="sum", log_target=False).item()


# ╭──── Activation‑patching experiment ╮


def run_experiment(base, aligned, tok, prompt: str, layer_spec, top_k: int):
    """Return stats for one prompt"""
    device = next(base.parameters()).device
    inp = tok(prompt, return_tensors="pt").to(device)
    L = base.config.num_hidden_layers
    # 1) cache aligned
    cache: Dict[str, torch.Tensor] = {}
    tmp = [
        aligned.get_submodule(f"model.layers.{i}.{c}").register_forward_hook(
            cap(f"model.layers.{i}.{c}", cache)
        )
        for i in range(L)
        for c in ("self_attn", "mlp")
    ]
    aligned(**inp)
    for h in tmp:
        h.remove()
    base_logits = base(**inp).logits[0, -1, :]
    # 2) cumulative patch sets
    layer_sets = (
        [tuple(range(k + 1)) for k in range(L)]
        if layer_spec == "all"
        else [tuple(layer_spec)]
    )
    results: List[Tuple[Sequence[int], float]] = []
    for ls in layer_sets:
        hs = [
            base.get_submodule(f"model.layers.{l}.{c}").register_forward_hook(
                patch(f"model.layers.{l}.{c}", cache)
            )
            for l in ls
            for c in ("self_attn", "mlp")
        ]
        kl = kld(base_logits, base(**inp).logits[0, -1, :])
        results.append((ls, kl))
        for h in hs:
            h.remove()
    # 3) star layer
    if layer_spec == "all":
        deltas = [results[i - 1][1] - r[1] if i else -1 for i, r in enumerate(results)]
        star = results[int(np.argmax(deltas))][0][-1]
    else:
        star = results[0][0][0]
    # 4) top‑K neurons & KLs
    mlp_key = f"model.layers.{star}.mlp"
    act = cache[mlp_key][0].detach()  # (seq, hidden)
    meanseq = act.abs().mean(0)  # reduce seq dim here
    vals, idxs_t = torch.topk(meanseq, k=min(top_k, meanseq.numel()))
    idxs = idxs_t.tolist()
    pt_hooks = [
        base.get_submodule(f"model.layers.{star}.{c}").register_forward_hook(
            patch(f"model.layers.{star}.{c}", cache)
        )
        for c in ("self_attn", "mlp")
    ]
    kl_patched = kld(base_logits, base(**inp).logits[0, -1, :])
    abl = base.get_submodule(mlp_key).register_forward_hook(zero_many(idxs))
    kl_ablated = kld(base_logits, base(**inp).logits[0, -1, :])
    abl.remove()
    [h.remove() for h in pt_hooks]
    return {
        "star_layer": star,
        "top_neurons": idxs,
        "kl_patched": kl_patched,
        "kl_ablated": kl_ablated,
    }


# ╭────────────── Prompt loader (returns DataFrame) ╮


def load_df(args) -> pd.DataFrame:
    if args.benchmark == "crows":
        df = pd.read_csv(args.crows_url)
        if args.bias_type:
            df = df[df["bias_type"] == args.bias_type]
        # prefix_2patch_mbbq.py  – in load_df()
        if args.num_prompts != "all":
            k = int(args.num_prompts)
            good_ids = (df.groupby("pair_id")["bias_type"]
                  .apply(lambda s: {"_more", "_less"} <= {b[-5:] for b in s})
                  .pipe(lambda m: m[m].index))
            # 2) sample k of those ids
            sample_ids = good_ids.to_series().sample(
                n=min(k, len(good_ids)), random_state=42).tolist()
            df = df[df["pair_id"].isin(sample_ids)]
            return df

    df = pd.read_csv(
        args.mbbq_path, sep="\t" if args.mbbq_path.endswith(".tsv") else ","
    )
    if args.bias_type and "bias_type" in df.columns:
        df = df[df["bias_type"] == args.bias_type]
    if args.num_prompts != "all":
        k = int(args.num_prompts)

        # ── keep only pair_ids that have BOTH bias variants ──────────
        good_ids = (df.groupby("pair_id")["bias_type"]
                      .apply(lambda s: {"_more", "_less"} <= {b[-5:] for b in s})
                      .pipe(lambda m: m[m].index))
        if len(good_ids) == 0:
            sys.exit("❌ No pair_ids contain both _more and _less rows")

        # sample k of those ids (or fewer if the file is tiny)
        sample_ids = np.random.choice(good_ids,
                                      size=min(k, len(good_ids)),
                                      replace=False)
        df = df[df["pair_id"].isin(sample_ids)]
        print(f"📊 load_df: selected {len(sample_ids)} pair_ids "
              f"→ {len(df)} rows ({len(sample_ids)} _more + {len(sample_ids)} _less)")
    return df


# ╭──────── Alignment‑specific neuron discovery ╮


def scan_alignment_neurons(
    aligned,
    tok,
    harmful: List[str],
    normal: List[str],
    layers,
    *,
    sigma: float,
    min_precision: float,
    min_support: int,
) -> Set[Tuple[int, int]]:
    """Return set of (layer, neuron_idx) that fire specifically on harmful prompts."""
    device = next(aligned.parameters()).device
    L = aligned.config.num_hidden_layers
    layers = range(L) if layers == "all" else layers

    def collect(plist: List[str], tag: str):
        acc = {l: [] for l in layers}
        for p in tqdm(plist, desc="🔬 Align Scan", unit="prompt"):
            cache = {}
            hooks = [
                aligned.get_submodule(f"model.layers.{l}.mlp").register_forward_hook(
                    cap(f"model.layers.{l}.mlp", cache)
                )
                for l in layers
            ]
            # Process input with padding
            inputs = tok(p, return_tensors="pt", padding=True, truncation=True).to(
                device
            )
            aligned(**inputs)
            for hh in hooks:
                hh.remove()

            # Get activation and reduce to per-neuron values
            for l in layers:
                act = cache[f"model.layers.{l}.mlp"][0]  # (seq_len, hidden_dim)
                # Average over sequence dimension to get (hidden_dim,)
                neuron_acts = act.abs().mean(dim=0)
                acc[l].append(neuron_acts)

        # Stack the averaged activations
        if len(plist) == 0:
            print(f"⚠️  ALIGN-SCAN: '{tag}' list is empty – skipping scan")
            return None
        for l in layers:
            if len(acc[l]) == 0:
                print(f"⚠️  ALIGN-SCAN: layer {l} collected 0 tensors in '{tag}'")
        return {l: torch.stack(acc[l], 0) for l in layers}

    # Rest of the function remains the same
    print("🔬 collecting normal activations …", flush=True)
    print("🔬 collecting harmful activations …", flush=True)
    norm = collect(normal,  "normal")
    harm = collect(harmful, "harmful")

    # If either list was empty, bail out early
    if norm is None or harm is None:
        print("⚠️  Alignment scan skipped – insufficient prompts")
        return set()
    selected: Set[Tuple[int, int]] = set()
    for l in layers:
        nm = norm[l]  # (N_norm, hidden)
        hm = harm[l]  # (N_harm, hidden)
        mu = nm.mean(0)
        sd = nm.std(0) + 1e-6
        thr = mu + sigma * sd
        fires_h = (hm > thr).float().sum(0)
        fires_n = (nm > thr).float().sum(0)
        precision = fires_h / (fires_h + fires_n + 1e-6)
        mask = (precision >= min_precision) & (fires_h >= min_support)
        for idx in torch.nonzero(mask).squeeze(1).tolist():
            selected.add((l, idx))
    print("\n=== Alignment‑specific neurons ===")
    for l, idx in sorted(selected):
        print(f"Layer {l:>2} neuron {idx:>4}")
    return selected


# ╭──────────────────────────── Main ╮


def main():
    args = get_cli()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("❌ CUDA not available; use --device cpu")
    if args.benchmark == "mbbq" and not args.mbbq_path:
        sys.exit("❌ --mbbq_path required")
    warnings.filterwarnings("ignore")

    # Initialize variables
    agg = []  # Initialize agg to store results
    star_hist = collections.Counter()
    neuron_hist = collections.Counter()

    # Model loading
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("📥 loading models …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.aligned_id, token=args.hf_token)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=cfg,
        token=args.hf_token,
    ).to(args.device)
    aligned = AutoModelForCausalLM.from_pretrained(
        args.aligned_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=cfg,
        token=args.hf_token,
    ).to(args.device)
    layer_spec = parse_layers(args.layers, base.config.num_hidden_layers)
    df = load_df(args)
    if df.empty:
        sys.exit("❌ no prompts after filter")

    # Bias scan
    rows = df.to_dict("records")
    make_prompt = (
        (lambda r: r["sent_more"])
        if args.benchmark == "crows"
        else (lambda r: f"{r['context']}\n{r['question']}")
    )
    print(f"🏃 bias scan on {len(rows)} prompts …", flush=True)

    for row in tqdm(rows, desc="▶ Bias Scan", unit="prompt"):
        prompt = make_prompt(row)
        r = run_experiment(base, aligned, tok, prompt, layer_spec, args.top_k)
        r.update(
            {
                "prompt": prompt,
                "bias_type": row["bias_type"],
                "example_id": row.get("example_id"),
                "pair_id": row.get("pair_id"),  #  ← NEW
                "lang": row.get("lang"),  #  ← NEW (optional)
            }
        )

        agg.append(r)  # Append results to agg
        star_hist[r["star_layer"]] += 1
        for n in r["top_neurons"]:
            neuron_hist[(r["star_layer"], n)] += 1
    bias_set = set(neuron_hist.keys())

    # Alignment scan
    alignment_set: Set[Tuple[int, int]] = set()
    if args.harmful_txt and args.normal_txt:
        harmful = Path(args.harmful_txt).read_text().splitlines()
        normal = Path(args.normal_txt).read_text().splitlines()
        
        print(f"🔧 Alignment scan will use {len(harmful)} harmful / {len(normal)} normal prompts")
        alignment_set = scan_alignment_neurons(
            aligned,
            tok,
            harmful,
            normal,
            layer_spec,
            sigma=args.sigma,
            min_precision=args.min_precision,
            min_support=args.min_support,
        )
    elif args.auto_pair_align:
        if args.benchmark == "crows":
            harmful = df["sent_more"].tolist()
            normal = df["sent_less"].tolist()
        else:
            more = df[df["bias_type"].str.endswith("_more")]
            less = df[df["bias_type"].str.endswith("_less")]
            harmful = (more["context"] + "\n" + more["question"]).tolist()
            normal = (less["context"] + "\n" + less["question"]).tolist()
        print(f"🔧 auto-pair alignment: {len(harmful)} harmful | {len(normal)} normal")
        alignment_set = scan_alignment_neurons(
            aligned,
            tok,
            harmful,
            normal,
            layer_spec,
            sigma=args.sigma,
            min_precision=args.min_precision,
            min_support=args.min_support,
        )
    elif args.harmful_txt or args.normal_txt:
        print("⚠️  need both harmful & normal TXT, or add --auto_pair_align")

    # Intersection
    inter = bias_set & alignment_set if alignment_set else set()
    if inter:
        print("\n=== Intersection (bias ∩ alignment) ===")
        for l, n in sorted(inter):
            print(f"Layer {l:>2} neuron {n:>4}")

    # Pair-gap
    if args.pair_gap:
        pg = args.pair_gap
        df_out = pd.DataFrame(agg)
        
        key = (
            "pair_id"
            if "pair_id" in df_out.columns
            else "example_id" if "example_id" in df_out.columns else "prompt"
        )
        more = df_out[df_out["bias_type"] == f"{pg}_more"].set_index(key)
        less = df_out[df_out["bias_type"] == f"{pg}_less"].set_index(key)
        both = more.join(less, lsuffix="_more", rsuffix="_less", how="inner")
        print("\n🔎 Pair-gap debug: columns =", df_out.columns.tolist())
        print("  rows _more:", len(more), "| rows _less:", len(less))
        print("  join key  :", key)
        print("  first 3 keys in more:", more.index[:3].tolist())
        print("  first 3 keys in less:", less.index[:3].tolist())
        print(f"  joined rows: {len(both)}  (expected = {df_out['pair_id'].nunique()})")
        if not both.empty:
            both["kl_gap"] = both["kl_patched_more"] - both["kl_patched_less"]
            mean_gap = both["kl_gap"].mean()
            print(f"\n=== Pair-gap summary ({pg}) ===\nmean ΔKL = {mean_gap:+.4f}")
            if args.save_csv:
                out_dir = Path("patch_out")
                out_dir.mkdir(exist_ok=True)   # Ensure directory exists
                both.to_csv(out_dir / f"pairgap_{pg}.csv")
        else:
            print(f"⚠️ pair_gap '{pg}' has no matched pairs")

    # CSV export
    if args.save_csv:
        out = Path("patch_out")
        out.mkdir(exist_ok=True)
        pd.DataFrame(agg).to_csv(out / "prompt_results.csv", index=False)
        pd.DataFrame(star_hist.most_common(), columns=["layer", "count"]).to_csv(
            out / "layer_stats.csv", index=False
        )
        pd.DataFrame(
            [(l, n, c) for (l, n), c in neuron_hist.items()],
            columns=["layer", "neuron_idx", "count"],
        ).to_csv(out / "neuron_stats.csv", index=False)
        if alignment_set:
            pd.DataFrame(list(alignment_set), columns=["layer", "neuron_idx"]).to_csv(
                out / "alignment_neurons.csv", index=False
            )
        if inter:
            pd.DataFrame(list(inter), columns=["layer", "neuron_idx"]).to_csv(
                out / "intersection.csv", index=False
            )
        print("📝 CSVs saved to", out)


if __name__ == "__main__":
    main()
