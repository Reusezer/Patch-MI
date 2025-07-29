#!/usr/bin/env python
"""prefix_patch_cli.py — De‑biasing neuron scanner (argument‑driven version)

This is the *argument‑based* rewrite of the original **prefix_patch_v3.py**.
It identifies Aligned‑model neurons that causally *reduce* social bias when
patched into a Base model.

-----------------------------------------------------------------------
❑ Basic usage (GPU recommended)
-----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python prefix_patch_cli.py \
  --bias_type race \
  --num_prompts 50 \
  --layers all \
  --top_k 1 \
  --report_top_layers 10 \
  --report_top_neurons 10 \
  --save_csv \
  --hf_token $HF_TOKEN

Arguments:
  --bias_type               CrowS‑Pairs bias category (race, gender, ...). default: all
  --num_prompts             #prompts or "all". default: 50
  --layers                  "all" | "5" | "1,3,7". default: all
  --top_k                   top‑K neurons per prompt for ablation. default: 1
  --report_top_layers       print N layers with highest star_count. default: 10
  --report_top_neurons      print N neurons with most hits. default: 10
  --save_csv                store prompt_results.csv, layer_stats.csv, neuron_stats.csv
  --hf_token                HuggingFace access token (or env HF_TOKEN)
  --device                  torch device string (cuda, cuda:1, cpu). default: cuda
  --base_id / --aligned_id  override HF model IDs (defaults: Llama‑3‑8B and -Instruct)
  --crows_url               dataset location. default: NYU CrowS‑Pairs CSV

The algorithm & metrics are unchanged from prefix_patch_v3.py; only the
interaction model has shifted from `input()` prompts to `argparse` flags, so
it can be scripted or batched in a cluster.
"""

import argparse, collections, importlib, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch, torch.nn.functional as F
import pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─────────────────────────── bitsandbytes hot‑patch (Colab) ──────────────── #
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

# ╭───────────────────────────── CLI parsing ─────────────────────────────╮ #

def parse_layers(arg: str, n_layers: int) -> Sequence[int] | str:
    if arg.strip().lower() == "all":
        return "all"
    idxs = sorted({int(x) for x in arg.split(",") if x})
    if any(i < 0 or i >= n_layers for i in idxs):
        raise argparse.ArgumentTypeError(f"layer index {idxs} out of range 0..{n_layers-1}")
    return idxs


def get_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="De‑bias neuron scanner (activation patching)")
    p.add_argument("--bias_type", default=None, help="CrowS‑Pairs bias_type filter (e.g. race)")
    p.add_argument("--num_prompts", default="50", help="N or 'all' (default 50)")
    p.add_argument("--layers", default="all", help="layer spec: all | k | i,j,k (default all)")
    p.add_argument("--top_k", type=int, default=1, help="top‑K neurons per prompt (default 1)")
    p.add_argument("--report_top_layers", type=int, default=10, help="print N top layers")
    p.add_argument("--report_top_neurons", type=int, default=10, help="print N top neurons")
    p.add_argument("--save_csv", action="store_true", help="save CSV outputs")
    p.add_argument("--device", default="cuda", help="torch device (default cuda)")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"), help="HF access token")
    p.add_argument("--base_id", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--aligned_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--crows_url", default=CROWS_URL_DEFAULT)
    return p.parse_args()

# ╭───────────────────────────── Hooks ───────────────────────────────────╮ #

def cap(name: str, cache: Dict[str, torch.Tensor]):
    def _h(_, __, out):
        cache[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return _h


def patch(name: str, src: Dict[str, torch.Tensor]):
    def _h(_, __, out):
        tgt = src[name].to(out[0].device if isinstance(out, tuple) else out.device)
        return (tgt,) + out[1:] if isinstance(out, tuple) else tgt
    return _h


def zero_many(idxs: List[int]):
    idxs_t = torch.as_tensor(idxs)
    def _h(_, __, out):
        x = out[0] if isinstance(out, tuple) else out
        x[..., idxs_t] = 0.0
        return (x,) if isinstance(out, tuple) else x
    return _h

# ╭──────────────────────────── Utilities ───────────────────────────────╮ #

def kld(p: torch.Tensor, q: torch.Tensor) -> float:
    p = F.softmax(p, -1)
    ql = F.log_softmax(q, -1)
    return F.kl_div(ql, p, reduction="sum", log_target=False).item()


@torch.inference_mode()
def run_experiment(base, aligned, tok, prompt: str, layer_spec, top_k: int):
    """Return metrics for one prompt (unchanged from v3 logic)."""
    device = next(base.parameters()).device
    inp = tok(prompt, return_tensors="pt").to(device)
    L = base.config.num_hidden_layers

    # 1) cache Aligned activations
    cache, tmp = {}, []
    for i in range(L):
        for c in ("self_attn", "mlp"):
            n = f"model.layers.{i}.{c}"
            tmp.append(aligned.get_submodule(n).register_forward_hook(cap(n, cache)))
    aligned(**inp)
    for h in tmp:
        h.remove()

    base_logits = base(**inp).logits[0, -1, :]

    # 2) layer sets (cumulative if 'all')
    layer_sets = (
        [(tuple(range(k + 1)) if layer_spec == "all" else tuple(layer_spec))]
        if layer_spec != "all"
        else [tuple(range(k + 1)) for k in range(L)]
    )

    results = []
    for ls in layer_sets:
        hooks = []
        for l in ls:
            for c in ("self_attn", "mlp"):
                n = f"model.layers.{l}.{c}"
                hooks.append(base.get_submodule(n).register_forward_hook(patch(n, cache)))
        kl = kld(base_logits, base(**inp).logits[0, -1, :])
        results.append({"layers": ls, "KL": kl})
        for h in hooks:
            h.remove()

    # 3) star layer selection (identical to v3)
    if layer_spec == "all":
        deltas = [results[i - 1]["KL"] - r["KL"] if i else 0 for i, r in enumerate(results)]
        star = deltas.index(max(deltas[1:]))
    else:
        star = (
            results[0]["layers"][0]
            if len(results) == 1
            else max(results, key=lambda r: r["KL"])["layers"][0]
        )

    # 4) extract top‑K active neurons in star layer
    mlp_key = f"model.layers.{star}.mlp"
    act = cache[mlp_key][0]
    meanabs = act.abs().mean(0)
    vals, idxs_t = torch.topk(meanabs, k=min(top_k, meanabs.numel()))
    idxs, vals = idxs_t.tolist(), vals.tolist()

    # 5) KL of patched vs ablated
    pt_hooks = [
        base.get_submodule(f"model.layers.{star}.{c}")
        .register_forward_hook(patch(f"model.layers.{star}.{c}", cache))
        for c in ("self_attn", "mlp")
    ]
    kl_patched = kld(base_logits, base(**inp).logits[0, -1, :])

    z = base.get_submodule(mlp_key).register_forward_hook(zero_many(idxs))
    kl_ablated = kld(base_logits, base(**inp).logits[0, -1, :])
    z.remove()
    for h in pt_hooks:
        h.remove()

    return {
        "prompt": prompt,
        "star_layer": star,
        "top_neurons": idxs,
        "top_neuron_scores": vals,
        "kl_base": 0.0,
        "kl_patched": kl_patched,
        "kl_ablated": kl_ablated,
    }

# ╭────────────────────────── Prompt loading ─────────────────────────────╮ #

def load_prompts(url: str, bias_type: str | None, n: str):
    df = pd.read_csv(url)
    if bias_type:
        df = df[df["bias_type"] == bias_type]
    if n != "all":
        df = df.head(int(n))
    return df["sent_more"].tolist()

# ╭─────────────────────────── Main routine ──────────────────────────────╮ #

def main():
    args = get_cli()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        sys.exit("❌ CUDA not available. Use --device cpu or enable GPU.")
    warnings.filterwarnings("ignore")

    # models
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("📥 Loading models …", flush=True)
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

    L = base.config.num_hidden_layers
    layer_spec = parse_layers(args.layers, L)

    prompts = load_prompts(args.crows_url, args.bias_type, args.num_prompts)
    if not prompts:
        sys.exit("❌ No prompts match that filter.")
    print(f"🏃 Processing {len(prompts)} prompts …", flush=True)

    agg_rows = []
    star_hist: collections.Counter[int] = collections.Counter()
    neuron_hist: collections.Counter[Tuple[int, int]] = collections.Counter()

    for i, p in enumerate(prompts, 1):
        row = run_experiment(base, aligned, tok, p, layer_spec, args.top_k)
        agg_rows.append(row)
        star_hist[row["star_layer"]] += 1
        for n_idx in row["top_neurons"]:
            neuron_hist[(row["star_layer"], n_idx)] += 1
        if args.num_prompts != "all":
            print(
                f"#{i}/{len(prompts)} | star {row['star_layer']} | KL patch {row['kl_patched']:.3f} | KL abl {row['kl_ablated']:.3f}")
        elif i % 20 == 0 or i == len(prompts):
            print(f"… {i}/{len(prompts)} done", flush=True)

    df = pd.DataFrame(agg_rows)
    mean_patch, mean_abl = df["kl_patched"].mean(), df["kl_ablated"].mean()
    print("\n=== Mean KL ===")
    print(f"patched: {mean_patch:.3f} | ablated: {mean_abl:.3f}")

    top_layers = star_hist.most_common(args.report_top_layers)
    print("\n★ Top layers ★")
    for l, c in top_layers:
        print(f"Layer {l}: {c} prompts")

    top_neurons = [((l,n),c) for (l,n),c in neuron_hist.items() if l in dict(top_layers)]
    top_neurons.sort(key=lambda x: x[1], reverse=True)
    top_neurons = top_neurons[: args.report_top_neurons]

    print("\n★ Top neurons ★")
    for (l, n), c in top_neurons:
        print(f"Layer {l}, neuron {n}: {c} hits")

    if args.save_csv:
        outdir = Path("patch_out")
        outdir.mkdir(exist_ok=True)
        df.to_csv(outdir / "prompt_results.csv", index=False)
        pd.DataFrame(top_layers, columns=["layer", "star_count"]).to_csv(outdir / "layer_stats.csv", index=False)
        pd.DataFrame([(l,n,c) for ((l,n),c) in top_neurons], columns=["layer","neuron_idx","count"]).to_csv(outdir / "neuron_stats.csv", index=False)
        print(f"\n📝 CSVs written to {outdir}/")


if __name__ == "__main__":
    main()
