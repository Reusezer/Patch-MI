#!/usr/bin/env python
"""bias_probe_fast_cli.py ― top-K ニューロンだけを因果テストする高速バイアス解析

アルゴリズム（プロンプトごと）
 1) ベースモデルで 1 回推論し、Δ₀ := logP(biased)−logP(anti) を取得。
 2) 指定層 ℓ の MLP 出力から |h| の平均が大きいニューロン上位 K 個を取得。
 3) そのニューロン i に対して h_i ← +C と置換し再推論 → Δ_{ℓ,i} 、ゲイン g_{ℓ,i} = Δ_{ℓ,i}−Δ₀。
 4) g_{ℓ,i} 最大の層を ★ としてカウント。
---------------------------------------------------------------------------
出力
  bias_layer.csv         層 → star_count
  bias_neuron_stats.csv  (層, ニューロン) → hit_count & mean_gain
  
  CUDA_VISIBLE_DEVICES=0 \
python bias_probe_fast_cli.py \
  --bias_type race \
  --num_prompts 50 \
  --layers 5,10,15 \        # ← 解析層を絞ってさらに高速化
  --top_k 2 \               # 各層で上位 2 ニューロンだけテスト
  --C 10 \
  --report_top_layers 5 \
  --report_top_neurons 10 \
  --save_csv \
  --hf_token $HF_TOKEN
  
  
  --bias_type
CrowS-Pairs の bias_type 列（race, gender, religion …）でデータを絞り込みます。指定しなければ全カテゴリを対象にします。

--num_prompts
処理するプロンプト数を決めます。整数を渡せばその件数だけ、all と書けばファイル全行を使います。既定は 50 行。

--layers
解析対象の層を指定します。all なら 0 から最終層まで全部。単一数字ならその層だけ、カンマ区切りなら任意の集合（例 1,3,7）。省略しても all と同義です。

--top_k
各層で「平均絶対活性が大きいニューロン」を上位 K 個だけ選び、因果テストします。K を大きくすると精度は上がりますが推論回数も増えます。初期値は 1。

--C
強制発火させるときにそのニューロンへ代入する定数値です。活性を +C に置き換えた状態で再推論し、バイアス増幅度を測ります。デフォルトは 10.0。

--report_top_layers
実行終了後、バイアス増幅が最も多く観測された層（★層）を何位まで表示するかを決めます。標準出力にだけ影響し、CSV には無関係です。

--report_top_neurons
同様に「増幅度が大きいニューロン」を何件まで表示するかの指定です。これも画面出力用。

--save_csv
このフラグを付けると bias_layer.csv と bias_neuron_stats.csv を bias_probe_out/ フォルダへ書き出します。付けなければファイルは作りません。

--device
推論に使うデバイス。cuda, cuda:1, cpu などを指定します。既定値は cuda。GPU が無い環境では cpu に変えてください。

--hf_token
HuggingFace Hub でプライベートモデルを引き出す場合のアクセストークン。環境変数 HF_TOKEN に設定しておけば省略可能です。

--base_id
解析対象となる “Base” モデルの HuggingFace ID。デフォルトは meta-llama/Meta-Llama-3-8B ですが、好きな LLM に変更できます。

--aligned_id
互換性のため受け取りますが、このスクリプトでは使いません（無視されます）。

--crows_url
CrowS-Pairs データセットのパスまたは URL を上書き

"""

# ─────────── imports ─────────── #
import argparse, collections, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch, torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# CrowS デフォルト URL
CROWS_URL = ("https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
             "data/crows_pairs_anonymized.csv")

# ─────────── CLI ─────────── #
def parse_layers(arg: str, n: int) -> Sequence[int] | str:
    if arg.lower() == "all":
        return list(range(n))
    idxs = sorted({int(x) for x in arg.split(",") if x})
    if any(i < 0 or i >= n for i in idxs):
        raise argparse.ArgumentTypeError(f"layer idx {idxs} out of range 0..{n-1}")
    return idxs

def cli() -> argparse.Namespace:
    P = argparse.ArgumentParser()
    P.add_argument("--bias_type", default=None)
    P.add_argument("--num_prompts", default="50")           # N or "all"
    P.add_argument("--layers", default="all")
    P.add_argument("--top_k", type=int, default=1)
    P.add_argument("--C", type=float, default=10.0)
    P.add_argument("--report_top_layers", type=int, default=10)
    P.add_argument("--report_top_neurons", type=int, default=10)
    P.add_argument("--save_csv", action="store_true")
    P.add_argument("--device", default="cuda")
    P.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    P.add_argument("--base_id", default="meta-llama/Meta-Llama-3-8B")
    P.add_argument("--aligned_id")          # 互換のため受け取るが未使用
    P.add_argument("--crows_url", default=CROWS_URL)
    return P.parse_args()

# ─────────── util ─────────── #
@torch.no_grad()
def logprob(model, tok, text: str, device):
    T = tok(text, return_tensors="pt").to(device)
    logits = model(**T).logits[:, :-1]
    tgt = T.input_ids[:, 1:]
    ll = F.log_softmax(logits, -1).gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    return ll.sum().item()

def bias_score(model, tok, pair, device):
    a, b = pair
    return logprob(model,tok,b,device) - logprob(model,tok,a,device)

# ─────────── main scan ─────────── #
def scan(model, tok, prompts, layers, top_k, C, device):
    mlps = [model.model.layers[i].mlp for i in layers]
    H    = model.config.hidden_size
    layer_hits = collections.Counter()
    neuron_stats: Dict[Tuple[int,int], List[float]] = collections.defaultdict(list)
    act_cache: Dict[int, torch.Tensor] = {}      # ★ 各層の出力を貯める辞書

    dtype  = torch.bfloat16
    devtype= "cuda" if str(device).startswith("cuda") else "cpu"
    ac = lambda: torch.amp.autocast(device_type=devtype, dtype=dtype)

    for a,b in tqdm(prompts, unit="prompt"):
        with ac(): base = bias_score(model,tok,(a,b),device)
        best_l, best_gain = -1, -1e9

        # forward once to get activations
        toks = tok(b, return_tensors="pt").to(device)
        act_cache.clear()                         # ★ 毎プロンプトで空に戻す
        hooks = []
        for li, mlp in enumerate(mlps):
            def make_store(index):
                # forward 出力を辞書に保存するクロージャ
                return (lambda _, __, out,
                             cache=act_cache, idx=index:
                             cache.__setitem__(idx, out.detach()))
            hooks.append(mlp.register_forward_hook(make_store(li)))
        with torch.no_grad(), ac():
            _ = model(**toks)
        for h in hooks:
            h.remove()

        for li, mlp in enumerate(mlps):
            acts = act_cache[li]
            meanabs = acts[0].abs().mean(0)
            vals, idxs = torch.topk(meanabs, k=min(top_k, H))
            for n in idxs.tolist():
                # 2) そのニューロンだけ +C
                def h(_,__,out,idx=n,val=C): out[...,idx].fill_(val); return out
                hook = mlp.register_forward_hook(h)
                with ac():
                    gain = bias_score(model,tok,(a,b),device) - base
                hook.remove()

                neuron_stats[(layers[li], n)].append(gain)
                if gain > best_gain:
                    best_gain, best_l = gain, layers[li]
        layer_hits[best_l] += 1
    return layer_hits, neuron_stats

# ─────────── main() ─────────── #
def main():
    args = cli()
    warnings.filterwarnings("ignore")

    # model
    cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.base_id, token=args.hf_token)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_id, device_map="auto",
        torch_dtype=torch.bfloat16, quantization_config=cfg,
        token=args.hf_token).to(args.device).eval()

    # prompts
    df = pd.read_csv(args.crows_url)
    if args.bias_type: df = df[df["bias_type"]==args.bias_type]
    if args.num_prompts!="all": df = df.head(int(args.num_prompts))
    prompts = list(zip(df["sent_more"], df["sent_less"]))
    if not prompts: sys.exit("❌ no prompts")

    layer_sel = parse_layers(args.layers, model.config.num_hidden_layers)
    layer_hist, neuron_map = scan(model, tok, prompts,
                                  layer_sel, args.top_k, args.C, args.device)

    # print summary
    print("\n★ top layers ★")
    for l,c in layer_hist.most_common(args.report_top_layers):
        print(f"layer {l}: {c} prompts")

    print("\n★ top neurons ★")
    top = sorted(neuron_map.items(), key=lambda kv: sum(kv[1])/len(kv[1]), reverse=True)
    for (l,n),g in top[:args.report_top_neurons]:
        print(f"layer {l}, neuron {n}: mean Δ {sum(g)/len(g):.3f} | hits {len(g)}")

    # CSV
    if args.save_csv:
        out = Path("bias_probe_out"); out.mkdir(exist_ok=True)
        pd.DataFrame(layer_hist.items(), columns=["layer","star_count"]).to_csv(out/"bias_layer.csv",index=False)
        pd.DataFrame(
            [{"layer":l,"neuron_idx":n,"hit_count":len(g),"mean_gain":sum(g)/len(g)}
             for (l,n),g in neuron_map.items()]
        ).to_csv(out/"bias_neuron_stats.csv",index=False)
        print(f"\n📝 CSVs written to {out}/")

if __name__ == "__main__":
    main()
