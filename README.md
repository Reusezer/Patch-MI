#  **PATCH-MI**

*Paired-Activation Tensor-Contrast for C**H**armonisation*

A lightweight CLI that **scans, patches, and probes neuron-level bias × alignment effects** in autoregressive LLMs.  It combines the paired-sentence idea of **CrowS-Pairs** with layer-wise *activation patching*, then reports:

* star layers that erase bias when patched
* top-K causal neurons in those layers
* gap statistics between paired prompts (Δ KL)
* alignment-specific “safety” neurons (optional)

---

## 1 . Installation

```bash
# Python ≥ 3.10, CUDA 11.x
conda create -n patchpairs python=3.10
conda activate patchpairs

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy tqdm bitsandbytes
```

Set your HF token once:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

---

## 2 . Quick start

### 2-A CrowS-Pairs

```bash
python prefix_patch_cli.py \
  --benchmark crows \
  --bias_type gender \
  --num_prompts 300 \
  --pair_gap gender \
  --save_csv
```

### 2-B MBBQ TSV

```bash
python prefix_patch_cli.py \
  --benchmark mbbq \
  --mbbq_path data/age_pair.tsv \
  --bias_type age \
  --pair_gap age \
  --top_k 5 --layers all \
  --save_csv
```

### 2-C Alignment safety scan

```bash
python prefix_patch_cli.py \
  --harmful_txt toxic.txt \
  --normal_txt benign.txt  \
  --min_precision 0.95 --sigma 2.5
```

Or derive the two lists automatically:

```bash
python prefix_patch_cli.py --auto_pair_align
```

---

## 3 . Output files (`patch_out/`)

| File                    | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `prompt_results.csv`    | per-prompt star-layer, Δ KL, and top-K neurons |
| `layer_stats.csv`       | histogram of star-layer frequency              |
| `neuron_stats.csv`      | histogram of individual causal neurons         |
| `pairgap_<bias>.csv`    | joined \_more/\_less rows + `kl_gap` column    |
| `alignment_neurons.csv` | layer/index pairs firing on harmful prompts    |
| `intersection.csv`      | neurons common to bias **and** alignment scans |

---

## 4 . Key CLI flags

| Flag                | Meaning                                             |                                               |
| ------------------- | --------------------------------------------------- | --------------------------------------------- |
| \`--layers "all"    | 0,7,15\`                                            | scan cumulative layers (all) or explicit list |
| `--top_k 3`         | how many high-impact neurons to keep per star layer |                                               |
| `--num_prompts 500` | cap dataset size for quick experiments              |                                               |
| `--pair_gap age`    | compute Δ KL between `age_more` and `age_less` rows |                                               |
| `--device cpu`      | run on CPU when CUDA isn’t available                |                                               |

---

## 5 . Program flow

```text
               ┌────────────┐  cache clean activations
 prompt  ─▶    │ aligned LM │──┐
               └────────────┘  │
                               │patch
               ┌────────────┐◀─┘
               │  base LM   │   measure KL  → star layer → top-K neurons
               └────────────┘
```



> **PATCH-Pairs**: because sometimes the quickest way to expose a bias neuron is to *patch it out and see what falls apart*.
