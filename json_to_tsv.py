#!/usr/bin/env python
# save as jsonl_to_tsv.py  (Google Colab に置く)

"""
Convert one or more MBBQ-style *.jsonl files into a single TSV
with columns: context    question    bias_type

Usage (Colab):
!python jsonl_to_tsv.py \
    --inputs Age_control_es.jsonl Age_control_nl.jsonl Age_control_tr.jsonl \
    --bias_type age \
    --output age.tsv
"""

from __future__ import annotations
import argparse, csv, json, pathlib, sys, urllib.request, tempfile

DEFAULT_BASE = (
    "https://raw.githubusercontent.com/Veranep/MBBQ/main/data/"
)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="local path / URL / file-name (space-separated)")
    p.add_argument("--bias_type", default="unknown",
                   help="value for bias_type column (default unknown)")
    p.add_argument("--output", default="out.tsv",
                   help="output TSV path (default out.tsv)")
    p.add_argument("--base_url", default=DEFAULT_BASE,
                   help="prefix used when input token has no slash")
    return p.parse_args()

def ensure_local(token: str, tmp_dir: pathlib.Path, base_url: str) -> pathlib.Path:
    if token.startswith(("http://", "https://")):        # full URL
        url, fname = token, tmp_dir / token.split("/")[-1]
    elif "/" in token or pathlib.Path(token).exists():   # local path
        return pathlib.Path(token)
    else:                                                # bare file-name
        url = base_url + token
        fname = tmp_dir / token
    if not fname.exists():
        print(f"⬇️  fetching {url}")
        urllib.request.urlretrieve(url, fname)
    return fname

def iter_jsonl(fp: pathlib.Path):
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
                
import re

def extract(ex: dict[str, str], lang: str):
    """Extract context, question, and example_id from JSONL."""
    ctx = ex.get("context") or ex.get("premise")
    q = ex.get("question") or ex.get("hypothesis")
    if not (ctx and q):
        raise ValueError("missing context/question keys")
    return ctx, q, ex.get("id") or ex.get("example_id"), lang

def main():
    args = parse()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="mbbq_dl_"))
    rows = []
    for token in args.inputs:
        fp = ensure_local(token, tmp, args.base_url)
        lang = re.search(r"_([a-z]{2})\.jsonl$", fp.name).group(1)
        for ex in iter_jsonl(fp):
            ctx, q, example_id, lang = extract(ex, lang)
            is_control = "_control_" in fp.name.lower()
            suffix = "_less" if is_control else "_more"
            pair_id = f"{example_id}_{lang}"
            rows.append({
                 "context": ctx,
                 "question": q,
                 "bias_type": f"{args.bias_type}{suffix}",
                "example_id": example_id,
                "lang": lang,
                "pair_id": pair_id
             })

    out = pathlib.Path(args.output if args.output.endswith(".tsv") else f"{args.output}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
             f,
             fieldnames=["context", "question", "bias_type",
                         "example_id", "lang", "pair_id"],
            delimiter="\t"
        )
        w.writeheader()
        w.writerows(rows)
    print(f"✅ wrote {len(rows):,} rows → {out}")

if __name__ == "__main__":
    main()