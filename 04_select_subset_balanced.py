# 04_select_subset_balanced.py — pick 40 per model (20 kept + 20 rejected), enforce len ≤ 400
# Usage: python 04_select_subset_balanced.py
# Outputs: data/subset_colabfold.fasta, results/subset_manifest.csv

import random
from pathlib import Path
from Bio import SeqIO
import pandas as pd

random.seed(42)

# FASTAs produced by 03_filter_compare.py
PROL_KEEP = Path("data/prollama_kept.fasta")
PROL_REJ  = Path("data/prollama_rejected.fasta")
PGPT_KEEP = Path("data/protgpt2_kept.fasta")
PGPT_REJ  = Path("data/protgpt2_rejected.fasta")

OUT_FASTA = Path("data/subset_colabfold.fasta")
OUT_CSV   = Path("results/subset_manifest.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)

MAX_LEN = 400   
TARGET_PER_BUCKET = 20

def read_fasta_to_dict(path: Path, max_len: int) -> dict:
    """Return {id: seq} filtered to sequences with len ≤ max_len."""
    d = {}
    if path.exists():
        total = 0
        kept  = 0
        for r in SeqIO.parse(str(path), "fasta"):
            total += 1
            seq = str(r.seq).strip().upper()
            if len(seq) <= max_len:
                d[r.id] = seq
                kept += 1
        print(f"[load] {path.name}: kept {kept}/{total} (len ≤ {max_len})")
    else:
        print(f"[warn] missing FASTA: {path}")
    return d

def sample_ids(pool: dict, k: int):
    ids = list(pool.keys())
    random.shuffle(ids)
    return ids[:min(k, len(ids))]

def main():
    # load kept/rejected per model with length filter
    prol_keep = read_fasta_to_dict(PROL_KEEP, MAX_LEN)
    prol_rej  = read_fasta_to_dict(PROL_REJ,  MAX_LEN)
    pgpt_keep = read_fasta_to_dict(PGPT_KEEP, MAX_LEN)
    pgpt_rej  = read_fasta_to_dict(PGPT_REJ,  MAX_LEN)

    if not all([prol_keep or prol_rej, pgpt_keep or pgpt_rej]):
        raise SystemExit("Run 03_filter_compare.py first to create kept/rejected FASTAs in data/")

    # sample up to TARGET_PER_BUCKET from each bucket (after len filter)
    picks = []
    picks += [(rid, "ProLLaMA", True)  for rid in sample_ids(prol_keep, TARGET_PER_BUCKET)]
    picks += [(rid, "ProLLaMA", False) for rid in sample_ids(prol_rej,  TARGET_PER_BUCKET)]
    picks += [(rid, "ProtGPT2", True)  for rid in sample_ids(pgpt_keep, TARGET_PER_BUCKET)]
    picks += [(rid, "ProtGPT2", False) for rid in sample_ids(pgpt_rej,  TARGET_PER_BUCKET)]

    # build a seq lookup across all four (already len-filtered)
    seqs = {}
    seqs.update(prol_keep); seqs.update(prol_rej)
    seqs.update(pgpt_keep); seqs.update(pgpt_rej)

    # write subset FASTA + manifest
    with open(OUT_FASTA, "w") as f:
        for rid, model, keep in picks:
            s = seqs.get(rid)
            if s:
                f.write(f">{rid}\n{s}\n")

    df = pd.DataFrame(picks, columns=["id","model","keep"])
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {OUT_FASTA} (n={len(df)}) and {OUT_CSV}")
    counts = df.groupby(["model","keep"]).size().rename("n")
    print(counts)

if __name__ == "__main__":
    main()
