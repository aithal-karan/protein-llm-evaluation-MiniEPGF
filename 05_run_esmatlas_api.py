#!/usr/bin/env python3
"""
Folds sequences via the public ESM Atlas API and computes mean pLDDT.

Inputs:
  - FASTA with one or more single-chain protein sequences (default: data/subset_colabfold.fasta)

Outputs:
  - PDB files in results/esmatlas_out/<id>.pdb
  - results/pLDDT.csv with columns: id,pLDDT

Usage:
  python 05_run_esmatlas_api.py \
      --fasta data/subset_colabfold.fasta \
      --out-dir results/esmatlas_out \
      --csv results/pLDDT.csv \
      --sleep 0.5 --retries 4 --timeout 180
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys
import requests
from Bio import SeqIO
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    def tqdm(x, **k): return x

API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fold sequences with ESM Atlas API (ESMFold) and compute mean pLDDT.")
    p.add_argument("--fasta", type=str, default="data/subset_colabfold_2.fasta", help="Input FASTA path.")
    p.add_argument("--out-dir", type=str, default="results/esmatlas_out_2", help="Directory to write PDBs.")
    p.add_argument("--csv", type=str, default="results/pLDDT_2.csv", help="Output CSV (id,pLDDT).")
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests (rate limiting).")
    p.add_argument("--retries", type=int, default=4, help="Max retries per sequence on HTTP failure.")
    p.add_argument("--timeout", type=int, default=180, help="Request timeout (seconds).")
    return p.parse_args()

def mean_plddt_from_pdb_text(pdb_text: str) -> float:
    """
    pLDDT is stored in the B-factor column (PDB columns 61–66 in ATOM records).
    We parse it robustly and return the mean across residues.
    """
    vals = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        # Standard fixed-width columns: B-factor at positions 61–66 (1-based)
        bf_str = line[60:66].strip()
        if not bf_str:
            # fallback: try splitting by whitespace and reading last column
            parts = line.split()
            if parts:
                bf_str = parts[-2] if len(parts) >= 2 else parts[-1]
        try:
            vals.append(float(bf_str))
        except Exception:
            pass
    return float(sum(vals) / len(vals)) if vals else 0.0

def fold_one(seq: str, timeout: int, retries: int) -> str:
    """
    POST the raw AA sequence to the ESM Atlas endpoint.
    Returns the PDB text on success. Raises on repeated failure.
    """
    backoff = 2.0
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(API_URL, data=seq.encode("utf-8"), timeout=timeout)
            if r.ok and r.text.startswith("HEADER"):
                return r.text
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
        # backoff & retry
        time.sleep(backoff)
        backoff *= 1.7
    raise RuntimeError(f"ESM Atlas API failed after {retries} retries: {last_err}")

def main():
    args = parse_args()
    fasta = Path(args.fasta)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv); csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not fasta.exists():
        print(f"[error] FASTA not found: {fasta}", file=sys.stderr)
        sys.exit(1)

    records = list(SeqIO.parse(str(fasta), "fasta"))
    if not records:
        print(f"[error] No sequences found in {fasta}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for rec in tqdm(records, desc="[ESM Atlas] folding", ncols=100):
        seq_id = rec.id
        seq = str(rec.seq).strip().upper()
        # sanitize to 20 AA letters
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        seq = "".join([a for a in seq if a in allowed])
        if len(seq) == 0:
            print(f"[warn] {seq_id}: empty/invalid sequence after cleaning; skipping")
            continue

        try:
            pdb_text = fold_one(seq=seq, timeout=args.timeout, retries=args.retries)
            # save PDB
            (out_dir / f"{seq_id}.pdb").write_text(pdb_text)
            # compute mean pLDDT
            mp = mean_plddt_from_pdb_text(pdb_text)
            rows.append({"id": seq_id, "pLDDT": round(mp, 3)})
        except Exception as e:
            print(f"[warn] {seq_id}: folding failed: {e}", file=sys.stderr)

        # polite rate limiting
        time.sleep(max(0.0, args.sleep))

    if not rows:
        print("[error] No successful folds; nothing to write.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(csv_path, index=False)
    print(f">> Wrote {csv_path} (n={len(df)}) and PDBs in {out_dir}")

if __name__ == "__main__":
    main()
