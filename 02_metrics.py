# 02_metrics.py — compute biophysical metrics with progress bars + summaries
# Usage (defaults if files exist):
#   python 02_metrics.py
# Or specify your own FASTAs (pairs of path:tag):
#   python 02_metrics.py --pairs data/gen_raw.fasta:ProLLaMA data/gen_raw_protgpt2.fasta:ProtGPT2
# Outputs:
#   results/metrics_prollama_raw.csv
#   results/metrics_protgpt2_raw.csv
#   results/metrics_raw.csv            (
#   prints a small summary at the end

import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

def aa_entropy(seq: str) -> float:
    # Shannon entropy over AA symbols
    freqs = [seq.count(a) / len(seq) for a in set(seq)]
    return -sum(f * np.log2(f) for f in freqs if f > 0)

def low_complexity_fraction(seq: str, window: int = 10, threshold: float = 0.8) -> float:
    # fraction of windows where one AA dominates (> threshold)
    n = len(seq)
    if n < window: return 0.0
    dom = 0
    for i in range(n - window + 1):
        w = seq[i:i + window]
        if max(w.count(a) / window for a in set(w)) > threshold:
            dom += 1
    return dom / n

def max_homopolymer(seq: str) -> int:
    # longest run of the same AA
    runs = re.findall(r"((\w)\2+)", seq)
    return max((len(r[0]) for r in runs), default=1)

def helix_sheet_ratio(seq: str) -> float:
    # very rough proxy of helix vs sheet propensity
    helix = sum(seq.count(a) for a in "ALMQEKRH")
    sheet = sum(seq.count(a) for a in "VIYCWF")
    return helix / max(1, sheet)

def clean_to_aa20(seq: str) -> str:
    seq = re.sub(r"[^A-Z]", "", seq.upper())
    return "".join(ch for ch in seq if ch in AA20)

def analyze_seq(seq: str) -> dict:
    pa = ProteinAnalysis(seq)
    return {
        "length": len(seq),
        "aa_entropy": aa_entropy(seq),
        "instability": pa.instability_index(),
        "gravy": pa.gravy(),
        "isoelectric_point": pa.isoelectric_point(),
        "low_complex": low_complexity_fraction(seq),
        "max_homopolymer": max_homopolymer(seq),
        "helix_sheet": helix_sheet_ratio(seq),
    }

def process_fasta(path: Path, tag: str) -> pd.DataFrame:

    total = sum(1 for _ in SeqIO.parse(str(path), "fasta"))
    rows = []
    kept = 0
    for rec in tqdm(SeqIO.parse(str(path), "fasta"), total=total, ncols=100, desc=f"[metrics:{tag}]"):
        raw = str(rec.seq)
        seq = clean_to_aa20(raw)
        if len(seq) < 50:
            continue
        d = analyze_seq(seq)
        d.update({"id": rec.id, "model": tag})
        rows.append(d)
        kept += 1
    df = pd.DataFrame(rows)
    print(f"  -> {tag}: {kept} / {total} sequences analyzed (len≥50, AA20-only).")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        nargs="*",
        help="Pairs of input:tag (e.g., data/gen_raw.fasta:ProLLaMA data/gen_raw_protgpt2.fasta:ProtGPT2)",
    )
    parser.add_argument("--outdir", default="results_2", help="Output directory for CSVs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Defaults if not provided
    pairs = []
    if args.pairs:
        for p in args.pairs:
            if ":" not in p:
                raise SystemExit(f"Bad --pairs entry (expected path:tag): {p}")
            path_str, tag = p.split(":", 1)
            pairs.append((Path(path_str), tag))
    else:
        # use standard filenames if they exist
        default_pairs = [
            (Path("data/gen_raw_prollama.fasta"), "ProLLaMA"),
            (Path("data/gen_raw_protgpt2.fasta"), "ProtGPT2"),
        ]
        pairs = [(p, t) for (p, t) in default_pairs if p.exists()]
        if not pairs:
            raise SystemExit("No inputs found. Provide --pairs path:tag or ensure default FASTAs exist.")


    dfs = []
    for path, tag in pairs:
        if not path.exists():
            print(f"[warn] missing file: {path} (skipping)")
            continue
        df = process_fasta(path, tag)
        dfs.append(df)
    
        safe_tag = tag.lower().replace(" ", "")
        per_path = outdir / f"metrics_{safe_tag}_raw.csv"
        df.to_csv(per_path, index=False)
        print(f"  -> wrote {per_path} (n={len(df)})")

    if not dfs:
        raise SystemExit("No data processed.")

    # Combined CSV
    all_df = pd.concat(dfs, ignore_index=True)
    combined_path = outdir / "metrics_raw.csv"
    all_df.to_csv(combined_path, index=False)
    print(f"\n>> wrote combined: {combined_path} (n={len(all_df)})")

    # Quick numeric summary
    by_model = all_df.groupby("model").agg(
        n=("length", "size"),
        mean_len=("length", "mean"),
        mean_instab=("instability", "mean"),
        mean_gravy=("gravy", "mean"),
        mean_lowcx=("low_complex", "mean"),
        max_run=("max_homopolymer", "max"),
    )
    pd.set_option("display.precision", 3)
    print("\n=== Summary by model ===")
    print(by_model)

if __name__ == "__main__":
    main()
