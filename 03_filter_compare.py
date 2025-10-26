
# 03_filter_compare.py — Mini-EPGF filtering + plots + summary (symmetric rules)
# Usage: python 03_filter_compare.py
# Produces:
#   results/filtered_summary.csv
#   data/prollama_kept.fasta, data/prollama_rejected.fasta
#   data/protgpt2_kept.fasta, data/protgpt2_rejected.fasta
#   results/plots/*

import re
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140

IN = Path("results/metrics_raw.csv")
FA_PRO = Path("data/gen_raw_prollama.fasta")      
FA_PG  = Path("data/gen_raw_protgpt2.fasta")
OUTP = Path("results/plots"); OUTP.mkdir(parents=True, exist_ok=True)

# ---- symmetric Mini-EPGF thresholds (apply to BOTH models) ----
INSTAB_MAX   = 40.0
LOWCX_MAX    = 0.35
HOMORUN_MAX  = 5        
GRAVY_MIN    = -2.0
GRAVY_MAX    = 0.0      
MIN_LENGTH   = 80
MIN_ENTROPY  = 3.0

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

def clean_to_aa20(seq: str) -> str:
    s = re.sub(r"[^A-Z]", "", str(seq).upper())
    return "".join(ch for ch in s if ch in AA20)

def load_fasta_dict(path: Path):
    d = {}
    if path.exists():
        for r in SeqIO.parse(str(path), "fasta"):
            d[r.id] = clean_to_aa20(str(r.seq))
    return d

def write_fasta(ids, src_dict, out_path: Path):
    ids = sorted(ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rid in ids:
            s = src_dict.get(rid)
            if s:
                f.write(f">{rid}\n{s}\n")

def main():
    if not IN.exists():
        raise SystemExit(f"Missing {IN}. Run 02_metrics.py first.")
    df = pd.read_csv(IN)

    # Symmetric keep rules (identical for both models)
    keep = (
        (df["instability"] < INSTAB_MAX) &
        (df["low_complex"] < LOWCX_MAX) &
        (df["max_homopolymer"] <= HOMORUN_MAX) &
        (df["gravy"].between(GRAVY_MIN, GRAVY_MAX)) &
        (df["length"] >= MIN_LENGTH) &
        (df["aa_entropy"] >= MIN_ENTROPY)
    )
    df["keep"] = keep

    # Per-model acceptance summary
    summary = df.groupby("model").agg(
        n=("id","size"),
        kept=("keep","sum"),
        keep_rate=("keep","mean"),
        mean_len=("length","mean"),
        mean_instab=("instability","mean"),
        mean_gravy=("gravy","mean"),
        mean_lowcx=("low_complex","mean"),
        max_run=("max_homopolymer","max"),
    ).reset_index()
    summary["keep_rate"] = summary["keep_rate"].round(3)
    out_sum = Path("results_2/filtered_summary.csv")
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)
    print("\n=== Mini-EPGF summary (symmetric rules) ===")
    print(summary)

    # Split IDs by model/keep
    prol_ids = df[(df.model=="ProLLaMA") & df.keep]["id"].tolist()
    prol_rej = df[(df.model=="ProLLaMA") & (~df.keep)]["id"].tolist()
    prot_ids = df[(df.model=="ProtGPT2") & df.keep]["id"].tolist()
    prot_rej = df[(df.model=="ProtGPT2") & (~df.keep)]["id"].tolist()

    # Write FASTAs (AA20-cleaned sequences from original model FASTAs)
    fa_pro = load_fasta_dict(FA_PRO)
    fa_pg  = load_fasta_dict(FA_PG)
    write_fasta(prol_ids, fa_pro, Path("data_2/prollama_kept.fasta"))
    write_fasta(prol_rej, fa_pro, Path("data_2/prollama_rejected.fasta"))
    write_fasta(prot_ids,  fa_pg,  Path("data_2/protgpt2_kept.fasta"))
    write_fasta(prot_rej,  fa_pg,  Path("data_2/protgpt2_rejected.fasta"))
    print("\nWrote kept/rejected FASTAs to data/")

    
    for col in ["instability","gravy","aa_entropy","low_complex"]:
        if col not in df.columns: continue
        plt.figure(figsize=(5.2,3.2))
        for m, sub in df.groupby("model"):
            sub[col].dropna().plot(kind="hist", bins=40, alpha=0.35, density=True, label=f"{m} raw")
            sub[sub.keep][col].dropna().plot(kind="hist", bins=40, alpha=0.65, density=True, label=f"{m} kept")
        plt.xlabel(col); plt.ylabel("density"); plt.legend()
        plt.tight_layout(); plt.savefig(OUTP/f"{col}_by_model_kept_hist.png"); plt.close()

    # GRAVY vs Instability scatter
    if {"gravy","instability"}.issubset(df.columns):
        plt.figure(figsize=(5.4,3.6))
        for m, sub in df.groupby("model"):
            plt.scatter(sub["gravy"], sub["instability"], s=10, alpha=0.35, label=f"{m} raw")
            subk = sub[sub.keep]
            plt.scatter(subk["gravy"], subk["instability"], s=12, alpha=0.7, label=f"{m} kept")
        plt.axhline(INSTAB_MAX, ls="--")
        plt.xlabel("GRAVY"); plt.ylabel("Instability")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout(); plt.savefig(OUTP/"gravy_vs_instability_by_model_kept.png"); plt.close()

    print(f"Saved plots in {OUTP}")

if __name__ == "__main__":
    main()
