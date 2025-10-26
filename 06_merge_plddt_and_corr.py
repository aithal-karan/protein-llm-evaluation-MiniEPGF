
# 06_merge_plddt_and_corr.py

from pathlib import Path
import pandas as pd
import numpy as np

METRICS_PATH = Path("results/metrics_raw.csv")
PLDDT_PATH   = Path("results/pLDDT.csv")
MANIFEST_PATH= Path("results/subset_manifest.csv")

OUT_MERGED   = Path("results/merged_metrics_plddt.csv")
OUT_SUMMARY  = Path("results/plddt_by_model_keep.csv")

def add_base_id(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    
    df["base_id"] = df[id_col].astype(str).str.split(pat=r"\|", n=1, expand=False).str[0]
    return df

def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 02_metrics.py first.")
    df = pd.read_csv(path)
    if "id" not in df.columns:   raise RuntimeError("metrics_raw.csv missing 'id'")
    if "model" not in df.columns:raise RuntimeError("metrics_raw.csv missing 'model'")
    return add_base_id(df, "id")

def load_plddt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 05_run_*pLDDT* script first.")
    df = pd.read_csv(path)
    if "id" not in df.columns:   raise RuntimeError("pLDDT.csv missing 'id'")
    if "pLDDT" not in df.columns:raise RuntimeError("pLDDT.csv missing 'pLDDT'")
    if pd.api.types.is_numeric_dtype(df["pLDDT"]) and df["pLDDT"].max() <= 1.5:
        df["pLDDT"] = df["pLDDT"] * 100.0
    return add_base_id(df, "id")

def load_manifest(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] {path} not found — proceeding without manifest (model/keep may be missing).")
        return None
    df = pd.read_csv(path)
    if "id" not in df.columns:      raise RuntimeError("subset_manifest.csv missing 'id'")
    if "model" not in df.columns:   raise RuntimeError("subset_manifest.csv missing 'model'")
    if "keep" not in df.columns:    raise RuntimeError("subset_manifest.csv missing 'keep'")
    df = add_base_id(df, "id")
    if df["keep"].dtype != bool:
        df["keep"] = df["keep"].astype(bool)
    return df[["base_id","model","keep"]].drop_duplicates()

def main():
    pd.set_option('future.no_silent_downcasting', True)

    metrics  = load_metrics(METRICS_PATH)
    plddt    = load_plddt(PLDDT_PATH)
    manifest = load_manifest(MANIFEST_PATH)

    pl_lab = plddt.merge(manifest, on="base_id", how="left") if manifest is not None else plddt.copy()

    drop_cols = [c for c in ["model","keep"] if c in metrics.columns]
    merged = pl_lab.merge(metrics.drop(columns=drop_cols, errors="ignore"),
                          on="base_id", how="left", suffixes=("_plddt","_metrics"))

    merged["id"] = np.where(merged.get("id_metrics").notna(), merged["id_metrics"],
                            np.where(merged.get("id_plddt").notna(), merged["id_plddt"], merged["base_id"]))

    front = ["id","base_id","model","keep","pLDDT"]
    merged = merged[front + [c for c in merged.columns if c not in front]]

    OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_MERGED, index=False)

    cov_total_metrics = metrics["base_id"].nunique()
    cov_folded        = merged["base_id"].nunique()
    print(f"merged n = {len(merged)} (unique base_id with pLDDT: {cov_folded} / metrics unique: {cov_total_metrics})")

    missing = sorted(set(metrics["base_id"]) - set(merged["base_id"]))
    if missing:
        print(f"{len(missing)} metrics rows lack pLDDT (likely not folded). E.g.: {missing[:5]}")

    have_plddt = merged[merged["pLDDT"].notna()].copy()

    if "model" in have_plddt.columns and "keep" in have_plddt.columns:
        grp = (have_plddt.groupby(["model","keep"], dropna=False)["pLDDT"]
               .mean().round(2))
        print("\n=== mean pLDDT by model × keep (0–100) ===")
        print(grp)
        grp.reset_index().to_csv(OUT_SUMMARY, index=False)
        print(f"\nWrote: {OUT_MERGED}\n       {OUT_SUMMARY}")
    else:
        print("\n[warn] Missing 'model' and/or 'keep' on folded subset; writing merged CSV only.")
        print(f"Wrote: {OUT_MERGED}")

    corr_cols = [c for c in ["instability","low_complex","gravy","aa_entropy","max_homopolymer","length"] if c in have_plddt.columns]
    if corr_cols:
        for m, sub in have_plddt.groupby("model", dropna=False):
            if sub.empty: 
                continue
            print(f"\n== correlations with pLDDT ({m}) ==")
            out = sub[["pLDDT"] + corr_cols].corr(numeric_only=True)["pLDDT"].round(3).sort_values(ascending=False)
            print(out)

if __name__ == "__main__":
    main()
