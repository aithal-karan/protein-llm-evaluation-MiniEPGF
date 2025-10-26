
# 07_analyze_subset.py (v2 clean)
# - Reads results/merged_metrics_plddt_2.csv
# - Writes summaries into results_2/
# - Plots:
#     * box+jitter of pLDDT by model×keep
#     * ECDFs of pLDDT per model (overall)


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- I/O ----
IN_MERGED = Path("results/merged_metrics_plddt.csv")
OUT_DIR   = Path("results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"; PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_BUCKET       = OUT_DIR / "stage7_bucket_summary.csv"
OUT_BUCKET_CI    = OUT_DIR / "stage7_bucket_summary_with_ci.csv"
OUT_FAMILY       = OUT_DIR / "stage7_family_counts.csv"
OUT_TESTS        = OUT_DIR / "stage7_significance_tests.csv"
OUT_BUNDLE       = OUT_DIR / "stage7_summary_bundle.csv"

plt.rcParams["figure.dpi"] = 140

# ---- helpers ----
def parse_family_from_id(id_str: str) -> str:
    if not isinstance(id_str, str): return "(unknown)"
    return id_str.split("|", 1)[1] if "|" in id_str else "(unknown)"

def se(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    return float(np.nan) if n <= 1 else float(np.nanstd(x, ddof=1) / np.sqrt(n))

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run Stage 6 first.")
    df = pd.read_csv(path)
    need = {"model","keep","id","pLDDT"}
    if not need.issubset(df.columns):
        missing = ", ".join(sorted(need - set(df.columns)))
        raise SystemExit(f"{path} missing columns: {missing}")
    df = df[df["pLDDT"].notna()].copy()
    if df["keep"].dtype != bool:
        df["keep"] = df["keep"].astype(bool)
    if "family" not in df.columns:
        df["family"] = df["id"].apply(parse_family_from_id)
    return df

def bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["model","keep"], dropna=False)["pLDDT"]
    out = grp.agg(n="size", mean="mean", median="median",
                  std=lambda s: np.nanstd(s, ddof=1))
    out["se"] = grp.apply(lambda s: se(s.values))
    out = out.reset_index()
    for c in ["mean","median","std","se"]:
        out[c] = out[c].astype(float)
    return out

def family_counts(df: pd.DataFrame) -> pd.DataFrame:
    tab = (df.groupby(["family","model","keep"])["id"]
             .count().rename("n").reset_index()
             .sort_values(["family","model","keep"]))
    return tab

def try_welch(a, b):
    try:
        from scipy import stats
    except Exception:
        return np.nan, np.nan
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if a.size < 3 or b.size < 3: return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)

# ---- plotting ----
def plot_box_jitter(df: pd.DataFrame, out_path: Path):
    # fixed order for 2×2 buckets
    order = [("ProLLaMA", False), ("ProLLaMA", True),
             ("ProtGPT2", False), ("ProtGPT2", True)]
    data, labels = [], []
    for m,k in order:
        vals = df[(df["model"]==m)&(df["keep"]==k)]["pLDDT"].dropna().values
        if vals.size == 0: continue
        data.append(vals); labels.append(f"{m}\nkeep={k}")

    if not data:
        print("[warn] No data for box plot.")
        return

    plt.figure(figsize=(7.6, 4.4))
    bp = plt.boxplot(data, showfliers=False, widths=0.6)
    # jittered points
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(loc=i, scale=0.05, size=vals.size)
        plt.scatter(x, vals, s=14, alpha=0.6)

    plt.xticks(range(1, len(labels)+1), labels)
    plt.ylabel("pLDDT (0–100)")
    plt.title("pLDDT by model × keep")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def ecdf(arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0: return np.array([]), np.array([])
    x = np.sort(arr)
    y = np.arange(1, x.size+1) / x.size
    return x, y

def plot_ecdf_by_model(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6.8, 4.2))
    for m, sub in df.groupby("model"):
        x, y = ecdf(sub["pLDDT"].values)
        if x.size == 0: continue
        plt.step(x, y, where="post", label=m)
    plt.xlabel("pLDDT (0–100)")
    plt.ylabel("ECDF")
    plt.title("Overall pLDDT ECDF by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

# ---- main ----
def main():
    df = load_data(IN_MERGED)

    # 1) bucket table + CIs
    tab = bucket_summary(df)
    tab_ci = tab.copy()
    tab_ci["ci95_lower"] = tab_ci["mean"] - 1.96*tab_ci["se"]
    tab_ci["ci95_upper"] = tab_ci["mean"] + 1.96*tab_ci["se"]
    tab.round(2).to_csv(OUT_BUCKET, index=False)
    tab_ci.round(2).to_csv(OUT_BUCKET_CI, index=False)

    print("\n=== Stage 7 (v2): 2×2 bucket summary (pLDDT) ===")
    print(tab.round(2))

    # 2) per-family counts
    fam = family_counts(df)
    fam.to_csv(OUT_FAMILY, index=False)
    print("\nTop per-family counts (head):")
    print(fam.head(12))

    # 3) stats tests (Welch)
    tests = []
    # overall
    t, p = try_welch(df[df.model=="ProLLaMA"]["pLDDT"],
                     df[df.model=="ProtGPT2"]["pLDDT"])
    tests.append({"contrast":"ProLLaMA vs ProtGPT2 (overall)",
                  "t":t, "p":p,
                  "n_pro":int((df.model=="ProLLaMA").sum()),
                  "n_pg2":int((df.model=="ProtGPT2").sum())})
    # per keep
    for k in [False, True]:
        sub = df[df["keep"]==k]
        t, p = try_welch(sub[sub.model=="ProLLaMA"]["pLDDT"],
                         sub[sub.model=="ProtGPT2"]["pLDDT"])
        tests.append({"contrast":f"ProLLaMA vs ProtGPT2 (keep={k})",
                      "t":t, "p":p,
                      "n_pro":int((sub.model=="ProLLaMA").sum()),
                      "n_pg2":int((sub.model=="ProtGPT2").sum())})
    pd.DataFrame(tests).to_csv(OUT_TESTS, index=False)

    # 4) bundle CSV (easy import into LaTeX/Sheets)
    bundle = []
    bci = tab_ci.copy(); bci["section"] = "bucket_stats_with_ci"
    bci["family"] = ""
    bundle.append(bci[["section","model","keep","n","mean","median","std","se","ci95_lower","ci95_upper","family"]])

    fam2 = fam.copy(); fam2["section"] = "family_counts"
    fam2["mean"]=fam2["median"]=fam2["std"]=fam2["se"]=fam2["ci95_lower"]=fam2["ci95_upper"]=""
    bundle.append(fam2[["section","model","keep","n","mean","median","std","se","ci95_lower","ci95_upper","family"]])

    wt = pd.DataFrame(tests); wt["section"]="welch_tests"; wt["family"]=""
    for col in ["keep","mean","median","std","se","ci95_lower","ci95_upper"]:
        wt[col] = ""
    wt = wt.rename(columns={"contrast":"contrast","t":"t","p":"p","n_pro":"n_pro","n_pg2":"n_pg2"})
    # pad columns to match
    bundle.append(wt[["section","contrast","t","p","n_pro","n_pg2","family"]])

    
    
    pd.concat(bundle, axis=0, ignore_index=True).to_csv(OUT_BUNDLE, index=False)

    # 5) plots
    plot_box_jitter(df, PLOTS_DIR / "box_pLDDT_by_model_keep.png")
    plot_ecdf_by_model(df, PLOTS_DIR / "ecdf_pLDDT_by_model.png")

    print("\nSaved:")
    print(f" - {OUT_BUCKET}")
    print(f" - {OUT_BUCKET_CI}")
    print(f" - {OUT_FAMILY}")
    print(f" - {OUT_TESTS}")
    print(f" - {OUT_BUNDLE}")
    print(f" - {PLOTS_DIR}/box_pLDDT_by_model_keep.png")
    print(f" - {PLOTS_DIR}/ecdf_pLDDT_by_model.png")

if __name__ == "__main__":
    main()