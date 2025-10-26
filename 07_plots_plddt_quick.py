import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

MERGED = Path("results/merged_metrics_plddt.csv")
PLOT_DIR = Path("results/plots"); PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MERGED)

# Box: pLDDT by keep (only if present and non-null)
if "keep" in df.columns and df["keep"].notna().any():
    plt.figure()
    df.boxplot(column="pLDDT", by="keep")
    plt.title("pLDDT by Mini-EPGF keep"); plt.suptitle(""); plt.ylabel("pLDDT (0–100)")
    plt.tight_layout(); plt.savefig(PLOT_DIR/"plddt_by_keep.png"); plt.close()

# Bar: mean pLDDT by model × (optional) keep
if "keep" in df.columns and df["keep"].notna().any():
    g = df.groupby(["model","keep"])["pLDDT"].mean().unstack()
    ax = g.plot(kind="bar"); ax.set_ylabel("mean pLDDT (0–100)")
    ax.set_title("mean pLDDT by model × keep")
    plt.tight_layout(); plt.savefig(PLOT_DIR/"plddt_model_keep_bar.png"); plt.close()
else:
    g = df.groupby(["model"])["pLDDT"].mean()
    ax = g.plot(kind="bar"); ax.set_ylabel("mean pLDDT (0–100)")
    ax.set_title("mean pLDDT by model")
    plt.tight_layout(); plt.savefig(PLOT_DIR/"plddt_model_bar.png"); plt.close()

# Scatter: instability vs pLDDT, color by model (if columns exist)
if "instability" in df.columns:
    markers = {"ProLLaMA":"o", "ProtGPT2":"^"}
    plt.figure()
    for m, sub in df.groupby("model"):
        plt.scatter(sub["instability"], sub["pLDDT"], label=m, marker=markers.get(m, "o"), alpha=0.85)
    plt.xlabel("Instability index"); plt.ylabel("pLDDT (0–100)"); plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_DIR/"plddt_vs_instability_scatter.png"); plt.close()

print("Saved plots under:", PLOT_DIR)
