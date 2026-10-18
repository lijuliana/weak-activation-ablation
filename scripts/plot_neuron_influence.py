"""Per-neuron downstream influence vs activation-magnitude bin.
    python scripts/plot_neuron_influence.py results/qwen3-1.7b-base figures/quick
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(res / "neuron_influence.csv")
order = ["weakest 50%", "10-50%", "1-10%", "0.1-1%", "top 0.1%"]
df["bin_name"] = pd.Categorical(df["bin_name"], order, ordered=True)
df["kl_per_h2"] = df.sum_kl / df.sum_h2            # ratio of sums (per-token ratio is noisy for tiny h)
tot = df.groupby(["layer", "neuron"]).sum_kl.transform("sum")
df["share"] = df.sum_kl / tot                     # share of the neuron's total downstream effect

def summarize(col):
    g = df.groupby("bin_name", observed=True)[col]
    med = g.median(); lo = g.quantile(0.25); hi = g.quantile(0.75)
    return med.reindex(order), lo.reindex(order), hi.reindex(order)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
x = np.arange(len(order))
for ax, col, title, ylab in [
    (axes[0], "mean_kl", "Downstream effect of zeroing the neuron", "mean KL(clean || ablated) at those tokens"),
    (axes[1], "share", "Share of the neuron's total effect", "fraction of summed KL"),
    (axes[2], "kl_per_h2", "Effect per unit activation^2", "KL / h^2  (flat = read linearly)")]:
    med, lo, hi = summarize(col)
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o-", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20)
    ax.set_title(title); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    if col != "share":
        ax.set_yscale("log")
axes[0].set_xlabel("tokens binned by the neuron's own |activation| (percentile)")
fig.suptitle(f"{df.groupby(['layer','neuron']).ngroups} neurons across layers {sorted(int(l) for l in df.layer.unique())}; median and IQR across neurons")
fig.tight_layout(); fig.savefig(out / "quick5_neuron_influence.png", dpi=150)

# per-layer version of the per-unit curve
fig, ax = plt.subplots(figsize=(7, 4))
for L, g in df.groupby("layer"):
    m = g.groupby("bin_name", observed=True).kl_per_h2.median().reindex(order)
    ax.plot(x, m, "o-", label=f"layer {L}")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=20); ax.set_yscale("log"); ax.grid(alpha=0.3)
ax.set_ylabel("KL / h^2 (median over neurons)"); ax.set_title("Effect per unit activation^2, by layer"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / "quick6_neuron_influence_by_layer.png", dpi=150)

tab = df.groupby("bin_name", observed=True).agg(n_tokens=("n", "median"), h_over_max=("mean_h_over_max", "median"),
                                                 kl=("mean_kl", "median"), share=("share", "median"),
                                                 kl_per_h2=("kl_per_h2", "median"), leakage=("mean_kl_out", "median")).reindex(order)
print(tab.to_string(float_format=lambda v: f"{v:.3g}"))
