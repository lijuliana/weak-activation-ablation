"""Plot the scale-weak-tail test: loss and PopQA vs multiplier applied to sub-threshold activations.
    python scripts/plot_scale_weak.py results/qwen3-1.7b-base figures/quick
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

res, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (res / "summary.jsonl").read_text().splitlines() if l.strip()]
summ = pd.DataFrame(rows)
base = summ[summ["mode"] == "off"].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for frac in sorted({r["frac"] for r in rows if r["mode"].startswith("scaleweak")}):
    pts = {0.0: summ[(summ["mode"] == "threshold") & (summ["frac"] == frac)].iloc[0], 1.0: base}
    for r in rows:
        if r["mode"].startswith("scaleweak") and r["frac"] == frac:
            pts[float(r["mode"].split("_x")[1])] = r
    xs = sorted(pts)
    sp = pts[0.0]["sparsity"]
    lab = f"cutoff {100*frac:g}% of max ({100*sp:.0f}% of activations affected)"
    axes[0].plot(xs, [pts[x]["fineweb_nll"] - base["fineweb_nll"] for x in xs], "o-", label=lab)
    axes[1].plot(xs, [100 * pts[x]["popqa_acc"] for x in xs], "o-", label=lab)
for ax in axes:
    ax.set_xlabel("multiplier applied to weak activations (0 = zeroed, 1 = untouched)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("increase in held-out NLL (nats/token)")
axes[0].set_title("Does amplifying the weak tail hurt?")
axes[1].set_ylabel("PopQA accuracy (%)")
axes[1].set_title("Same, factual recall")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / "quick4_scale_weak_tail.png", dpi=150)
print(summ[summ["mode"].str.startswith("scaleweak") | (summ["mode"] == "threshold") & summ["frac"].isin([0.005, 0.015])][["cond", "fineweb_nll", "wt103_nll", "popqa_acc"]].to_string(index=False))
