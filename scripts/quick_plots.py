"""Two plain matplotlib plots of the headline result (default styling on purpose).
    python scripts/quick_plots.py results/qwen3-1.7b-base figures/quick
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

res, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
summ = pd.DataFrame([json.loads(l) for l in (res / "summary.jsonl").read_text().splitlines() if l.strip()])
base = summ[summ["mode"] == "off"].iloc[0]
m = pd.read_csv(res / "metrics.csv")

# --- plot 1: trivia accuracy vs % of activations zeroed, weakest-first vs random ----------
fig, ax = plt.subplots(figsize=(6.5, 4))
for mode, lab, mk in [("threshold", "zero the weakest activations", "o-"), ("random", "zero the same number at random", "s--"),
                      ("randomfrac", "zero a fixed random fraction", "^:")]:
    d = summ[summ["mode"] == mode].groupby("frac").mean(numeric_only=True).sort_values("sparsity")
    if len(d):
        ax.plot([0] + list(100 * d["sparsity"]), [100 * base["popqa_acc"]] + list(100 * d["popqa_acc"]), mk, label=lab)
ax.set_xlabel("% of all MLP activations zeroed")
ax.set_ylabel("PopQA accuracy (%)")
ax.set_title("Qwen3-1.7B: delete weak activations vs delete random ones")
ax.set_xlim(-2, 100)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out / "quick1_accuracy_vs_zeroed.png", dpi=150); plt.close(fig)

# --- plot 2: which facts survive, obscure vs famous, under weakest-first ablation ------------
t = m[m["mode"] == "threshold"].sort_values("sparsity")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(100 * t["sparsity"], 100 * (1 - t["forget_bin1"]), "o-", label="obscure subjects (rarest fifth)")
ax.plot(100 * t["sparsity"], 100 * (1 - t["forget_bin5"]), "s-", label="famous subjects (most popular fifth)")
ax.set_xlabel("% of all MLP activations zeroed (weakest first)")
ax.set_ylabel("% of previously-correct answers still correct")
ax.set_title("Which facts survive? Famous ones go first, not obscure ones")
ax.set_xlim(20, 100); ax.set_ylim(0, 105)
ax.legend(loc="lower left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out / "quick2_who_survives.png", dpi=150); plt.close(fig)

# --- plot 3: same comparison for random damage, at matched loss ----------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4))
for mode, lab, mk in [("threshold", "weakest-first", "o-"), ("randomfrac", "random fraction", "^:")]:
    d = m[m["mode"] == mode].sort_values("d_nll_fineweb")
    if len(d):
        ax.plot(d["d_nll_fineweb"], 100 * (d["retention_bin1"] - d["retention_bin5"]), mk, label=lab)
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(left=-0.002)
ax.set_xlabel("overall damage: increase in held-out loss (nats/token)")
ax.set_ylabel("retention gap, obscure minus famous (pct points)")
ax.set_title("Above 0 = obscure facts hold up better than famous ones")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out / "quick3_gap_at_matched_damage.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in out.glob("*.png")))
