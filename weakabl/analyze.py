"""Aggregate sweep outputs into metrics (with bootstrap CIs) and figures.

    python -m weakabl.analyze --results results/qwen3-1.7b-base --figures figures/qwen3-1.7b-base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

# ---- palette (validated with the dataviz skill's validator) ---------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COND = {"threshold": "#2a78d6", "random": "#eb6834"}            # categorical slots 1, 2
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blue, rare -> popular
COND_LABEL = {"threshold": "weak-activation ablation", "random": "count-matched random ablation"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "legend.frameon": False,
    "lines.linewidth": 1.8, "lines.markersize": 5,
})


# ---- loading ----------------------------------------------------------------------------
def load_results(res: Path):
    rows = [json.loads(l) for l in (res / "summary.jsonl").read_text().splitlines() if l.strip()]
    summ = pd.DataFrame(rows)
    q = pd.read_csv(res / "popqa_questions.csv")
    meta = json.loads((res / "popqa_meta.json").read_text())
    correct = {r["cond"]: np.load(res / f"popqa_correct_{r['cond']}.npy") for r in rows}
    nll = {r["cond"]: np.load(res / f"nll_fineweb_{r['cond']}.npy").astype(np.float32) for r in rows}
    nll_wt = {r["cond"]: np.load(res / f"nll_wt103_{r['cond']}.npy").astype(np.float32) for r in rows}
    return summ, q, meta, correct, nll, nll_wt


def within_relation_bins(q: pd.DataFrame, n_bins: int) -> np.ndarray:
    """Popularity quantile bins computed *within each relation type* (PopQA `prop`), so every bin has
    the same mix of relations. Guards against guessable relations (e.g. occupation) being concentrated
    in the global rare bin."""
    logp = np.log10(np.maximum(q["s_pop"].to_numpy(float), 1.0))
    out = np.zeros(len(q), dtype=int)
    for prop, idx in q.groupby("prop").indices.items():
        lp = logp[idx]
        ranks = np.argsort(np.argsort(lp, kind="stable"), kind="stable")
        out[idx] = np.minimum((ranks * n_bins) // len(idx), n_bins - 1)
    return out


MODES = ("threshold", "random", "randomfrac")
COND["randomfrac"] = "#1baf7a"  # categorical slot 3 (aqua)
COND_LABEL["randomfrac"] = "fixed-fraction random ablation"


def conditions(summ: pd.DataFrame):
    """-> {(mode, frac): [cond names (one per seed)]}, fracs sorted."""
    groups: dict[tuple[str, float], list[str]] = {}
    for _, r in summ.iterrows():
        if r["mode"] == "off":
            continue
        groups.setdefault((r["mode"], float(r["frac"])), []).append(r["cond"])
    fracs = sorted({f for (_, f) in groups})
    return groups, fracs


# ---- metrics with bootstrap -----------------------------------------------------------------
def popqa_metrics(base: np.ndarray, conds: list[np.ndarray], bins: np.ndarray, n_bins: int,
                  n_boot: int = 1000, seed: int = 0):
    """Per-bin accuracy retention and forget rate (P(wrong | base-correct)), averaged over seeds,
    with bootstrap CIs over questions (paired resampling). Returns dict of arrays (n_bins,) and (2, n_bins)."""
    C = np.stack(conds)  # (S, N)
    rng = np.random.default_rng(seed)

    def stat(idx):
        out_r, out_f, out_g = [], [], []
        for b in range(n_bins):
            sel = idx[bins[idx] == b]
            b0 = base[sel]
            acc0 = b0.mean()
            accs = C[:, sel].mean(1)
            out_r.append(np.mean(accs / acc0) if acc0 > 0 else np.nan)
            known = b0
            out_f.append(np.mean([(~C[s, sel][known]).mean() for s in range(C.shape[0])]) if known.sum() else np.nan)
            out_g.append(np.mean([(C[s, sel][~known]).mean() for s in range(C.shape[0])]) if (~known).sum() else np.nan)
        return np.array(out_r), np.array(out_f), np.array(out_g)

    N = len(base)
    r, f, g = stat(np.arange(N))
    boots = [stat(rng.integers(0, N, N)) for _ in range(n_boot)]
    R = np.stack([b[0] for b in boots]); Fg = np.stack([b[1] for b in boots]); G = np.stack([b[2] for b in boots])
    gap_r = R[:, 0] - R[:, -1]
    gap_f = Fg[:, 0] - Fg[:, -1]
    return {
        "retention": r, "retention_ci": np.nanpercentile(R, [2.5, 97.5], axis=0),
        "forget": f, "forget_ci": np.nanpercentile(Fg, [2.5, 97.5], axis=0),
        "gain": g, "gain_ci": np.nanpercentile(G, [2.5, 97.5], axis=0),
        "gap_retention": r[0] - r[-1], "gap_retention_ci": np.nanpercentile(gap_r, [2.5, 97.5]),
        "gap_forget": f[0] - f[-1], "gap_forget_ci": np.nanpercentile(gap_f, [2.5, 97.5]),
        "acc": np.array([np.mean(C[:, bins == b].mean(1)) for b in range(n_bins)]),
        "acc_all": float(C.mean()),
    }


def nll_delta(base: np.ndarray, conds: list[np.ndarray], n_boot: int = 1000, seed: int = 0):
    """Mean change in per-token NLL vs baseline, CI by bootstrap over sequences."""
    d = np.stack([c - base for c in conds]).mean(0)  # (n_seq, T)
    per_seq = d.mean(1)
    rng = np.random.default_rng(seed)
    boots = [per_seq[rng.integers(0, len(per_seq), len(per_seq))].mean() for _ in range(n_boot)]
    return float(per_seq.mean()), np.percentile(boots, [2.5, 97.5])


def token_freq_bins(res: Path, n_bins: int = 5):
    """Target-token frequency quintile for every held-out FineWeb position, using unigram counts
    from the calibration chunks when available (else the held-out chunks themselves)."""
    args = json.loads((res / "args.json").read_text())
    cache = Path(args["cache"])
    tag = f"fwedu_s{args['seq_len']}_c{args['n_calib']}_h{args['n_heldout']}_skip0"
    held = np.load(cache / f"{tag}_heldout.npy")
    calib_f = cache / f"{tag}_calib.npy"
    src = np.load(calib_f) if calib_f.exists() else held
    counts = np.bincount(src.ravel(), minlength=int(max(src.max(), held.max())) + 1)
    targets = held[:, 1:]
    freq = counts[targets]
    edges = np.quantile(np.log10(freq + 1), np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, np.log10(freq + 1), side="right") - 1, 0, n_bins - 1)
    labels = [f"{10**edges[i]:.0f}-{10**edges[i+1]:.0f}" for i in range(n_bins)]
    return b, labels, calib_f.exists()


# ---- figures -------------------------------------------------------------------------------
def _xaxis(ax, label="threshold X (% of each neuron's calibrated max |activation|)"):
    ax.set_xlabel(label)
    ax.margins(x=0.03)


def _series(summ, fracs, groups, nll, mode):
    xs, ys, cis, sp = [], [], [], []
    for f in fracs:
        cs = groups.get((mode, f))
        if not cs:
            continue
        m, ci = nll_delta(nll["baseline"], [nll[c] for c in cs])
        xs.append(100 * f); ys.append(m); cis.append(ci)
        sp.append(100 * summ[summ["cond"].isin(cs)]["sparsity"].mean())
    return np.array(xs), np.array(ys), np.array(cis), np.array(sp)


def fig_lm_loss(summ, fracs, groups, nll, nll_wt, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    base = summ[summ["mode"] == "off"].iloc[0]
    for mode in MODES:
        if not any(m == mode for (m, _) in groups):
            continue
        xs, y, ci, sp = _series(summ, fracs, groups, nll, mode)
        _, yw, ciw, _ = _series(summ, fracs, groups, nll_wt, mode)
        kw = dict(color=COND[mode], label=COND_LABEL[mode], markeredgecolor=SURFACE, markeredgewidth=1)
        if mode != "randomfrac":
            axes[0].plot(xs, y, "-o", **kw); axes[0].fill_between(xs, ci[:, 0], ci[:, 1], color=COND[mode], alpha=0.12, linewidth=0)
            axes[2].plot(xs, yw, "-o", **kw); axes[2].fill_between(xs, ciw[:, 0], ciw[:, 1], color=COND[mode], alpha=0.12, linewidth=0)
        o = np.argsort(sp)
        axes[1].plot(sp[o], y[o], "-o", **kw)
    axes[0].set_title(f"FineWeb-edu held-out loss (baseline {base['fineweb_nll']:.2f})")
    axes[0].set_ylabel("increase in NLL per token (nats)"); _xaxis(axes[0])
    axes[1].set_title("Loss against the fraction of activations zeroed")
    axes[1].set_ylabel("increase in NLL per token (nats)"); axes[1].set_xlabel("% of MLP activations zeroed (all layers)")
    axes[2].set_title(f"WikiText-103 held-out loss (baseline {base['wt103_nll']:.2f})")
    axes[2].set_ylabel("increase in NLL per token (nats)"); _xaxis(axes[2])
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.set_yscale("symlog", linthresh=0.01)
    fig.savefig(out / "fig1_lm_loss.png", dpi=200)
    fig.savefig(out / "fig1_lm_loss.svg")
    plt.close(fig)


def fig_popqa_bins(metrics, fracs, bin_labels, out: Path, base_acc_by_bin):
    n_bins = len(bin_labels)
    modes = [m for m in MODES if any(k[0] == m for k in metrics)]
    fig, axes = plt.subplots(2, len(modes), figsize=(5 * len(modes), 7), constrained_layout=True, squeeze=False)
    for j, mode in enumerate(modes):
        keys = [f for f in fracs if (mode, f) in metrics]
        ms = [metrics[(mode, f)] for f in keys]
        if mode == "randomfrac":
            xs = np.array([100 * f for f in keys]); xlabel = "% of activations zeroed at random"
        else:
            xs = np.array([100 * f for f in keys]); xlabel = None
        for b in range(n_bins):
            lab = f"bin {b+1}: {bin_labels[b]} views/mo (base acc {100*base_acc_by_bin[b]:.0f}%)"
            for row, key in enumerate(("retention", "forget")):
                y = np.array([m[key][b] for m in ms]); ci = np.array([m[key + "_ci"][:, b] for m in ms])
                ax = axes[row, j]
                ax.plot(xs, y, "-o", color=RAMP[b], label=lab if row == 0 else None, markeredgecolor=SURFACE, markeredgewidth=1)
                ax.fill_between(xs, ci[:, 0], ci[:, 1], color=RAMP[b], alpha=0.12, linewidth=0)
        axes[0, j].set_title(f"{COND_LABEL[mode]}: accuracy retention")
        axes[1, j].set_title(f"{COND_LABEL[mode]}: forget rate")
        axes[0, j].axhline(1.0, color=INK2, linewidth=0.8)
        axes[0, j].set_ylabel("PopQA accuracy / baseline accuracy")
        axes[1, j].set_ylabel("P(wrong | correct at baseline)")
        if xlabel:
            axes[1, j].set_xlabel(xlabel); axes[0, j].set_xlabel(xlabel)
        else:
            _xaxis(axes[1, j]); _xaxis(axes[0, j])
        for row in range(2):
            axes[row, j].set_xscale("symlog", linthresh=0.1)
    axes[0, 0].legend(loc="lower left", title="subject popularity (rare -> popular)", fontsize=8, title_fontsize=8)
    fig.savefig(out / "fig2_popqa_by_popularity.png", dpi=200)
    fig.savefig(out / "fig2_popqa_by_popularity.svg")
    plt.close(fig)


def fig_gap(metrics, fracs, dnll, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for mode in MODES:
        keys = [f for f in fracs if (mode, f) in metrics]
        if not keys:
            continue
        xs = np.array([100 * f for f in keys]); dn = np.array([dnll[(mode, f)] for f in keys])
        kw = dict(color=COND[mode], label=COND_LABEL[mode], markeredgecolor=SURFACE, markeredgewidth=1)
        yr = np.array([metrics[(mode, f)]["gap_retention"] for f in keys]); cr = np.array([metrics[(mode, f)]["gap_retention_ci"] for f in keys])
        yf = np.array([metrics[(mode, f)]["gap_forget"] for f in keys]); cf = np.array([metrics[(mode, f)]["gap_forget_ci"] for f in keys])
        if mode != "randomfrac":
            axes[0].plot(xs, yr, "-o", **kw); axes[0].fill_between(xs, cr[:, 0], cr[:, 1], color=COND[mode], alpha=0.12, linewidth=0)
        o = np.argsort(dn)
        axes[1].plot(dn[o], yr[o], "-o", **kw); axes[1].fill_between(dn[o], cr[o, 0], cr[o, 1], color=COND[mode], alpha=0.12, linewidth=0)
        axes[2].plot(dn[o], yf[o], "-o", **kw); axes[2].fill_between(dn[o], cf[o, 0], cf[o, 1], color=COND[mode], alpha=0.12, linewidth=0)
    for ax in axes:
        ax.axhline(0, color=INK2, linewidth=0.8)
    axes[0].set_title("Retention gap: rarest bin minus most popular bin")
    axes[0].set_ylabel("retention(rare) - retention(popular)"); _xaxis(axes[0]); axes[0].set_xscale("symlog", linthresh=0.1)
    axes[1].set_title("Retention gap at matched overall damage")
    axes[1].set_ylabel("retention(rare) - retention(popular)")
    axes[1].set_xlabel("increase in held-out NLL per token (nats, FineWeb-edu)"); axes[1].set_xscale("symlog", linthresh=0.01)
    axes[2].set_title("Forget-rate gap at matched overall damage")
    axes[2].set_ylabel("forget(rare) - forget(popular)")
    axes[2].set_xlabel("increase in held-out NLL per token (nats, FineWeb-edu)"); axes[2].set_xscale("symlog", linthresh=0.01)
    axes[0].legend(loc="upper left", fontsize=8)
    for ax in axes[1:]:
        ax.set_xlim(left=-0.002)
    fig.text(0.01, -0.03, "Retention gap below zero (forget gap above zero): rare facts lose relatively more. 95% bootstrap CIs over questions.", color=INK2, fontsize=8)
    fig.savefig(out / "fig3_rare_vs_popular_gap.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig3_rare_vs_popular_gap.svg", bbox_inches="tight")
    plt.close(fig)


def fig_token_freq(fracs, groups, nll, res: Path, out: Path):
    try:
        fb, labels, from_calib = token_freq_bins(res)
    except FileNotFoundError:
        return None
    modes = [m for m in ("threshold", "random") if any(k[0] == m for k in groups)]
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 3.6), constrained_layout=True, sharey=True, squeeze=False)
    axes = axes[0]
    for j, mode in enumerate(modes):
        for b in range(len(labels)):
            xs, ys = [], []
            for f in fracs:
                cs = groups.get((mode, f))
                if not cs:
                    continue
                d = np.stack([nll[c] - nll["baseline"] for c in cs]).mean(0)
                xs.append(100 * f); ys.append(d[fb == b].mean())
            axes[j].plot(xs, ys, "-o", color=RAMP[b], label=f"{labels[b]}", markeredgecolor=SURFACE, markeredgewidth=1)
        axes[j].set_title(COND_LABEL[mode])
        _xaxis(axes[j]); axes[j].set_xscale("symlog", linthresh=0.1); axes[j].set_yscale("symlog", linthresh=0.01)
    axes[0].set_ylabel("increase in NLL per token (nats)")
    src = "calibration set" if from_calib else "held-out set"
    axes[0].legend(title=f"token count in {src} (rare -> common)", fontsize=8, title_fontsize=8)
    fig.suptitle("Held-out loss increase by target-token frequency", fontweight="semibold")
    fig.savefig(out / "fig4_token_frequency.png", dpi=200)
    fig.savefig(out / "fig4_token_frequency.svg")
    plt.close(fig)
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--figures", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--binning", choices=["within-relation", "global"], default="within-relation",
                   help="popularity bins per relation type (default) or global quantiles as recorded by run.py")
    a = p.parse_args()
    res, out = Path(a.results), Path(a.figures)
    out.mkdir(parents=True, exist_ok=True)

    summ, q, meta, correct, nll, nll_wt = load_results(res)
    groups, fracs = conditions(summ)
    n_bins = len(meta["bin_labels"])
    if a.binning == "within-relation":
        bins = within_relation_bins(q, n_bins)
        logp = np.log10(np.maximum(q["s_pop"].to_numpy(float), 1.0))
        bin_labels = [f"median 10^{np.median(logp[bins == b]):.1f}" for b in range(n_bins)]
    else:
        bins = q["bin"].to_numpy()
        bin_labels = meta["bin_labels"]
    base = correct["baseline"]
    base_acc_by_bin = [base[bins == b].mean() for b in range(n_bins)]

    metrics, dnll, rows = {}, {}, []
    for (mode, f), cs in groups.items():
        m = popqa_metrics(base, [correct[c] for c in cs], bins, n_bins, n_boot=a.n_boot)
        metrics[(mode, f)] = m
        d, dci = nll_delta(nll["baseline"], [nll[c] for c in cs], n_boot=a.n_boot)
        dw, dwci = nll_delta(nll_wt["baseline"], [nll_wt[c] for c in cs], n_boot=a.n_boot)
        dnll[(mode, f)] = d
        rows.append({"mode": mode, "frac": f, "n_seeds": len(cs),
                     "sparsity": summ[summ["cond"].isin(cs)]["sparsity"].mean(),
                     "d_nll_fineweb": d, "d_nll_fineweb_lo": dci[0], "d_nll_fineweb_hi": dci[1],
                     "d_nll_wt103": dw, "d_nll_wt103_lo": dwci[0], "d_nll_wt103_hi": dwci[1],
                     "popqa_acc": m["acc_all"],
                     **{f"retention_bin{b+1}": m["retention"][b] for b in range(n_bins)},
                     **{f"forget_bin{b+1}": m["forget"][b] for b in range(n_bins)},
                     **{f"gain_bin{b+1}": m["gain"][b] for b in range(n_bins)},
                     "gap_retention": m["gap_retention"], "gap_retention_lo": m["gap_retention_ci"][0],
                     "gap_retention_hi": m["gap_retention_ci"][1],
                     "gap_forget": m["gap_forget"], "gap_forget_lo": m["gap_forget_ci"][0],
                     "gap_forget_hi": m["gap_forget_ci"][1]})
    table = pd.DataFrame(rows).sort_values(["mode", "frac"])
    table.to_csv(res / ("metrics.csv" if a.binning == "within-relation" else "metrics_globalbins.csv"), index=False)

    fig_lm_loss(summ, fracs, groups, nll, nll_wt, out)
    fig_popqa_bins(metrics, fracs, bin_labels, out, base_acc_by_bin)
    fig_gap(metrics, fracs, dnll, out)
    fig_token_freq(fracs, groups, nll, res, out)

    calib = torch.load(res / "calibration.pt")
    amax = calib["amax"]
    print(f"binning={a.binning}; baseline: PopQA acc {base.mean():.4f}; by bin (rare->popular) "
          f"{[round(float(x),3) for x in base_acc_by_bin]}; bins {bin_labels}; n={len(base)}")
    print(f"calibration: {calib['n_tokens']} tokens; per-neuron max |a| median {amax.median():.3f}, "
          f"p99 {amax.flatten().quantile(0.99):.3f}, max {amax.max():.3f}")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(table[["mode", "frac", "sparsity", "d_nll_fineweb", "d_nll_wt103", "popqa_acc",
                     "retention_bin1", f"retention_bin{n_bins}", "gap_retention", "gap_retention_lo",
                     "gap_retention_hi", "gap_forget", "gap_forget_lo", "gap_forget_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
