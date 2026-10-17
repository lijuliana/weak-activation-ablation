"""Per-neuron downstream influence as a function of the neuron's own activation magnitude.

For each sampled neuron j (layer L): bin tokens by percentile of |h_j| over a held-out pool
(top 0.1%, 0.1-1%, 1-10%, 10-50%, weakest 50%). For each bin, zero h_j only at the tokens in that
bin and measure KL(p_clean || p_ablated) at those same tokens (same-position effect). Also record
KL at tokens outside the bin in the same pass (cross-token leakage control).

    python -m weakabl.neuron_influence --out results/qwen3-1.7b-base --n-seqs 160 --per-layer 15
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .data import fineweb_edu_chunks
from .run import pick_device

BINS = [(0.0, 0.001, "top 0.1%"), (0.001, 0.01, "0.1-1%"), (0.01, 0.1, "1-10%"),
        (0.1, 0.5, "10-50%"), (0.5, 1.0, "weakest 50%")]  # fraction of tokens from the top, by |h|


class SingleNeuronAblator:
    """Pre-hook on one layer's down_proj: zero column `j` at positions where `mask` is True."""

    def __init__(self, model):
        self.layers = [l.mlp.down_proj for l in model.model.layers]
        self.layer = None; self.j = None; self.mask = None; self.record = None
        self.handles = [m.register_forward_pre_hook(self._hook(i)) for i, m in enumerate(self.layers)]

    def _hook(self, i):
        def hook(module, args):
            h = args[0]
            if self.record is not None and i == self.record[0]:
                self.record[1].append(h[:, :, self.record[2]].detach().float().cpu())  # (B,T,len(js))
            if self.layer == i and self.mask is not None:
                h = h.clone()
                h[:, :, self.j] = torch.where(self.mask, torch.zeros_like(h[:, :, self.j]), h[:, :, self.j])
                return (h,)
            return None
        return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--out", default="results/qwen3-1.7b-base")
    p.add_argument("--cache", default="data_cache")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--n-seqs", type=int, default=160, help="held-out sequences used as the token pool")
    p.add_argument("--layers", default="0,4,8,12,16,20,24,27")
    p.add_argument("--per-layer", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="fp32 by default: weak-bin effects are ~1e-5 nats, below bf16 run-to-run noise")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), attn_implementation="eager").to(device).eval()
    _, heldout = fineweb_edu_chunks(tok, a.seq_len, 512, 256, Path(a.cache))
    pool = heldout[: a.n_seqs]
    T = pool.shape[1]
    layers = [int(x) for x in a.layers.split(",")]
    calib = torch.load(out / "calibration.pt")["amax"]  # (n_layers, d_ff) per-neuron max from calibration
    rng = np.random.default_rng(a.seed)
    abl = SingleNeuronAblator(model)

    # ---- pass 0: record activations of candidate neurons, and clean log-probs are recomputed per batch later
    cands = {}
    for L in layers:
        alive = np.where(calib[L].numpy() > 1e-3)[0]
        cands[L] = rng.choice(alive, size=a.per_layer, replace=False)
    acts = {L: [] for L in layers}
    with torch.no_grad():
        for L in layers:
            abl.record = (L, [], torch.as_tensor(cands[L], device=device))
            for s in range(0, len(pool), a.batch):
                ids = torch.from_numpy(pool[s:s + a.batch].astype(np.int64)).to(device)
                model(input_ids=ids)
            acts[L] = torch.cat(abl.record[1], 0)  # (n_seqs, T, per_layer)
            abl.record = None
    print(f"recorded activations for {len(layers)*a.per_layer} neurons over {pool.size} tokens")
    # numerical floor: KL between two clean passes on the same batch (should be ~0 in fp32)
    with torch.no_grad():
        ids = torch.from_numpy(pool[: a.batch].astype(np.int64)).to(device)
        lp1 = F.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
        lp2 = F.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
        floor = (lp1.exp() * (lp1 - lp2)).sum(-1)
        print(f"numerical floor: mean KL between two clean passes = {floor.mean().item():.2e} (max {floor.max().item():.2e})")
        del lp1, lp2

    # percentile thresholds per neuron (exclude position 0)
    bins_mask = {}  # (L, j) -> list of (n_seqs, T) bool masks per bin
    for L in layers:
        for k, j in enumerate(cands[L]):
            v = acts[L][:, :, k].abs().clone(); v[:, 0] = -1.0  # drop position 0 from the pool
            flat = v.flatten(); order = torch.argsort(flat, descending=True)
            n = int((flat >= 0).sum())
            ranks = torch.empty_like(order); ranks[order] = torch.arange(len(flat))
            masks = []
            for lo, hi, _ in BINS:
                m = (ranks >= int(lo * n)) & (ranks < int(hi * n)) & (flat >= 0)
                masks.append(m.view(v.shape))
            bins_mask[(L, int(j))] = masks

    # ---- main loop: per batch, clean log-probs once, then per (neuron, bin) an ablated pass
    rows = {}
    t0 = time.time()
    with torch.no_grad():
        for s in tqdm(range(0, len(pool), a.batch), desc="batches"):
            ids = torch.from_numpy(pool[s:s + a.batch].astype(np.int64)).to(device)
            clean_lp = F.log_softmax(model(input_ids=ids).logits.float(), dim=-1)  # (B,T,V) fp32
            clean_p = clean_lp.exp()
            for L in layers:
                for k, j in enumerate(cands[L]):
                    hvals = acts[L][s:s + a.batch, :, k].to(device)  # (B,T)
                    for b, (lo, hi, name) in enumerate(BINS):
                        m = bins_mask[(L, int(j))][b][s:s + a.batch].to(device)
                        if not m.any():
                            continue
                        abl.layer, abl.j, abl.mask = L, int(j), m
                        abl_lp = F.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
                        kl = (clean_p * (clean_lp - abl_lp)).sum(-1)  # (B,T) KL(clean||ablated)
                        key = (L, int(j), b)
                        r = rows.setdefault(key, dict(layer=L, neuron=int(j), bin=b, bin_name=name, n=0, sum_abs_h=0.0,
                                                      sum_h2=0.0, sum_kl=0.0, sum_kl_h2=0.0, n_out=0, sum_kl_out=0.0,
                                                      max_over_calib=float(calib[L][int(j)])))
                        hin = hvals[m].abs()
                        r["n"] += int(m.sum()); r["sum_abs_h"] += float(hin.sum()); r["sum_h2"] += float((hin ** 2).sum())
                        r["sum_kl"] += float(kl[m].sum()); r["sum_kl_h2"] += float((kl[m] / (hin ** 2 + 1e-12)).sum())
                        mo = (~m); mo[:, 0] = False
                        r["n_out"] += int(mo.sum()); r["sum_kl_out"] += float(kl[mo].sum())
            abl.layer = abl.j = abl.mask = None
            if s == 0:
                per_batch = time.time() - t0
                print(f"first batch {per_batch:.0f}s -> est total {per_batch * len(pool) / a.batch / 60:.0f} min")
    df = pd.DataFrame(rows.values())
    df["mean_abs_h"] = df.sum_abs_h / df.n; df["mean_h2"] = df.sum_h2 / df.n
    df["mean_kl"] = df.sum_kl / df.n; df["mean_kl_per_h2"] = df.sum_kl_h2 / df.n
    df["mean_kl_out"] = df.sum_kl_out / df.n_out.clip(lower=1)
    df["mean_h_over_max"] = df.mean_abs_h / df.max_over_calib
    df.to_csv(out / "neuron_influence.csv", index=False)
    print(df.groupby("bin_name")[["mean_h_over_max", "mean_kl", "mean_kl_per_h2", "mean_kl_out"]].median().to_string())
    print(f"done in {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
