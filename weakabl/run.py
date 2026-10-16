"""Run the full sweep: calibrate, then for each condition measure held-out LM loss and PopQA.

Example (full run on a GPU):
    python -m weakabl.run --model Qwen/Qwen3-1.7B-Base --out results/qwen3-1.7b-base

Each condition is checkpointed to its own files; re-running skips finished conditions.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ablation import Ablator
from .data import (build_fewshot_prefix, fineweb_edu_chunks, is_correct, load_popqa, parse_possible_answers,
                   popularity_bins, wikitext103_test_chunks)
from .evaluate import lm_nll_per_token, popqa_generate


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--out", default="results/qwen3-1.7b-base")
    p.add_argument("--cache", default="data_cache")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--n-calib", type=int, default=512, help="calibration sequences (x seq-len tokens)")
    p.add_argument("--n-heldout", type=int, default=256, help="held-out FineWeb-edu sequences")
    p.add_argument("--wt103-max-seqs", type=int, default=240, help="held-out WikiText-103 test sequences")
    p.add_argument("--fracs", default="0,0.001,0.0025,0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3",
                   help="thresholds as fraction of each neuron's calibrated max |activation|")
    p.add_argument("--random-seeds", default="0,1", help="seeds for the count-matched random control")
    p.add_argument("--lm-batch", type=int, default=8)
    p.add_argument("--popqa-batch", type=int, default=96)
    p.add_argument("--n-shot", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--popqa-limit", type=int, default=0, help="debug: evaluate only N PopQA questions")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    print(f"device={device} dtype={dtype} model={args.model}")
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()
    ablator = Ablator(model, seed=args.seed)

    # ---- data --------------------------------------------------------------------
    calib, heldout = fineweb_edu_chunks(tok, args.seq_len, args.n_calib, args.n_heldout, cache)
    wt103 = wikitext103_test_chunks(tok, args.seq_len, cache, max_seqs=args.wt103_max_seqs)
    print(f"calib {calib.shape}, heldout fineweb {heldout.shape}, wikitext103 {wt103.shape}")

    df = load_popqa(cache)
    prefix, ex_ids = build_fewshot_prefix(df, args.n_shot, seed=args.seed)
    df = df[~df["id"].isin(ex_ids)].reset_index(drop=True)
    if args.popqa_limit:
        df = df.sample(n=args.popqa_limit, random_state=args.seed).reset_index(drop=True)
    bins, bin_labels = popularity_bins(df["s_pop"].to_numpy(), n_bins=args.n_bins)
    df["bin"] = bins
    prompts = [prefix + f"Q: {q}\nA:" for q in df["question"]]
    answers = [parse_possible_answers(a) for a in df["possible_answers"]]
    df[["id", "subj", "prop", "s_pop", "o_pop", "bin", "question"]].to_csv(out / "popqa_questions.csv", index=False)
    (out / "popqa_meta.json").write_text(json.dumps({"bin_labels": bin_labels, "n_shot": args.n_shot,
                                                      "exemplar_ids": [int(i) for i in ex_ids], "n": len(df)}, indent=2))
    print(f"PopQA: {len(df)} questions, bins (rare->popular): {bin_labels}")

    # ---- calibration ----------------------------------------------------------------
    calib_f = out / "calibration.pt"
    if calib_f.exists():
        c = torch.load(calib_f)
        amax = c["amax"]
        print("loaded calibration")
    else:
        t0 = time.time()
        ablator.set("calibrate")
        with torch.no_grad():
            for s in range(0, len(calib), args.lm_batch):
                ids = torch.from_numpy(calib[s : s + args.lm_batch].astype(np.int64)).to(device)
                model(input_ids=ids)
        amax = ablator.calibration_tensor(include_pos0=False)
        amax0 = ablator.calibration_tensor(include_pos0=True)
        torch.save({"amax": amax, "amax_incl_pos0": amax0, "n_tokens": int(calib.size)}, calib_f)
        print(f"calibrated on {calib.size} tokens in {time.time()-t0:.0f}s; "
              f"median per-neuron max={amax.median():.3f}, max={amax.max():.3f}, "
              f"neurons with pos0 max > 2x = {(amax0 > 2*amax).float().mean():.4f}")
    ablator.load_calibration(amax, device)

    # ---- conditions -------------------------------------------------------------------
    fracs = [float(x) for x in args.fracs.split(",")]
    seeds = [int(s) for s in args.random_seeds.split(",")]
    conds = [("off", 0.0, 0)]
    for f in fracs:
        if f > 0:
            conds.append(("threshold", f, 0))
            for s in seeds:
                conds.append(("random", f, s))

    summary_f = out / "summary.jsonl"
    done = set()
    if summary_f.exists():
        done = {json.loads(l)["cond"] for l in summary_f.read_text().splitlines() if l.strip()}

    for mode, frac, seed in conds:
        cond = "baseline" if mode == "off" else f"{mode}_f{frac:g}_s{seed}"
        if cond in done:
            print(f"skip {cond} (done)")
            continue
        t0 = time.time()
        ablator.set(mode, frac, seed=seed, track=True)
        nll_fw = lm_nll_per_token(model, heldout, args.lm_batch, device, desc=f"{cond} fineweb")
        sparsity_layers = ablator.stats.fraction_per_layer()
        sparsity = ablator.stats.fraction()
        ablator.set(mode, frac, seed=seed, track=False)
        nll_wt = lm_nll_per_token(model, wt103, args.lm_batch, device, desc=f"{cond} wt103")
        np.save(out / f"nll_fineweb_{cond}.npy", nll_fw.astype(np.float16))
        np.save(out / f"nll_wt103_{cond}.npy", nll_wt.astype(np.float16))

        ablator.set(mode, frac, seed=seed, track=False)
        gens = popqa_generate(model, tok, prompts, args.popqa_batch, args.max_new_tokens, device, desc=f"{cond} popqa")
        correct = np.array([is_correct(g, a) for g, a in zip(gens, answers)], dtype=bool)
        np.save(out / f"popqa_correct_{cond}.npy", correct)
        with gzip.open(out / f"popqa_gens_{cond}.jsonl.gz", "wt") as fh:
            for i, g in enumerate(gens):
                fh.write(json.dumps({"id": int(df.iloc[i]["id"]), "gen": g, "correct": bool(correct[i])}) + "\n")

        acc_by_bin = [float(correct[bins == b].mean()) for b in range(args.n_bins)]
        row = {"cond": cond, "mode": mode, "frac": frac, "seed": seed,
               "fineweb_nll": float(nll_fw.mean()), "wt103_nll": float(nll_wt.mean()),
               "sparsity": sparsity, "sparsity_per_layer": sparsity_layers,
               "popqa_acc": float(correct.mean()), "popqa_acc_by_bin": acc_by_bin,
               "seconds": time.time() - t0}
        with summary_f.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{cond}: fw_nll={row['fineweb_nll']:.4f} wt_nll={row['wt103_nll']:.4f} "
              f"sparsity={sparsity:.3f} popqa={row['popqa_acc']:.4f} by_bin={[round(a,3) for a in acc_by_bin]} "
              f"({row['seconds']:.0f}s)")

    ablator.remove()
    print("done")


if __name__ == "__main__":
    main()
