"""Datasets: generic calibration / held-out text, and PopQA with popularity bins."""
from __future__ import annotations

import os
import re
import string
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

# ----------------------------------------------------------------------------------
# Generic text
# ----------------------------------------------------------------------------------

def _chunk_token_stream(token_lists, seq_len: int, n_seqs: int) -> np.ndarray:
    """Concatenate documents (separated by nothing) and cut into fixed-length chunks."""
    buf: list[int] = []
    out = []
    for toks in token_lists:
        buf.extend(toks)
        while len(buf) >= seq_len and len(out) < n_seqs:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(out) >= n_seqs:
            break
    assert len(out) == n_seqs, f"only got {len(out)} / {n_seqs} sequences"
    return np.asarray(out, dtype=np.int32)


def fineweb_edu_chunks(tokenizer, seq_len: int, n_calib: int, n_heldout: int, cache_dir: Path,
                       skip_docs: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint calibration and held-out chunks from HuggingFaceFW/fineweb-edu (sample-10BT),
    read in stream order so the split is deterministic. Cached as .npy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fwedu_s{seq_len}_c{n_calib}_h{n_heldout}_skip{skip_docs}"
    fc, fh = cache_dir / f"{tag}_calib.npy", cache_dir / f"{tag}_heldout.npy"
    if fc.exists() and fh.exists():
        return np.load(fc), np.load(fh)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    it = iter(ds)
    for _ in range(skip_docs):
        next(it)

    def tok_iter():
        for ex in it:
            yield tokenizer(ex["text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    gen = tok_iter()
    calib = _chunk_token_stream(gen, seq_len, n_calib)
    heldout = _chunk_token_stream(gen, seq_len, n_heldout)  # continues the same iterator: disjoint docs
    np.save(fc, calib)
    np.save(fh, heldout)
    return calib, heldout


def wikitext103_test_chunks(tokenizer, seq_len: int, cache_dir: Path, max_seqs: int | None = None) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"wt103test_s{seq_len}_m{max_seqs}.npy"
    if f.exists():
        return np.load(f)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "".join(ds["text"])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(ids) // seq_len
    if max_seqs is not None:
        n = min(n, max_seqs)
    arr = np.asarray(ids[: n * seq_len], dtype=np.int32).reshape(n, seq_len)
    np.save(f, arr)
    return arr


# ----------------------------------------------------------------------------------
# PopQA
# ----------------------------------------------------------------------------------

def load_popqa(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / "popqa_test.parquet"
    if f.exists():
        return pd.read_parquet(f)
    ds = load_dataset("akariasai/PopQA", split="test")
    df = ds.to_pandas()
    df.to_parquet(f)
    return df


def popularity_bins(s_pop: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, list[str]]:
    """Equal-count quantile bins on log10 subject popularity (monthly Wikipedia page views).
    Returns bin index per question (0 = rarest) and human-readable labels."""
    logp = np.log10(np.maximum(s_pop.astype(float), 1.0))
    edges = np.quantile(logp, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, logp, side="right") - 1, 0, n_bins - 1)
    labels = [f"10^{edges[i]:.1f}-10^{edges[i+1]:.1f}" for i in range(n_bins)]
    return idx, labels


def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(generation: str, possible_answers: list[str]) -> bool:
    """PopQA metric (Mallen et al. 2023): any gold alias appears as a substring of the generation."""
    g = normalize_answer(generation)
    return any(normalize_answer(a) and normalize_answer(a) in g for a in possible_answers)


def parse_possible_answers(x) -> list[str]:
    import json
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return [str(a) for a in v] if isinstance(v, list) else [str(v)]
        except json.JSONDecodeError:
            return [x]
    return [str(a) for a in x]


def build_fewshot_prefix(df: pd.DataFrame, n_shot: int, seed: int) -> tuple[str, np.ndarray]:
    """Fixed exemplars drawn from PopQA (excluded from evaluation). Returns prefix and exemplar ids."""
    rng = np.random.default_rng(seed)
    ex_idx = rng.choice(len(df), size=n_shot, replace=False)
    parts = []
    for i in ex_idx:
        q = df.iloc[i]["question"]
        a = parse_possible_answers(df.iloc[i]["possible_answers"])[0]
        parts.append(f"Q: {q}\nA: {a}\n\n")
    return "".join(parts), df.iloc[ex_idx]["id"].to_numpy()
