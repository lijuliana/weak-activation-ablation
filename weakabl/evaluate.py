"""Readouts: held-out LM loss (per token) and PopQA generation accuracy."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def lm_nll_per_token(model, chunks: np.ndarray, batch_size: int, device, desc: str = "lm") -> np.ndarray:
    """Return per-token NLL (float32) of shape (n_seqs, seq_len - 1): predicting token t+1 from <= t."""
    out = []
    for s in tqdm(range(0, len(chunks), batch_size), desc=desc, leave=False):
        ids = torch.from_numpy(chunks[s : s + batch_size].astype(np.int64)).to(device)
        logits = model(input_ids=ids).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
        out.append(nll.view(ids.shape[0], -1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def popqa_generate(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int, device,
                   desc: str = "popqa") -> list[str]:
    """Greedy generation; returns the text up to the first newline for each prompt."""
    gens: list[str] = []
    tokenizer.padding_side = "left"
    # Sort by length to reduce padding waste, then restore order.
    order = np.argsort([len(p) for p in prompts])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    sorted_prompts = [prompts[i] for i in order]
    for s in tqdm(range(0, len(sorted_prompts), batch_size), desc=desc, leave=False):
        batch = sorted_prompts[s : s + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for row in tokenizer.batch_decode(new, skip_special_tokens=True):
            gens.append(row.split("\n")[0].strip())
    return [gens[i] for i in inv]
