# Do weak MLP activations carry important information?

A small, reproducible test of a disputed claim. Goodfire's essay *Neural Networks Are Already Clean*
(Cammarata, Oct 2026) argues that weak neuron activations can be removed at little cost, so neurons
are not hiding superposed information there. Neel Nanda's reply predicts the opposite: weak
activations carry superposed facts, so removing them should disproportionately hurt rare, long-tail
knowledge. Neither side has run the experiment on a language model (see `docs/novelty_check.md`).

## Experiment

- **Model:** `Qwen/Qwen3-1.7B-Base`, bf16, greedy decoding. Base model so there is no chat template or
  thinking mode in the way of a plain few-shot factual-recall measurement.
- **Neuron activation:** the SwiGLU output `act(gate(x)) * up(x)`, i.e. the input to `down_proj`, in
  every one of the 28 MLP layers (6144 neurons each).
- **Calibration:** per-neuron max |activation| over 512 x 1024 tokens of FineWeb-edu, excluding
  position 0 of each sequence so first-token massive activations do not set the scale.
- **Weak-activation ablation:** zero every activation with |a| < X% of its neuron's calibrated max,
  for X in {1, 2, 3, 5, 7.5, 10, 15, 20, 30}.
- **Control:** for every (layer, token) zero the *same number* of activations chosen uniformly at
  random (two seeds). This separates "weak activations matter" from "rare facts are fragile to any
  perturbation".
- **Readouts:** (a) held-out LM loss on disjoint FineWeb-edu text and on WikiText-103 test;
  (b) PopQA accuracy (8-shot, substring match against gold aliases, as in Mallen et al. 2023), split
  into five equal-count bins of subject popularity (monthly Wikipedia page views).
- **Key statistics:** per-bin accuracy retention (accuracy / baseline accuracy) and forget rate
  (P(wrong | correct at baseline)); the rare-minus-popular gap in each; 95% bootstrap CIs over
  questions. The decisive comparison is the gap under weak-activation ablation versus under the
  count-matched random control, at matched X and at matched LM-loss damage.

## Layout

```
weakabl/ablation.py   forward-pre-hooks on down_proj: calibration, threshold and random ablation
weakabl/data.py       FineWeb-edu / WikiText-103 chunks, PopQA loading, popularity bins, prompts, metric
weakabl/evaluate.py   per-token LM loss; batched greedy PopQA generation
weakabl/run.py        the sweep (checkpointed per condition; re-running resumes)
weakabl/analyze.py    metrics.csv with bootstrap CIs, and the figures
scripts/run_all.sh    full pipeline;  scripts/smoke.sh  tiny end-to-end check on Qwen3-0.6B-Base
docs/novelty_check.md literature scan and gap analysis;  docs/refs.bib  verified bibliography
results/              summary.jsonl, per-condition PopQA correctness + generations, metrics.csv
figures/              fig1-fig4
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
bash scripts/smoke.sh          # ~5 min on a laptop, checks the pipeline end to end
bash scripts/run_all.sh        # ~1-2 h on one L40S / L4 class GPU
```

All randomness is seeded (`--seed`, `--random-seeds`); data splits are taken in stream order and
cached under `data_cache/`.

## Results

_(filled in below once the sweep has run)_
