#!/usr/bin/env bash
# Tiny end-to-end run (small model, few sequences, 64 PopQA questions) to check the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
${PYTHON:-python3} -m weakabl.run --model Qwen/Qwen3-0.6B-Base --out results/smoke --n-calib 8 --n-heldout 4 \
  --wt103-max-seqs 4 --popqa-limit 64 --fracs 0,0.05,0.2 --random-seeds 0 --lm-batch 2 --popqa-batch 16
${PYTHON:-python3} -m weakabl.analyze --results results/smoke --figures figures/smoke
