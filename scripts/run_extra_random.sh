#!/usr/bin/env bash
# Extra control: zero a fixed fraction of activations uniformly at random (not count-matched),
# to cover the low-damage regime for the matched-damage comparison. Appends to the same results dir.
set -euo pipefail
cd "$(dirname "$0")/.."
${PYTHON:-python3} -m weakabl.run --model Qwen/Qwen3-1.7B-Base --out results/qwen3-1.7b-base \
  --fracs 0 --random-fracs 0.005,0.01,0.02,0.05,0.1 "$@"
${PYTHON:-python3} -m weakabl.analyze --results results/qwen3-1.7b-base --figures figures/qwen3-1.7b-base
