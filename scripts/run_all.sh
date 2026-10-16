#!/usr/bin/env bash
# Full sweep on Qwen3-1.7B-Base (one GPU, ~1-2 h), then analysis + figures.
set -euo pipefail
cd "$(dirname "$0")/.."
${PYTHON:-python3} -m weakabl.run --model Qwen/Qwen3-1.7B-Base --out results/qwen3-1.7b-base "$@"
${PYTHON:-python3} -m weakabl.analyze --results results/qwen3-1.7b-base --figures figures/qwen3-1.7b-base
