#!/usr/bin/env bash
# Scale the weak tail instead of zeroing it: does the network read sub-threshold activations at all?
set -euo pipefail
cd "$(dirname "$0")/.."
${PYTHON:-python3} -m weakabl.run --model Qwen/Qwen3-1.7B-Base --out results/qwen3-1.7b-base \
  --fracs 0 --scale-fracs 0.005,0.015 --scales 0.5,2,4 "$@"
