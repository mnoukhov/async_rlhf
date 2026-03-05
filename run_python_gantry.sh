#!/usr/bin/env bash
set -euo pipefail

IMAGE="ai2/cuda12.8-dev-ubuntu22.04-notorch"
WEKA_MOUNT="oe-adapt-default:/weka/oe-adapt-default"
DEFAULT_PYTHON="3.10"

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo no-git-branch)"
branch="${branch//\//-}"
ts="$(date -u +%Y%m%d-%H%M%S)"
name="${branch}-${ts}"

gantry run \
  --beaker-image "$IMAGE" \
  --weka="$WEKA_MOUNT" \
  --uv-all-extras \
  --show-logs \
  --default-python-version "$DEFAULT_PYTHON" \
  --name "$name" \
  -- ./train_gen_eval_gsm8k_mila.sh "$@"
