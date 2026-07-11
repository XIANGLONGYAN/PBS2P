#!/usr/bin/env bash
set -euo pipefail

# Set these paths before running.
MODEL=${MODEL:-/path/to/Llama-2-7b-hf}
DATA_ROOT=${DATA_ROOT:-./data}
DEVICE=${DEVICE:-cuda:0}

# Paper configuration: PBS2P with SPBO, CFS, and 6:8 semi-structured sparsity.
python run_pbs2p.py "$MODEL" c4 \
  --data-root "$DATA_ROOT" \
  --device "$DEVICE" \
  --n 6 \
  --m 8 \
  --cfs \
  --results-file results/llama2-7b-6x8.json \
  --log-file logs/llama2-7b-6x8.log

# SPBO ablation (vanilla one-shot prune-then-binarize):
# python run_pbs2p.py "$MODEL" c4 --data-root "$DATA_ROOT" \
#   --device "$DEVICE" --n 6 --m 8 --cfs --ablation one-shot