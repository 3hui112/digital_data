#!/usr/bin/env bash
# Finish Phase 1 on one GPU: train selective Mamba + LTI (3 seeds each, starter defaults), then all analysis.
# usage: bash run_remaining.sh <GPU index (nvidia-smi)>     env: DATA, VENV (see run_phase1.sh)
set -euo pipefail
GPU=${1:?usage: bash run_remaining.sh <gpu>}
ROOT=$(cd "$(dirname "$0")" && pwd)
export DATA=${DATA:-$ROOT/data} VENV=${VENV:-$HOME/.venvs/mamba-dd}
bash "$ROOT/run_phase1.sh" "$GPU" mamba lti cnn lstm ema       # finished (model, seed) pairs are skipped
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
"$VENV/bin/python" phase1/signal_tasks.py > logs/signal_tasks.out 2>&1
"$VENV/bin/python" phase1/summarize.py | tee logs/summarize.out
"$VENV/bin/python" phase1/dissect.py > logs/dissect.out 2>&1
tar czf "$ROOT/phase1_outputs.tgz" runs logs figs results
echo "-> $ROOT/phase1_outputs.tgz"
