#!/usr/bin/env bash
# Launch Check A (GPU $1) and Check B (GPU $2): 6 runs each, all in parallel. Logs in logs/.
set -euo pipefail
cd "$(dirname "$0")"
GA=${1:?gpu for A}; GB=${2:?gpu for B}
PY=${VENV:-$HOME/.venvs/mamba-dd}/bin/python
STARTER=$(cd .. && pwd)/mamba_starter
export DATA=${DATA:-$(cd .. && pwd)/data}
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs runs
pids=()
for s in 3 4 5; do       # Check A: new seeds, unfiltered training
  CUDA_VISIBLE_DEVICES=$GA $PY $STARTER/train.py --dataset har --data_root $DATA --model mamba --seeds $s --threads 1 \
    --out runs/A_mamba_s$s > logs/A_mamba_s$s.log 2>&1 & pids+=($!)
  CUDA_VISIBLE_DEVICES=$GA $PY $STARTER/train.py --dataset har --data_root $DATA --model mamba --lti --seeds $s --threads 1 \
    --out runs/A_lti_s$s > logs/A_lti_s$s.log 2>&1 & pids+=($!)
done
for s in 0 1 2; do       # Check B: retrain with 0.3-3 Hz removed from train and validation
  CUDA_VISIBLE_DEVICES=$GB $PY train_bandstopped.py --lo 0.3 --hi 3 --seed $s --out runs/B_mamba_s$s > logs/B_mamba_s$s.log 2>&1 & pids+=($!)
  CUDA_VISIBLE_DEVICES=$GB $PY train_bandstopped.py --lo 0.3 --hi 3 --seed $s --lti --out runs/B_lti_s$s > logs/B_lti_s$s.log 2>&1 & pids+=($!)
done
echo "launched ${#pids[@]} runs: ${pids[*]}"
wait
echo ALL_TRAINING_DONE
