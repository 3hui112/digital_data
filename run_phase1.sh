#!/usr/bin/env bash
# Phase 1 Task 1.3/1.4/1.5 with the course starter (mamba_starter/, unmodified, default hyperparameters).
# One process per (model, seed) so seeds run in parallel; each writes runs/<model>/seed<S>/.
#
# usage:  bash run_phase1.sh <GPU index (nvidia-smi)> <model ...>
#         models: mamba lti cnn lstm ema
#   e.g.  bash run_phase1.sh 0 mamba            # other server: selective Mamba x 3 seeds
#         bash run_phase1.sh 4 lti cnn lstm ema
# env:    VENV (default ~/.venvs/mamba-dd), DATA (default ./data; HAR is downloaded if missing), SEEDS (default "0 1 2")
set -euo pipefail
GPU=${1:?usage: bash run_phase1.sh <gpu> <model ...>}; shift
ROOT=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$HOME/.venvs/mamba-dd}
DATA=${DATA:-$ROOT/data}
SEEDS=${SEEDS:-"0 1 2"}
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  if command -v uv >/dev/null; then
    uv venv "$VENV" --python 3.12 -q
    VIRTUAL_ENV=$VENV uv pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    VIRTUAL_ENV=$VENV uv pip install -q -r "$ROOT/mamba_starter/requirements.txt"
  else
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    "$VENV/bin/pip" install -q -r "$ROOT/mamba_starter/requirements.txt"
  fi
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT/mamba_starter"
if [ ! -d "$DATA/UCI HAR Dataset" ]; then
  "$PY" -c "from data import download_har; download_har('$DATA')"
fi
mkdir -p "$ROOT/runs" "$ROOT/logs"

pids=()
for m in "$@"; do
  for s in $SEEDS; do
    out="$ROOT/runs/$m/seed$s"; log="$ROOT/logs/${m}_s$s.log"
    if [ -f "$out/summary.json" ]; then echo "skip $m s$s (done)"; continue; fi
    rm -rf "$out"
    case $m in
      mamba) args=(--model mamba) ;;
      lti)   args=(--model mamba --lti) ;;
      cnn)   args=(--model cnn) ;;
      lstm)  args=(--model lstm) ;;
      ema)   "$PY" train_ema.py --data_root "$DATA" --seeds $s --out "$out" > "$log" 2>&1 & pids+=($!); continue ;;
      *) echo "unknown model $m"; exit 1 ;;
    esac
    "$PY" train.py --dataset har --data_root "$DATA" "${args[@]}" --seeds $s --threads 1 --out "$out" > "$log" 2>&1 &
    pids+=($!)
  done
done
echo "launched ${#pids[@]} jobs on GPU $GPU: ${pids[*]}"
wait
grep -h '"mean"' -B1 "$ROOT"/runs/*/seed*/summary.json >/dev/null 2>&1 || true
echo "done. results: $ROOT/runs/<model>/seed<S>/summary.json"
