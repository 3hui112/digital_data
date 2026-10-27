"""Shared setup for team analysis scripts: import the (unmodified) course starter, thread limits, paths."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # digital_data/
STARTER = ROOT / "mamba_starter"
sys.path.insert(0, str(STARTER))

import numpy as np
import torch

torch.set_num_threads(1)

from data import HAR_CHANNELS, HAR_CLASSES, load_har, subject_split  # noqa: E402

DATA = Path(os.environ.get("DATA", ROOT / "data"))
RUNS = Path(os.environ.get("RUNS", ROOT / "runs"))
FIG, RES = ROOT / "figs", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
FS = 50.0
GROUPS = {"body_acc": [0, 1, 2], "body_gyro": [3, 4, 5], "total_acc": [6, 7, 8]}
SHORT = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]


def dev_split():
    """Starter development split (split_seed=42, val_fraction=0.2): raw, unnormalised train/val windows."""
    x, y, s = load_har(DATA)
    tr, va = subject_split(s, 0.2, 42)
    return dict(X_train=x[tr], y_train=y[tr], X_val=x[va], y_val=y[va],
                val_subjects=np.unique(s[va]).tolist())
