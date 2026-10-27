"""Quick E1 check: evaluation-time band-stop drops on existing Phase 1 checkpoints (validation only)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal

REPO = Path(__file__).resolve().parents[2]
STARTER = REPO / "mamba_starter"
RUNS = REPO / "runs"                      # Phase 1 checkpoints (best.pt are not in git; retrain with run_phase1.sh)
DATA = Path(os.environ.get("DATA", REPO / "data"))
sys.path.insert(0, str(STARTER))
from data import Standardizer, load_har  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402
from train import evaluate_model, make_loader  # noqa: E402

BANDS = [(0.3, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20)]
DYN = [0, 1, 2]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_all, y_all, _ = load_har(DATA, "train")


def run(model, ckpt, x_raw, y):
    x = Standardizer.from_state(ckpt["normalizer"]).transform(x_raw)
    r = evaluate_model(model, make_loader(x, y, ckpt["config"], validation=True), dev, 6)
    cm = np.array(r["confusion_matrix"], dtype=float)
    tp = np.diag(cm); f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-12)
    return r["accuracy"] * 100, r["macro_f1"] * 100, f1[DYN].mean() * 100


out = {}
for m in ["mamba", "lti", "cnn"]:
    for s in range(3):
        model, ckpt = load_checkpoint(RUNS / m / f"seed{s}" / f"seed_{s}" / "best.pt", dev)
        idx = ckpt["split"]["val_indices"]
        x_raw, y = x_all[idx], y_all[idx]
        full = run(model, ckpt, x_raw, y)
        for lo, hi in BANDS:
            sos = signal.butter(4, [lo, hi], btype="bandstop", fs=50, output="sos")
            xf = signal.sosfiltfilt(sos, x_raw, axis=1).astype(np.float32)
            r = run(model, ckpt, xf, y)
            out.setdefault(m, {}).setdefault(f"{lo}-{hi}", []).append([full[i] - r[i] for i in range(3)])
        print(m, s, "full acc/F1", [round(v, 2) for v in full], flush=True)

print("\ndrop = full - bandstop, mean ± std over 3 seeds (percentage points)")
print(f"{'model':6}{'band':>9}{'acc drop':>16}{'F1 drop':>16}{'dyn-F1 drop':>16}  per-seed F1 drops")
for m, bands in out.items():
    for b, v in bands.items():
        v = np.array(v)
        cells = "".join(f"{v[:, i].mean():>9.2f} ± {v[:, i].std(ddof=1):4.2f}" for i in range(3))
        print(f"{m:6}{b:>9}{cells}  {np.round(v[:, 1], 2).tolist()}")
json.dump(out, open(Path(__file__).with_name("e1_quick.json"), "w"), indent=1)
