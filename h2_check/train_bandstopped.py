"""Check B: retrain a starter model on band-stopped HAR (train + validation filtered identically).

Uses the starter unmodified: development_data / Standardizer / train_seed are imported. The normalizer is fitted on
the UNFILTERED training subset, so filtered and unfiltered inputs share one input scaling.
usage: python train_bandstopped.py --lo 0.3 --hi 3 --seed 0 --out runs/B_mamba_s0 [--lti]
"""
import argparse
import os
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
STARTER = REPO / "mamba_starter"
sys.path.insert(0, str(STARTER))
from data import Standardizer, development_data  # noqa: E402
from train import parser as starter_parser, train_seed  # noqa: E402


def bandstop(x, lo, hi):
    sos = signal.butter(4, [lo, hi], btype="bandstop", fs=50, output="sos")
    return signal.sosfiltfilt(sos, x, axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lo", type=float, required=True)
    p.add_argument("--hi", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lti", action="store_true")
    p.add_argument("--data_root", default=os.environ.get("DATA", str(REPO / "data")))
    a = p.parse_args()

    argv = ["--dataset", "har", "--data_root", a.data_root, "--model", "mamba", "--seeds", str(a.seed),
            "--threads", "1", "--out", a.out] + (["--lti"] if a.lti else [])
    cfg = vars(starter_parser().parse_args(argv))
    torch.set_num_threads(cfg["threads"])
    out = Path(a.out)
    if (out / "config.json").exists():
        raise FileExistsError(f"{out} already contains a run")
    xtr, ytr, xva, yva, manifest = development_data(cfg)
    norm = Standardizer().fit(xtr)                         # unfiltered training subset only
    xtr, xva = norm.transform(bandstop(xtr, a.lo, a.hi)), norm.transform(bandstop(xva, a.lo, a.hi))
    out.mkdir(parents=True, exist_ok=True)
    cfg.update(bandstop_train_val=[a.lo, a.hi], bandstop_filter="butter N=4 bandstop, sosfiltfilt, fs=50",
               normalizer="fit on unfiltered training subset",
               versions={"python": platform.python_version(), "torch": str(torch.__version__)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    result = train_seed(cfg, a.seed, xtr, ytr, xva, yva, norm, manifest)
    (out / "summary.json").write_text(json.dumps({"split": "validation (band-stopped)", "runs": [result]}, indent=2) + "\n")
    print("DONE", json.dumps({k: result[k] for k in ["seed", "epoch", "accuracy", "macro_f1"]}))


if __name__ == "__main__":
    main()
