"""Phase 2 training run (final_protocol.md): one model, one seed, one filter condition.

Train and validation are filtered identically; the normalizer is fitted on the UNFILTERED training subset
(same as h2_check/train_bandstopped.py). --bands none trains on the original input (E1).
usage: python phase2/train_condition.py --model mamba [--lti] --seed 6 --bands 0.3-3 --out phase2/runs/E2_0.3-3_mamba_s6
"""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mamba_starter"))
from data import Standardizer, development_data  # noqa: E402
from train import parser as starter_parser, train_seed  # noqa: E402


def bandstop(x, lo, hi):
    sos = signal.butter(4, [lo, hi], btype="bandstop", fs=50, output="sos")
    return signal.sosfiltfilt(sos, x, axis=1).astype(np.float32)


def parse_bands(text):
    """'none' -> []; '0.3-3' -> [(0.3, 3.0)]; '0.3-3+3-7' -> cascade, applied left to right."""
    return [] if text == "none" else [tuple(float(v) for v in b.split("-")) for b in text.split("+")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mamba", "cnn"], required=True)
    p.add_argument("--lti", action="store_true")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--bands", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dt_min", type=float, default=None, help="E4 only; default keeps the starter value")
    p.add_argument("--dt_max", type=float, default=None, help="E4 only; default keeps the starter value")
    p.add_argument("--data_root", default=os.environ.get("DATA", str(REPO / "data")))
    a = p.parse_args()
    if a.lti and a.model != "mamba":
        raise ValueError("--lti applies to mamba only")

    argv = ["--dataset", "har", "--data_root", a.data_root, "--model", a.model, "--seeds", str(a.seed),
            "--threads", "1", "--out", a.out] + (["--lti"] if a.lti else [])
    argv += [f for k in ("dt_min", "dt_max") if getattr(a, k) is not None for f in (f"--{k}", str(getattr(a, k)))]
    cfg = vars(starter_parser().parse_args(argv))
    torch.set_num_threads(cfg["threads"])
    out = Path(a.out)
    if (out / "summary.json").exists():
        raise FileExistsError(f"{out} already contains a finished run")
    xtr, ytr, xva, yva, manifest = development_data(cfg)
    norm = Standardizer().fit(xtr)                         # unfiltered training subset only
    for lo, hi in parse_bands(a.bands):
        xtr, xva = bandstop(xtr, lo, hi), bandstop(xva, lo, hi)
    xtr, xva = norm.transform(xtr), norm.transform(xva)
    out.mkdir(parents=True, exist_ok=True)
    cfg.update(bandstop_train_val=parse_bands(a.bands), bandstop_filter="butter N=4 bandstop, sosfiltfilt, fs=50",
               normalizer="fit on unfiltered training subset", protocol="final_protocol.md",
               versions={"python": platform.python_version(), "torch": str(torch.__version__)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    result = train_seed(cfg, a.seed, xtr, ytr, xva, yva, norm, manifest)
    (out / "summary.json").write_text(json.dumps({"split": "validation (filtered as in config)", "runs": [result]},
                                                 indent=2) + "\n")
    print("DONE", json.dumps({k: result[k] for k in ["seed", "epoch", "accuracy", "macro_f1"]}))


if __name__ == "__main__":
    main()
