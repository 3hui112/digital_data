"""Evaluate Check A and Check B exactly as fixed in PREREGISTRATION.md (validation only)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STARTER = REPO / "mamba_starter"
sys.path.insert(0, str(STARTER))
from data import Standardizer, load_har  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402
from train import evaluate_model, make_loader  # noqa: E402

BANDS = [(0.3, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20)]
DYN, DELTA = [0, 1, 2], 1.0
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_all, y_all, _ = load_har(Path(os.environ.get("DATA", REPO / "data")), "train")


def bandstop(x, lo, hi):
    return signal.sosfiltfilt(signal.butter(4, [lo, hi], btype="bandstop", fs=50, output="sos"), x, axis=1).astype(np.float32)


def metrics(model, ckpt, x_raw, y):
    x = Standardizer.from_state(ckpt["normalizer"]).transform(x_raw)
    r = evaluate_model(model, make_loader(x, y, ckpt["config"], validation=True), dev, 6)
    cm = np.array(r["confusion_matrix"], dtype=float)
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1e-12)
    return {"acc": r["accuracy"] * 100, "macro_f1": r["macro_f1"] * 100, "dyn_f1": f1[DYN].mean() * 100}


def drops(run_dir, seed, base_band, test_bands):
    model, ckpt = load_checkpoint(run_dir / f"seed_{seed}" / "best.pt", dev)
    idx = ckpt["split"]["val_indices"]
    x_raw, y = x_all[idx], y_all[idx]
    if base_band:
        x_raw = bandstop(x_raw, *base_band)
    ref = metrics(model, ckpt, x_raw, y)
    out = {"reference": ref}
    for lo, hi in test_bands:
        m = metrics(model, ckpt, bandstop(x_raw, lo, hi), y)
        out[f"{lo}-{hi}"] = {k: ref[k] - m[k] for k in ref}
    return out


def verdict(q):
    return "SUPPORTED" if all(v > DELTA for v in q) else "FALSIFIED" if all(v < -DELTA for v in q) else "INCONCLUSIVE"


res = {"A": {}, "B": {}}
for m in ["mamba", "lti"]:
    for s in [3, 4, 5]:
        res["A"][f"{m}_s{s}"] = drops(HERE / f"runs/A_{m}_s{s}", s, None, BANDS)
    for s in [0, 1, 2]:
        res["B"][f"{m}_s{s}"] = drops(HERE / f"runs/B_{m}_s{s}", s, (0.3, 3), [(3, 5), (5, 8)])

lines = []
for check, seeds, bands in [("A", [3, 4, 5], ["1-2"]), ("B", [0, 1, 2], ["3-5", "5-8"])]:
    lines.append(f"== Check {check} (seeds {seeds}) ==")
    for m in ["mamba", "lti"]:
        ref = np.array([[res[check][f"{m}_s{s}"]["reference"][k] for k in ["acc", "macro_f1", "dyn_f1"]] for s in seeds])
        lines.append(f"  {m:5} reference acc {ref[:,0].mean():.2f}±{ref[:,0].std(ddof=1):.2f}  macro-F1 {ref[:,1].mean():.2f}±{ref[:,1].std(ddof=1):.2f}  dyn-F1 {ref[:,2].mean():.2f}")
    for metric in ["dyn_f1", "macro_f1", "acc"]:
        q = [np.mean([res[check][f"mamba_s{s}"][b][metric] - res[check][f"lti_s{s}"][b][metric] for b in bands]) for s in seeds]
        tag = verdict(q) if metric == "dyn_f1" else "(reported only)"
        lines.append(f"  q_s [{metric:8}] = {[round(v, 2) for v in q]}  -> {tag}")
    for b in bands:
        for m in ["mamba", "lti"]:
            v = [res[check][f"{m}_s{s}"][b]["dyn_f1"] for s in seeds]
            lines.append(f"  dyn-F1 drop {b:>4} Hz {m:5}: {[round(x, 2) for x in v]}")
lines.append("== Check A: all bands, macro-F1 drop mean over seeds 3-5 ==")
for b in [f"{lo}-{hi}" for lo, hi in BANDS]:
    lines.append(f"  {b:>7} Hz  " + "  ".join(f"{m} {np.mean([res['A'][f'{m}_s{s}'][b]['macro_f1'] for s in [3,4,5]]):6.2f}" for m in ["mamba", "lti"]))
text = "\n".join(lines)
print(text)
(HERE / "results.txt").write_text(text + "\n")
(HERE / "results.json").write_text(json.dumps(res, indent=1))
