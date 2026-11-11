"""Pre-freeze check 1 (Team 7-style): is the class dependence of Δ consistent across seeds, and beyond initialization?

Selective checkpoints: Phase 1 seeds 0-2, Check A seeds 3-5 (full input), Check B seeds 0-2 (0.3-3 Hz removed).
Each is compared with the same-seed untrained model (seed_everything(seed) + build_model reproduces train.py's init).
Statistic per layer: mean Δ over time and channels for each validation window, then
  ratio  = mean(dynamic windows) / mean(static windows)
  effect = (mean dyn - mean static) / pooled std   (window level)
Pre-freeze check 2: band-stop leaves the window mean (posture/gravity) of total_acc unchanged.
Validation only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:1]
src = (HERE / "evaluate_checks.py").read_text().split("res = {")[0]       # reuse helpers only
exec(compile(src, str(HERE / "evaluate_checks.py"), "exec"))
from models import build_model  # noqa: E402
from train import seed_everything  # noqa: E402

PHASE1_RUNS = Path(os.environ.get("PHASE1_RUNS", HERE.parent / "runs"))
DYN, STAT = [0, 1, 2], [3, 4, 5]


@torch.no_grad()
def window_delta(model, x):
    model.set_record(True)
    out = [[] for _ in model.layers]
    for i in range(0, len(x), 256):
        model(torch.from_numpy(x[i:i + 256]).to(dev))
        for j, rec in enumerate(model.records):
            out[j].append(rec["delta"].mean((1, 2)))               # (b,) mean over time and channels
    model.set_record(False)
    return [torch.cat(o).numpy() for o in out]


def summary(d, y):
    dyn, sta = d[np.isin(y, DYN)], d[np.isin(y, STAT)]
    pooled = np.sqrt((dyn.var(ddof=1) + sta.var(ddof=1)) / 2)
    return {"dyn": float(dyn.mean()), "static": float(sta.mean()), "ratio": float(dyn.mean() / sta.mean()),
            "effect": float((dyn.mean() - sta.mean()) / pooled),
            "by_class": [float(d[y == k].mean()) for k in range(6)]}


items = [("full", s, PHASE1_RUNS / f"mamba/seed{s}", None) for s in [0, 1, 2]] \
      + [("full", s, HERE / f"runs/A_mamba_s{s}", None) for s in [3, 4, 5]] \
      + [("lowoff", s, HERE / f"runs/B_mamba_s{s}", (0.3, 3)) for s in [0, 1, 2]]
res = {"delta": {}, "dc": {}}
for g, s, run_dir, base in items:
    model, ckpt = load_checkpoint(run_dir / f"seed_{s}" / "best.pt", dev)
    idx = ckpt["split"]["val_indices"]
    x_raw, y = x_all[idx], y_all[idx]
    if base:
        x_raw = bandstop(x_raw, *base)
    x = Standardizer.from_state(ckpt["normalizer"]).transform(x_raw)
    seed_everything(s)
    init = build_model(**ckpt["model_kwargs"]).to(dev).eval()
    tr, it = window_delta(model, x), window_delta(init, x)
    res["delta"][f"{g}_s{s}"] = {f"layer{j+1}": {"trained": summary(tr[j], y), "init": summary(it[j], y)}
                                 for j in range(len(tr))}
    print(g, s, " | ".join(f"L{j+1} trained ratio {summary(tr[j], y)['ratio']:.3f} eff {summary(tr[j], y)['effect']:+.2f}"
                            f" / init ratio {summary(it[j], y)['ratio']:.3f} eff {summary(it[j], y)['effect']:+.2f}"
                            for j in range(len(tr))), flush=True)

# check 2: window means of total_acc (channels 6-8 in the starter's channel order is verified below)
idx = load_checkpoint(PHASE1_RUNS / "mamba/seed0/seed_0/best.pt", "cpu")[1]["split"]["val_indices"]
xv, yv = x_all[idx], y_all[idx]
for lo, hi in [(0.3, 3), (0.3, 1), (3, 7)]:
    xf = bandstop(xv, lo, hi)
    m0, m1 = xv.mean(1), xf.mean(1)                                 # (n, 9) window means
    res["dc"][f"{lo}-{hi}"] = {"max_abs_mean_change_per_channel": np.abs(m1 - m0).max(0).tolist(),
                               "median_abs_mean_change_per_channel": np.median(np.abs(m1 - m0), 0).tolist(),
                               "channel_abs_mean": np.abs(m0).mean(0).tolist()}
    print("dc", lo, hi, "median |Δmean|", np.round(np.median(np.abs(m1 - m0), 0), 5), "vs |mean|", np.round(np.abs(m0).mean(0), 4))
(HERE / "delta_class_check.json").write_text(json.dumps(res, indent=1))
