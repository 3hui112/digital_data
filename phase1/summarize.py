"""Task 1.4/1.5 table: validation mean ± sample std over seeds, mean row-normalised confusion matrices."""
from common import FIG, RES, RUNS, SHORT

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = [("mamba", "Mamba (selective)"), ("lti", "Mamba-LTI"), ("cnn", "CNN"), ("lstm", "LSTM"), ("ema", "EMA bank + LogReg")]
rows, cms = {}, {}
for key, label in MODELS:
    runs = []
    for s in sorted((RUNS / key).glob("seed*/summary.json")):
        runs += json.loads(s.read_text())["runs"]
    if not runs:
        continue
    acc, f1 = np.array([r["accuracy"] for r in runs]), np.array([r["macro_f1"] for r in runs])
    cm = np.mean([np.array(r["confusion_matrix"]) / np.array(r["confusion_matrix"]).sum(1, keepdims=True) for r in runs], 0)
    sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else None
    rows[label] = dict(seeds=[r["seed"] for r in runs], acc_mean=float(acc.mean()), acc_std=sd(acc),
                       f1_mean=float(f1.mean()), f1_std=sd(f1), params=runs[0].get("parameters"),
                       best_epochs=[r.get("epoch") for r in runs], sit_as_stand=float(cm[3, 4]), stand_as_sit=float(cm[4, 3]))
    cms[label] = cm

print(f"{'model':<22}{'seeds':>8}{'val acc (%)':>18}{'val macro-F1 (%)':>20}{'params':>9}")
for k, r in rows.items():
    f = lambda m, s: f"{m*100:.2f} ± {s*100:.2f}" if s is not None else f"{m*100:.2f}"
    print(f"{k:<22}{len(r['seeds']):>8}{f(r['acc_mean'], r['acc_std']):>18}{f(r['f1_mean'], r['f1_std']):>20}{str(r['params']):>9}")

fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4))
for ax, (k, cm) in zip(np.atleast_1d(axes), cms.items()):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if cm[i, j] >= .005:
                ax.text(j, i, f"{cm[i,j]*100:.1f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > .5 else "black")
    ax.set_xticks(range(6), SHORT, rotation=45, fontsize=7); ax.set_yticks(range(6), SHORT, fontsize=7)
    ax.set_title(k, fontsize=9); ax.set_xlabel("pred"); ax.set_ylabel("true")
fig.suptitle("Validation confusion matrices (row-normalised %, mean over seeds)")
fig.tight_layout(); fig.savefig(FIG / "t14_confusion.png", dpi=130); plt.close(fig)
(RES / "phase1_models.json").write_text(json.dumps(rows, indent=1))
