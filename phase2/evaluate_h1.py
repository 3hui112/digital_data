"""E1 / H1 verdict on validation (final_protocol.md section 5), seeds 6-8.

Each full-input checkpoint is evaluated on the same validation windows with every E1 band-stop condition.
Drops are macro-F1 percentage points (accuracy and dynamic-class F1 are reported alongside), with the fraction
of validation signal power each filter removes.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.argv = sys.argv[:1]
src = (REPO / "h2_check/evaluate_checks.py").read_text().split("res = {")[0]       # reuse helpers only
exec(compile(src, str(REPO / "h2_check/evaluate_checks.py"), "exec"))

SEEDS, MODELS = [6, 7, 8], ["mamba", "lti", "cnn"]
LOW, MID, HIGH = [(0.3, 1), (1, 2), (2, 3)], [(3, 7), (7, 11)], [(11, 15), (15, 19)]
BANDS = LOW + MID + HIGH + [(3, 8)]
key = lambda b: f"{b[0]}-{b[1]}"

res = {m: {s: drops(REPO / f"phase2/runs/E1_{m}_s{s}", s, None, BANDS) for s in SEEDS} for m in MODELS}

idx = load_checkpoint(REPO / "phase2/runs/E1_mamba_s6/seed_6/best.pt", "cpu")[1]["split"]["val_indices"]
xv = x_all[idx] - x_all[idx].mean(1, keepdims=True)
energy = {key(b): float(1 - (bandstop(xv, *b) ** 2).sum() / (xv ** 2).sum()) for b in BANDS}


def seed_check(m, s):
    d = {key(b): res[m][s][key(b)]["macro_f1"] for b in BANDS}
    low = max(d[key(b)] for b in LOW)
    above = [d[key(b)] for b in MID + HIGH]
    high = [d[key(b)] for b in HIGH]
    support = all(low - x > DELTA for x in above) and all(x <= DELTA for x in high)
    against = any(x - low > DELTA for x in above) or any(x > DELTA for x in high)
    return support, against


checks = {m: [seed_check(m, s) for s in SEEDS] for m in MODELS}
if all(sup for m in MODELS for sup, _ in checks[m]):
    verdict = "SUPPORTED"
elif any(all(ag for _, ag in checks[m]) for m in MODELS):
    verdict = "FALSIFIED"
else:
    verdict = "INCONCLUSIVE"

lines = ["H1 (E1, validation, seeds 6-8): macro-F1 drop in pp; removed power fraction of the validation signal",
         "band      removed " + "  ".join(f"{m:>17}" for m in MODELS)]
for b in BANDS:
    cells = ["/".join(f"{res[m][s][key(b)]['macro_f1']:5.2f}" for s in SEEDS) for m in MODELS]
    lines.append(f"{key(b):>8}  {energy[key(b)]*100:6.2f}%  " + "  ".join(f"{c:>17}" for c in cells))
lines.append("reference macro-F1: " + "  ".join(
    f"{m} " + "/".join(f"{res[m][s]['reference']['macro_f1']:.2f}" for s in SEEDS) for m in MODELS))
lines.append("dynamic-class F1 drop (pp):")
for b in BANDS:
    lines.append(f"{key(b):>8}  " + "  ".join(
        f"{m} " + "/".join(f"{res[m][s][key(b)]['dyn_f1']:5.2f}" for s in SEEDS) for m in MODELS))
lines.append("per-seed (support, against): " + "  ".join(f"{m} {checks[m]}" for m in MODELS))
lines.append(f"H1 verdict: {verdict}")
text = "\n".join(lines)
print(text)
(REPO / "phase2/results").mkdir(exist_ok=True)
(REPO / "phase2/results/h1.txt").write_text(text + "\n")
(REPO / "phase2/results/h1.json").write_text(json.dumps(
    {"drops": {m: {str(s): v for s, v in r.items()} for m, r in res.items()}, "removed_power": energy,
     "checks": checks, "verdict": verdict}, indent=1))
