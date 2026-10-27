"""EXPLORATORY (not in PREREGISTRATION.md): remove 3–8 Hz in one band-stop, on all existing checkpoints.

Added after Check A/B results were seen, on request. Same filter and metrics as evaluate_checks.py; validation only.
- pilot : Phase 1 checkpoints (selective, LTI, CNN; seeds 0–2), reference = unfiltered validation
- A     : new seeds 3–5 (selective, LTI), reference = unfiltered validation
- B     : 0.3–3 Hz-removed retraining (selective, LTI; seeds 0–2), reference = 0.3–3 Hz-removed validation
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:1]
src = (HERE / "evaluate_checks.py").read_text().split("res = {")[0]       # reuse helpers only
exec(compile(src, str(HERE / "evaluate_checks.py"), "exec"))

PHASE1_RUNS = Path(os.environ.get("PHASE1_RUNS", HERE.parent / "runs"))
BAND = [(3, 8)]
groups = {
    "pilot": [(m, s, PHASE1_RUNS / m / f"seed{s}", None) for m in ["mamba", "lti", "cnn"] for s in [0, 1, 2]],
    "A": [(m, s, HERE / f"runs/A_{m}_s{s}", None) for m in ["mamba", "lti"] for s in [3, 4, 5]],
    "B": [(m, s, HERE / f"runs/B_{m}_s{s}", (0.3, 3)) for m in ["mamba", "lti"] for s in [0, 1, 2]],
}
res, lines = {}, ["EXPLORATORY: joint 3-8 Hz band-stop (not pre-registered). Values in %, validation only."]
for g, items in groups.items():
    lines.append(f"== {g} ==")
    for m, s, run_dir, base in items:
        r = drops(run_dir, s, base, BAND)
        res.setdefault(g, {})[f"{m}_s{s}"] = r
        ref, d = r["reference"], r["3-8"]
        lines.append(f"  {m:5} s{s}: macro-F1 {ref['macro_f1']:6.2f} -> {ref['macro_f1'] - d['macro_f1']:6.2f} (drop {d['macro_f1']:6.2f})"
                     f" | dyn-F1 {ref['dyn_f1']:6.2f} -> {ref['dyn_f1'] - d['dyn_f1']:6.2f} (drop {d['dyn_f1']:6.2f})"
                     f" | acc drop {d['acc']:6.2f}")
    seeds = sorted({s for _, s, _, _ in items})
    for k in ["dyn_f1", "macro_f1"]:
        q = [res[g][f"mamba_s{s}"]["3-8"][k] - res[g][f"lti_s{s}"]["3-8"][k] for s in seeds]
        lines.append(f"  selective - LTI drop [{k}] = {[round(v, 2) for v in q]}")
text = "\n".join(lines)
print(text)
(HERE / "exploratory_joint_3_8.txt").write_text(text + "\n")
(HERE / "exploratory_joint_3_8.json").write_text(json.dumps(res, indent=1))
