"""EXPLORATORY (not in PREREGISTRATION.md): sweep the edges of the "mid" band used for H2.

Run on pilot checkpoints only (seeds 0-5), validation only, to choose a band before the seed 6-8 confirmation.
- B   : 0.3-3 Hz-removed retraining (selective, LTI; seeds 0-2), reference = 0.3-3 Hz-removed validation (revised H2)
- full: Phase 1 seeds 0-2 + Check A seeds 3-5 (selective, LTI), reference = unfiltered validation (original H2)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:1]
src = (HERE / "evaluate_checks.py").read_text().split("res = {")[0]       # reuse helpers only
exec(compile(src, str(HERE / "evaluate_checks.py"), "exec"))

PHASE1_RUNS = Path(os.environ.get("PHASE1_RUNS", HERE.parent / "runs"))
LOS, HIS = [3, 3.5, 4, 5], [6, 7, 8, 9, 10, 12, 15, 20]
BANDS = [(lo, hi) for lo in LOS for hi in HIS if hi - lo >= 2] + [(5, 8)]
groups = {
    "B": [(m, s, HERE / f"runs/B_{m}_s{s}", (0.3, 3)) for m in ["mamba", "lti"] for s in [0, 1, 2]],
    "full": [(m, s, PHASE1_RUNS / m / f"seed{s}", None) for m in ["mamba", "lti"] for s in [0, 1, 2]]
          + [(m, s, HERE / f"runs/A_{m}_s{s}", None) for m in ["mamba", "lti"] for s in [3, 4, 5]],
}
res = {}
for g, items in groups.items():
    for m, s, run_dir, base in items:
        res.setdefault(g, {})[f"{m}_s{s}"] = drops(run_dir, s, base, BANDS)
        (HERE / "exploratory_band_sweep.json").write_text(json.dumps(res, indent=1))   # save after every checkpoint
        print(time.strftime("%H:%M:%S"), g, m, s, "done", flush=True)
