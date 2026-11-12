"""E2 retraining control (final_protocol.md section 6): information loss vs. distribution shift, seeds 6-8.

For each band b and model m:
  eval-only drop  = F1(E1 model, original val) - F1(E1 model, val with b removed)      (from phase2/results/h1.json)
  retrained drop  = F1(E1 model, original val) - F1(E2_b model, val with b removed)    (E2 summary.json, best checkpoint)
Recovery to within delta in every seed suggests the remaining bands can substitute (the eval-only drop was mostly
distribution shift); a persistent retrained drop suggests information that is hard to replace with this model/budget.
Macro-F1, percentage points.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "phase2/runs"
SEEDS, MODELS, DELTA = [6, 7, 8], ["mamba", "lti", "cnn"], 1.0
BANDS = ["0.3-1", "1-2", "2-3", "3-7", "7-11", "11-15", "15-19", "0.3-3"]
h1 = json.loads((REPO / "phase2/results/h1.json").read_text())["drops"]
f1 = lambda run: json.loads((RUNS / run / "summary.json").read_text())["runs"][0]["macro_f1"] * 100

out, lines = {}, ["E2 (validation, seeds 6-8): macro-F1 drop vs. full-input E1, pp. eval-only -> retrained; verdict"]
for b in BANDS:
    row = []
    for m in MODELS:
        full = [f1(f"E1_{m}_s{s}") for s in SEEDS]
        retr = [full[i] - f1(f"E2_{b}_{m}_s{s}") for i, s in enumerate(SEEDS)]
        ev = [h1[m][str(s)][b]["macro_f1"] if b in h1[m][str(s)] else None for s in SEEDS]
        tag = "recovers" if all(x <= DELTA for x in retr) else "persists" if all(x > DELTA for x in retr) else "mixed"
        out.setdefault(b, {})[m] = {"eval_only": ev, "retrained": retr, "reading": tag}
        evs = "/".join("  -  " if x is None else f"{x:5.2f}" for x in ev)
        row.append(f"{m} {evs} -> {'/'.join(f'{x:5.2f}' for x in retr)} {tag}")
    lines.append(f"{b:>6}  " + " | ".join(row))
text = "\n".join(lines)
print(text)
(REPO / "phase2/results/e2.txt").write_text(text + "\n")
(REPO / "phase2/results/e2.json").write_text(json.dumps(out, indent=1))
