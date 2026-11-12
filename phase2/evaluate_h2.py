"""H2 verdict on validation (final_protocol.md section 5), seeds 6-8.

Each E2 low-off checkpoint (trained with 0.3-3 Hz removed) is evaluated on the low-off validation windows and on the
same windows with 3-7 Hz also removed (cascade, 0.3-3 Hz first). c = macro-F1 drop; q_cond = c(selective) - c(LTI).
Secondary: joint 3-8 Hz, dynamic-class F1, CNN.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.argv = sys.argv[:1]
src = (REPO / "h2_check/evaluate_checks.py").read_text().split("res = {")[0]       # reuse helpers only
exec(compile(src, str(REPO / "h2_check/evaluate_checks.py"), "exec"))

SEEDS, MODELS, BANDS = [6, 7, 8], ["mamba", "lti", "cnn"], [(3, 7), (3, 8)]
res = {m: {s: drops(REPO / f"phase2/runs/E2_0.3-3_{m}_s{s}", s, (0.3, 3), BANDS) for s in SEEDS} for m in MODELS}

out, lines = {}, ["H2 (E2 low-off checkpoints, validation, seeds 6-8); values in pp"]
for b in ["3-7", "3-8"]:
    for met in ["macro_f1", "dyn_f1"]:
        c = {m: [res[m][s][b][met] for s in SEEDS] for m in MODELS}
        q = [cs - cl for cs, cl in zip(c["mamba"], c["lti"])]
        if all(x > DELTA for x in c["mamba"]) and all(x > DELTA for x in q):
            v = "SUPPORTED"
        elif all(x <= DELTA for x in c["mamba"]) or all(x <= DELTA for x in q):
            v = "FALSIFIED"
        else:
            v = "INCONCLUSIVE"
        primary = b == "3-7" and met == "macro_f1"
        out[f"{b}/{met}"] = {"c": c, "q_cond": q, "verdict": v, "primary": primary}
        lines.append(f"{b:>4} {met:8} c sel {' / '.join(f'{x:5.2f}' for x in c['mamba'])} | lti "
                     f"{' / '.join(f'{x:5.2f}' for x in c['lti'])} | cnn {' / '.join(f'{x:5.2f}' for x in c['cnn'])}"
                     f" | q {' / '.join(f'{x:5.2f}' for x in q)} -> {v}{'  (PRIMARY)' if primary else ''}")
lines.append("low-off reference macro-F1: " + "  ".join(
    f"{m} " + "/".join(f"{res[m][s]['reference']['macro_f1']:.2f}" for s in SEEDS) for m in MODELS))
text = "\n".join(lines)
print(text)
(REPO / "phase2/results").mkdir(exist_ok=True)
(REPO / "phase2/results/h2.txt").write_text(text + "\n")
(REPO / "phase2/results/h2.json").write_text(json.dumps(
    {"drops": {m: {str(s): v for s, v in r.items()} for m, r in res.items()}, "summary": out}, indent=1))
