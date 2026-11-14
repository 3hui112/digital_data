"""E4 (optional) on validation, seeds 6-8: does the E1 / H1 band pattern of selective Mamba depend on the
Delta initialization range? Same band-stop evaluation as evaluate_h1.py, for dt in [1e-4, 1e-2] and [1e-2, 1]
next to the default [1e-3, 1e-1] (the E1 selective checkpoints). Per-seed H1 checks use the same rule.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.argv = sys.argv[:1]
src = (REPO / "h2_check/evaluate_checks.py").read_text().split("res = {")[0]       # reuse helpers only
exec(compile(src, str(REPO / "h2_check/evaluate_checks.py"), "exec"))

SEEDS = [6, 7, 8]
SETTINGS = {"dt[1e-4,1e-2]": "E4_dt1e-4-1e-2_mamba_s{}", "dt[1e-3,1e-1] (E1)": "E1_mamba_s{}",
            "dt[1e-2,1]": "E4_dt1e-2-1_mamba_s{}"}
LOW, MID, HIGH = [(0.3, 1), (1, 2), (2, 3)], [(3, 7), (7, 11)], [(11, 15), (15, 19)]
BANDS = LOW + MID + HIGH + [(3, 8)]
key = lambda b: f"{b[0]}-{b[1]}"

res = {n: {s: drops(REPO / "phase2/runs" / p.format(s), s, None, BANDS) for s in SEEDS} for n, p in SETTINGS.items()}


def seed_check(n, s):
    d = {key(b): res[n][s][key(b)]["macro_f1"] for b in BANDS}
    low = max(d[key(b)] for b in LOW)
    support = all(low - d[key(b)] > DELTA for b in MID + HIGH) and all(d[key(b)] <= DELTA for b in HIGH)
    against = any(d[key(b)] - low > DELTA for b in MID + HIGH) or any(d[key(b)] > DELTA for b in HIGH)
    return support, against


cell = lambda n, b, m: "/".join(f"{res[n][s][key(b)][m]:5.2f}" for s in SEEDS)
checks = {n: [seed_check(n, s) for s in SEEDS] for n in SETTINGS}
lines = ["E4 (validation, seeds 6-8, selective Mamba): macro-F1 drop in pp by Delta init range",
         "band     " + "  ".join(f"{n:>20}" for n in SETTINGS)]
for b in BANDS:
    lines.append(f"{key(b):>8} " + "  ".join(
        f"{cell(n, b, 'macro_f1'):>20}" for n in SETTINGS))
lines.append("reference macro-F1: " + "  ".join(
    f"{n} " + "/".join(f"{res[n][s]['reference']['macro_f1']:.2f}" for s in SEEDS) for n in SETTINGS))
lines.append("dynamic-class F1 drop (pp):")
for b in BANDS:
    lines.append(f"{key(b):>8} " + "  ".join(
        f"{cell(n, b, 'dyn_f1'):>20}" for n in SETTINGS))
lines.append("per-seed H1 rule (support, against): " + "  ".join(f"{n} {checks[n]}" for n in SETTINGS))
text = "\n".join(lines)
print(text)
(REPO / "phase2/results/e4.txt").write_text(text + "\n")
(REPO / "phase2/results/e4.json").write_text(json.dumps(
    {"drops": {n: {str(s): v for s, v in r.items()} for n, r in res.items()}, "checks": checks}, indent=1))
