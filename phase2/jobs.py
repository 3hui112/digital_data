"""Write phase2/jobs.txt in priority order (final_protocol.md section 6): E1, E2 low-off, then the other E2 bands."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [6, 7, 8]
MODELS = [("mamba", ""), ("lti", "--lti"), ("cnn", "")]
E2_BANDS = ["0.3-1", "1-2", "2-3", "3-7", "7-11", "11-15", "15-19"]


def job(exp, bands, name, flag, seed):
    model = "cnn" if name == "cnn" else "mamba"
    run = f"{exp}_{bands}_{name}_s{seed}" if exp == "E2" else f"{exp}_{name}_s{seed}"
    return f"{run}\t--model {model} {flag} --seed {seed} --bands {bands} --out phase2/runs/{run}".replace("  ", " ")


lines = [job("E1", "none", n, f, s) for s in SEEDS for n, f in MODELS]
lines += [job("E2", "0.3-3", n, f, s) for s in SEEDS for n, f in MODELS]
lines += [job("E2", b, n, f, s) for b in E2_BANDS for s in SEEDS for n, f in MODELS]
(HERE / "jobs.txt").write_text("\n".join(lines) + "\n")
print(len(lines), "jobs")
