"""Run phase2/jobs.txt on GPUs as they become free (final_protocol.md).

A GPU is taken when nobody else uses it (memory.used < FREE_MIB) and then holds up to SLOTS of our runs.
A run is finished when <out>/summary.json exists; a failed run is retried once. Re-running the scheduler resumes.
usage: python phase2/scheduler.py            env: DATA, VENV, FREE_MIB (1500), SLOTS (6), POLL (60)
"""
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "phase2/logs"
PY = Path(os.environ.get("VENV", Path.home() / ".venvs/mamba-dd")) / "bin/python"
FREE_MIB, SLOTS, POLL = (int(os.environ.get(k, d)) for k, d in [("FREE_MIB", 1500), ("SLOTS", 6), ("POLL", 60)])


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG / "scheduler.log", "a") as f:
        f.write(line + "\n")


def gpu_memory():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return {int(i): int(m) for i, m in (l.split(", ") for l in out.strip().splitlines())}


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    jobs = [l.split("\t") for l in (REPO / "phase2/jobs.txt").read_text().splitlines() if l.strip()]
    done = lambda args: (REPO / args.split("--out ")[1].strip() / "summary.json").exists()
    pending = [(r, a) for r, a in jobs if not done(a)]
    running, tries = {}, {}                                  # run -> (Popen, gpu); run -> attempts
    log(f"start: {len(jobs)} jobs, {len(pending)} pending")
    while pending or running:
        for run, (proc, gpu) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[run]
            args = dict(jobs)[run]
            if done(args):
                log(f"done {run} (gpu {gpu})")
            elif tries[run] < 2:
                log(f"FAILED {run} (gpu {gpu}, exit {proc.returncode}); will retry")
                pending.insert(0, (run, args))
            else:
                log(f"FAILED {run} twice (exit {proc.returncode}); giving up")
        mem = gpu_memory()
        ours = {}
        for _, gpu in running.values():
            ours[gpu] = ours.get(gpu, 0) + 1
        usable = [g for g in sorted(mem) if g in ours or mem[g] < FREE_MIB]
        for gpu in usable:
            while pending and ours.get(gpu, 0) < SLOTS:
                run, args = pending.pop(0)
                tries[run] = tries.get(run, 0) + 1
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                           PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                with open(LOG / f"{run}.log", "a") as f:
                    proc = subprocess.Popen([str(PY), "phase2/train_condition.py", *args.split()], cwd=REPO, env=env,
                                            stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                running[run] = (proc, gpu)
                ours[gpu] = ours.get(gpu, 0) + 1
                log(f"launch {run} on gpu {gpu} (attempt {tries[run]})")
        time.sleep(POLL)
    log("all jobs finished")


if __name__ == "__main__":
    main()
