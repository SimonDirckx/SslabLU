#!/usr/bin/env python3
# =============================================================================
# channel_ridgectr_sweep.py
#
# Runs channel_barotropic_timestep.py over the (N, p) grid used by
# channel_sslablu_convergence.py, with the ridge crest at x/L = 0.5
# (SSLABLU_RIDGE_MIDPANEL=0, the _ridgectr runs):
#   N = 8, 16, 32  (npan_y = N, npan_x fixed: h-refinement in x is the slab count)
#   p = 8, 12, 16
# Runs are serial, cheapest first. Each run's stdout/stderr is echoed and saved
# to channel_ridgectr_sweep_logs/p<p>_N<N>.log. A configuration whose log names
# an output directory that already holds channel_timestep_ssh.npz is skipped,
# so an interrupted sweep resumes where it stopped (delete the log to rerun).
#
# Usage (from anywhere; runs execute in the repo root, where the
# run_sslablu_channel_* directories land), with the sslabluenv interpreter:
#   python test/validation/channel_ridgectr_sweep.py
#   SSLABLU_SWEEP_NS     slab counts N   (default 8,16,32)
#   SSLABLU_SWEEP_PS     orders p        (default 8,12,16)
#   SSLABLU_NPAN_X       passed through  (default 4 in the timestep script)
# Every other SSLABLU_* variable in the environment (SOLVER, DT_H, NSTEPS,
# SSH_NS, ...) passes through unchanged; N, P, NPAN_Y and RIDGE_MIDPANEL are
# set here.
# =============================================================================

import os
import sys
import time
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "test" / "validation" / "channel_barotropic_timestep.py"
LOGS = REPO / "channel_ridgectr_sweep_logs"

Ns = [int(v) for v in os.environ.get("SSLABLU_SWEEP_NS", "8,16,32").split(",") if v]
Ps = [int(v) for v in os.environ.get("SSLABLU_SWEEP_PS", "8,12,16").split(",") if v]
configs = sorted(((N, p) for N in Ns for p in Ps), key=lambda c: (c[0] * c[1] ** 2, c))


def finished_dir(log):
    """Output directory named in a previous log, if it holds the ssh npz."""
    if not log.exists():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("output directory"):
            d = REPO / line.split("=", 1)[1].strip()
            return d if (d / "channel_timestep_ssh.npz").exists() else None
    return None


LOGS.mkdir(exist_ok=True)
results = []       # (N, p, status, seconds, output dir)
for i, (N, p) in enumerate(configs):
    log = LOGS / ("p%d_N%d.log" % (p, N))
    done = finished_dir(log)
    if done is not None:
        print("[%d/%d] p=%d N=%d: skip, already in %s" % (i + 1, len(configs), p, N, done.name))
        results.append((N, p, "skipped", 0.0, done.name))
        continue

    env = dict(os.environ, SSLABLU_N=str(N), SSLABLU_P=str(p), SSLABLU_NPAN_Y=str(N),
               SSLABLU_RIDGE_MIDPANEL="0", PYTHONUNBUFFERED="1")
    print("[%d/%d] p=%d N=%d npan_y=%d: running (log: %s)"
          % (i + 1, len(configs), p, N, N, log.relative_to(REPO)), flush=True)
    tic = time.time()
    outdir = ""
    with open(log, "w") as fl:
        proc = subprocess.Popen([sys.executable, str(SCRIPT)], cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            fl.write(line)
            if line.startswith("output directory"):
                outdir = line.split("=", 1)[1].strip()
        rc = proc.wait()
    toc = time.time() - tic
    results.append((N, p, "ok" if rc == 0 else "FAILED (exit %d)" % rc, toc, outdir))
    print("[%d/%d] p=%d N=%d: %s in %.1f s" % (i + 1, len(configs), p, N, results[-1][2], toc), flush=True)

print("")
print("=============SUMMARY  (ridge crest at x/L = 0.5)=============")
print("   N   p   status            time [s]   output directory")
for N, p, status, toc, outdir in results:
    print("  %2d  %2d   %-16s %9.1f   %s" % (N, p, status, toc, outdir))
print("=================================================")
if any(r[2].startswith("FAILED") for r in results):
    sys.exit(1)
