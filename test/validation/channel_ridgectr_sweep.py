#!/usr/bin/env python3
# =============================================================================
# channel_ridgectr_sweep.py
#
# Runs channel_barotropic_timestep.py over a DoF / slab-layout / solver grid,
# with the ridge crest at x/L = 0.5 (SSLABLU_RIDGE_MIDPANEL=0, the _ridgectr
# runs). Loops, outermost first:
#   npan_y   = 8, 16, 32
#   p        = 8, 12, 16
#   N        = 8, 16, ..., npan_y (powers of 2, as red-black needs), with
#              npan_x = 4 npan_y / N. N npan_x = 4 npan_y keeps the leaf grid
#              (2 npan_y x npan_y leaves, aspect 1:2) fixed for a given npan_y,
#              so every N at one (npan_y, p) is the SAME discretization, split
#              into slabs differently: dense rb agrees across N to round-off.
#   variant  rb      dense red-black, dense S-maps (RK = 0)
#            rbhbs   HBS red-black at RB_RK = p/2, p, 3p/2, with the S-maps
#                    compressed at the same rank (RK = RB_RK; rbhbs needs RK > 0,
#                    and both stay below the HBS leaf size 2p)
# Runs are serial, in loop order (cheapest npan_y first). Each run's
# stdout/stderr is echoed and saved to
#   channel_ridgectr_sweep_logs/p<p>_N<N>_pan<npan_x>x<npan_y>_<variant>.log
# (variant rb, or rbhbs_rk<RK>). A run whose log names an output directory
# that already holds channel_timestep_ssh.npz is skipped, so an interrupted
# sweep resumes where it stopped (delete the log to rerun).
#
# GPU by default (SSLABLU_GPU=1: hpsalt leaf ops, and the torch HBS red-black
# for rbhbs); the sweep checks first that a CUDA kernel actually runs, i.e.
# that the interpreter is the sslabluenv_gpu one. SSLABLU_GPU=0 in the
# environment runs on CPU instead. Output directories carry no GPU token.
#
# Usage (from anywhere; runs execute in the repo root, where the
# run_sslablu_channel_* directories land):
#   python test/validation/channel_ridgectr_sweep.py
#   SSLABLU_SWEEP_NPAN_YS  npan_y values               (default 8,16,32)
#   SSLABLU_SWEEP_PS       orders p                    (default 8,12,16)
#   SSLABLU_SWEEP_VARIANTS rb and/or rank multiples of p for rbhbs
#                          (default rb,0.5,1,1.5)
# Every other SSLABLU_* variable in the environment (DT_H, NSTEPS, SSH_NS,
# RNG_SEED, ...) passes through unchanged; N, P, NPAN_X, NPAN_Y, SOLVER, RK,
# RB_RK and RIDGE_MIDPANEL are set here.
# =============================================================================

import os
import sys
import time
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "test" / "validation" / "channel_barotropic_timestep.py"
LOGS = REPO / "channel_ridgectr_sweep_logs"

NPYs = [int(v) for v in os.environ.get("SSLABLU_SWEEP_NPAN_YS", "8,16,32").split(",") if v]
Ps = [int(v) for v in os.environ.get("SSLABLU_SWEEP_PS", "8,12,16").split(",") if v]
VARIANTS = [v.strip() for v in os.environ.get("SSLABLU_SWEEP_VARIANTS", "rb,0.5,1,1.5").split(",") if v.strip()]
GPU = os.environ.get("SSLABLU_GPU", "1") != "0"

if GPU:
    import torch
    try:
        float((torch.zeros(1, device="cuda") + 1).sum())
    except Exception as e:
        sys.exit("SSLABLU_GPU=1 but no CUDA kernel runs with %s (%s: %s). Use the "
                 "sslabluenv_gpu env, or set SSLABLU_GPU=0 for a CPU sweep."
                 % (sys.executable, type(e).__name__, str(e).splitlines()[0]))

# (npan_y, p, N, npan_x, solver, RK, RB_RK) in loop order
configs = []
for npy in NPYs:
    for p in Ps:
        N = 8
        while N <= npy:
            npx = 4 * npy // N
            if npx % 2 or 4 * npy % N:
                sys.exit("npan_x = 4 npan_y / N = 4*%d/%d must be an even integer" % (npy, N))
            for v in VARIANTS:
                if v == "rb":
                    configs.append((npy, p, N, npx, "rb", 0, 0))
                else:
                    rk = float(v) * p
                    if rk != int(rk) or not 0 < rk < 2 * p:
                        sys.exit("variant %s: RK = %s p = %g must be an integer in (0, 2p)" % (v, v, rk))
                    configs.append((npy, p, N, npx, "rbhbs", int(rk), int(rk)))
            N *= 2


def finished_dir(log):
    """Output directory named in a previous log, if it holds the ssh npz."""
    if not log.exists():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("output directory"):
            d = REPO / line.split("=", 1)[1].strip()
            return d if (d / "channel_timestep_ssh.npz").exists() else None
    return None


def label(c):
    npy, p, N, npx, solver, rk, _ = c
    return "p=%d N=%d pan=%dx%d %s" % (p, N, npx, npy, solver if rk == 0 else "%s rk=%d" % (solver, rk))


LOGS.mkdir(exist_ok=True)
print("%d runs on %s" % (len(configs), "GPU" if GPU else "CPU"), flush=True)
results = []       # (config, status, seconds, output dir)
for i, c in enumerate(configs):
    npy, p, N, npx, solver, rk, rbrk = c
    tag = "%d/%d" % (i + 1, len(configs))
    log = LOGS / ("p%d_N%d_pan%dx%d_%s.log" % (p, N, npx, npy, solver if rk == 0 else "rbhbs_rk%d" % rk))
    done = finished_dir(log)
    if done is not None:
        print("[%s] %s: skip, already in %s" % (tag, label(c), done.name))
        results.append((c, "skipped", 0.0, done.name))
        continue

    env = dict(os.environ, SSLABLU_N=str(N), SSLABLU_P=str(p), SSLABLU_NPAN_X=str(npx),
               SSLABLU_NPAN_Y=str(npy), SSLABLU_SOLVER=solver, SSLABLU_RK=str(rk),
               SSLABLU_GPU="1" if GPU else "0", SSLABLU_RIDGE_MIDPANEL="0",
               PYTHONUNBUFFERED="1")
    if solver == "rbhbs":
        env["SSLABLU_RB_RK"] = str(rbrk)
    else:
        env.pop("SSLABLU_RB_RK", None)
    print("[%s] %s: running (log: %s)" % (tag, label(c), log.relative_to(REPO)), flush=True)
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
    results.append((c, "ok" if rc == 0 else "FAILED (exit %d)" % rc, toc, outdir))
    print("[%s] %s: %s in %.1f s" % (tag, label(c), results[-1][1], toc), flush=True)

print("")
print("=============SUMMARY  (ridge crest at x/L = 0.5, %s)=============" % ("GPU" if GPU else "CPU"))
print("  npan_y   p   N  npan_x  solver  RK=RB_RK   status            time [s]   output directory")
for (npy, p, N, npx, solver, rk, rbrk), status, toc, outdir in results:
    print("  %5d  %3d  %2d  %6d  %-6s  %8s   %-16s %9.1f   %s"
          % (npy, p, N, npx, solver, "-" if rk == 0 else str(rk), status, toc, outdir))
print("=================================================")
if any(r[1].startswith("FAILED") for r in results):
    sys.exit(1)
