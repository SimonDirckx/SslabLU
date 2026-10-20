#!/usr/bin/env python3
# =============================================================================
# channel_ridgectr_sweep.py
#
# Runs channel_barotropic_timestep.py over a DoF / slab-layout / solver grid,
# with the ridge crest at x/L = 0.5 (SSLABLU_RIDGE_MIDPANEL=0, the _ridgectr
# runs). Loops, outermost first:
#   npan_y   (default 8, 16, 32)
#   p        (default 8, 12, 16)
#   N        = 8, 16, ..., npan_y (powers of 2, as red-black needs), with
#              npan_x = 4 npan_y / N. N npan_x = 4 npan_y keeps the leaf grid
#              (2 npan_y x npan_y leaves, aspect 1:2) fixed for a given npan_y,
#              so every N at one (npan_y, p) is the SAME discretization, split
#              into slabs differently: dense rb agrees across N to round-off.
#   variant  one solver / rank setting, from the tokens below
#
# Variant tokens (SSLABLU_SWEEP_VARIANTS, comma-separated):
#   rb         dense red-black on dense S-maps (RK = 0)
#   <f>        e.g. 0.5: rbhbs with RK = RB_RK = f p (both compressions at once)
#   smap:<r>   dense red-black on S-maps compressed at absolute rank RK = r
#              (densified before the solve): the S-map compression error alone
#   solver:<r> rbhbs at absolute solver rank RB_RK = r on S-maps compressed at
#              RK = SSLABLU_SWEEP_RK_HI p (default 1.5 p): the solver's own
#              pivot / fill-in compression error, over a near-exact S-map floor
# Ranks must lie in (0, 2p), below the HBS leaf size 2p; an absolute rank that
# doesn't fit a given p is skipped for that p, with a note.
#
# Presets (SSLABLU_SWEEP_TEST) set the defaults; every SSLABLU_SWEEP_* variable
# overrides its preset value:
#   solver   (default) the rb vs rbhbs sweep: variants rb,0.5,1,1.5 over
#            npan_y = 8,16,32, p = 8,12,16. 72 runs.
#   rankiso  separates the two rank effects: variants rb, smap:4,6,8,12 and
#            solver:4,6,8,12 over npan_y = 32 (the only npan_y with N = 32),
#            p = 8,12,16, and SSLABLU_COMPARE_FORMS=0 (no non-conservative
#            twin run: halves the cost; the npz holds the conservative form
#            only either way). Up to 81 runs.
#
# Output: every run directory, plus the per-run logs in <outdir>/logs/, goes
# into one subdirectory of the repo root,
#   central_ridge_<test>_<gpu|cpu>_runs/      (SSLABLU_SWEEP_OUTDIR overrides)
# Runs are serial, in loop order (cheapest npan_y first); each run's
# stdout/stderr is echoed and saved to
#   <outdir>/logs/p<p>_N<N>_pan<npan_x>x<npan_y>_<variant>.log
# A run whose log names an output directory that already holds
# channel_timestep_ssh.npz is skipped, so an interrupted sweep resumes where
# it stopped (delete the log to rerun). Moving an older sweep's run
# directories and logs into <outdir> (logs into logs/) lets it resume there.
#
# GPU by default (SSLABLU_GPU=1: hpsalt leaf ops, and the torch HBS red-black
# for rbhbs); the sweep checks first that a CUDA kernel actually runs, i.e.
# that the interpreter is the sslabluenv_gpu one. SSLABLU_GPU=0 in the
# environment runs on CPU instead.
#
# Usage (from anywhere):
#   python test/validation/channel_ridgectr_sweep.py                 # solver preset
#   SSLABLU_SWEEP_TEST=rankiso python test/validation/channel_ridgectr_sweep.py
#   SSLABLU_SWEEP_TEST     solver | rankiso                (default solver)
#   SSLABLU_SWEEP_NPAN_YS  npan_y values
#   SSLABLU_SWEEP_PS       orders p
#   SSLABLU_SWEEP_VARIANTS variant tokens (see above)
#   SSLABLU_SWEEP_RK_HI    S-map rank of solver:<r>, as a multiple of p (default 1.5)
#   SSLABLU_SWEEP_OUTDIR   output subdirectory (relative to the repo root)
# Every other SSLABLU_* variable in the environment (DT_H, NSTEPS, SSH_NS,
# RNG_SEED, ...) passes through unchanged; N, P, NPAN_X, NPAN_Y, SOLVER, RK,
# RB_RK, GPU and RIDGE_MIDPANEL are set here.
# =============================================================================

import os
import sys
import time
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "test" / "validation" / "channel_barotropic_timestep.py"

PRESETS = {
    "solver":  dict(npan_ys="8,16,32", ps="8,12,16", variants="rb,0.5,1,1.5", env={}),
    "rankiso": dict(npan_ys="32", ps="8,12,16",
                    variants="rb,smap:4,smap:6,smap:8,smap:12,solver:4,solver:6,solver:8,solver:12",
                    env={"SSLABLU_COMPARE_FORMS": "0"}),
}
TEST = os.environ.get("SSLABLU_SWEEP_TEST", "solver")
if TEST not in PRESETS:
    sys.exit("SSLABLU_SWEEP_TEST must be one of %s, got %r" % (", ".join(PRESETS), TEST))
PRE = PRESETS[TEST]
GPU = os.environ.get("SSLABLU_GPU", "1") != "0"

NPYs = [int(v) for v in os.environ.get("SSLABLU_SWEEP_NPAN_YS", PRE["npan_ys"]).split(",") if v]
Ps = [int(v) for v in os.environ.get("SSLABLU_SWEEP_PS", PRE["ps"]).split(",") if v]
VARIANTS = [v.strip() for v in os.environ.get("SSLABLU_SWEEP_VARIANTS", PRE["variants"]).split(",") if v.strip()]
RK_HI = float(os.environ.get("SSLABLU_SWEEP_RK_HI", "1.5"))
OUT = REPO / os.environ.get("SSLABLU_SWEEP_OUTDIR",
                            "central_ridge_%s_%s_runs" % (TEST, "gpu" if GPU else "cpu"))
LOGS = OUT / "logs"

if GPU:
    import torch
    try:
        float((torch.zeros(1, device="cuda") + 1).sum())
    except Exception as e:
        sys.exit("SSLABLU_GPU=1 but no CUDA kernel runs with %s (%s: %s). Use the "
                 "sslabluenv_gpu env, or set SSLABLU_GPU=0 for a CPU sweep."
                 % (sys.executable, type(e).__name__, str(e).splitlines()[0]))


def variant(tok, p):
    """(solver, RK, RB_RK, log tag) for one token at order p, or a reason to skip."""
    def fits(r):
        return r == int(r) and 0 < r < 2 * p
    if tok == "rb":
        return ("rb", 0, 0, "rb")
    kind, _, val = tok.partition(":")
    if not val:                                   # <f>: RK = RB_RK = f p
        r = float(kind) * p
        if not fits(r):
            sys.exit("variant %s: RK = %g at p = %d must be an integer in (0, 2p)" % (tok, r, p))
        return ("rbhbs", int(r), int(r), "rbhbs_rk%d" % r)
    r = int(val)
    if not fits(r):
        return "rank %d not in (0, 2p) = (0, %d)" % (r, 2 * p)
    if kind == "smap":
        return ("rb", r, 0, "rb_rk%d" % r)
    if kind == "solver":
        hi = RK_HI * p
        if not fits(hi):
            sys.exit("SSLABLU_SWEEP_RK_HI = %g: RK = %g at p = %d must be an integer in (0, 2p)" % (RK_HI, hi, p))
        return ("rbhbs", int(hi), r, "rbhbs_rk%d_rbrk%d" % (hi, r))
    sys.exit("unknown variant token %r (rb, <f>, smap:<r> or solver:<r>)" % tok)


# (npan_y, p, N, npan_x, solver, RK, RB_RK, log tag) in loop order
configs, skipped = [], []
for npy in NPYs:
    for p in Ps:
        N = 8
        while N <= npy:
            npx = 4 * npy // N
            if npx % 2 or 4 * npy % N:
                sys.exit("npan_x = 4 npan_y / N = 4*%d/%d must be an even integer" % (npy, N))
            for tok in VARIANTS:
                v = variant(tok, p)
                if isinstance(v, str):
                    if N == 8:
                        skipped.append("p=%d %s: %s" % (p, tok, v))
                    continue
                configs.append((npy, p, N, npx) + v)
            N *= 2


def finished_dir(log):
    """Output directory named in a previous log, if it holds the ssh npz."""
    if not log.exists():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("output directory"):
            d = OUT / line.split("=", 1)[1].strip()
            return d if (d / "channel_timestep_ssh.npz").exists() else None
    return None


def label(c):
    npy, p, N, npx, solver, rk, rbrk, _ = c
    return "p=%d N=%d pan=%dx%d %s RK=%d%s" % (p, N, npx, npy, solver, rk,
                                              (" RB_RK=%d" % rbrk) if solver == "rbhbs" else "")


LOGS.mkdir(parents=True, exist_ok=True)
print("%s sweep: %d runs on %s, into %s/" % (TEST, len(configs), "GPU" if GPU else "CPU",
                                            OUT.relative_to(REPO)), flush=True)
for s in skipped:
    print("  skipped variant " + s)
results = []       # (config, status, seconds, output dir)
for i, c in enumerate(configs):
    npy, p, N, npx, solver, rk, rbrk, tag = c
    k = "%d/%d" % (i + 1, len(configs))
    log = LOGS / ("p%d_N%d_pan%dx%d_%s.log" % (p, N, npx, npy, tag))
    done = finished_dir(log)
    if done is not None:
        print("[%s] %s: skip, already in %s" % (k, label(c), done.name))
        results.append((c, "skipped", 0.0, done.name))
        continue

    env = dict(os.environ, **{key: os.environ.get(key, val) for key, val in PRE["env"].items()})
    env.update(SSLABLU_N=str(N), SSLABLU_P=str(p), SSLABLU_NPAN_X=str(npx),
               SSLABLU_NPAN_Y=str(npy), SSLABLU_SOLVER=solver, SSLABLU_RK=str(rk),
               SSLABLU_GPU="1" if GPU else "0", SSLABLU_RIDGE_MIDPANEL="0",
               PYTHONUNBUFFERED="1")
    if solver == "rbhbs":
        env["SSLABLU_RB_RK"] = str(rbrk)
    else:
        env.pop("SSLABLU_RB_RK", None)
    print("[%s] %s: running (log: %s)" % (k, label(c), log.relative_to(REPO)), flush=True)
    tic = time.time()
    outdir = ""
    with open(log, "w") as fl:
        proc = subprocess.Popen([sys.executable, str(SCRIPT)], cwd=OUT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            fl.write(line)
            if line.startswith("output directory"):
                outdir = line.split("=", 1)[1].strip()
        rc = proc.wait()
    toc = time.time() - tic
    results.append((c, "ok" if rc == 0 else "FAILED (exit %d)" % rc, toc, outdir))
    print("[%s] %s: %s in %.1f s" % (k, label(c), results[-1][1], toc), flush=True)

print("")
print("=============SUMMARY  (%s sweep, ridge crest at x/L = 0.5, %s)============="
      % (TEST, "GPU" if GPU else "CPU"))
print("  npan_y   p   N  npan_x  solver   RK  RB_RK   status            time [s]   output directory")
for (npy, p, N, npx, solver, rk, rbrk, _), status, toc, outdir in results:
    print("  %5d  %3d  %2d  %6d  %-6s  %3d  %5s   %-16s %9.1f   %s"
          % (npy, p, N, npx, solver, rk, str(rbrk) if solver == "rbhbs" else "-", status, toc, outdir))
print("=================================================")
print("output in %s/" % OUT.relative_to(REPO))
if any(r[1].startswith("FAILED") for r in results):
    sys.exit(1)
