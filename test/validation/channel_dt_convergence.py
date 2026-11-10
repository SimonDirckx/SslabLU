#!/usr/bin/env python3
# =============================================================================
# channel_dt_convergence.py
#
# How much of the SslabLU channel error is temporal? Runs
# channel_barotropic_timestep.py (dense rb, conservative form only, ridge
# centered) to the production final time over a ladder of time steps, on one or
# more spatial grids, then compares the time error with the spatial error.
#
#   time error     backward Euler is first order, so with the ladder dt, dt/2,
#                  ... the Richardson estimate X* = 2 X(dt_min) - X(2 dt_min)
#                  stands in for the dt -> 0 solution of that grid, and
#                  e_t(dt) = ||X(dt) - X*|| / ||X*||. The observed order
#                  log2(d(dt) / d(dt/2)), d(dt) = ||X(dt) - X(dt/2)||, checks
#                  that the ladder is in the asymptotic regime (~1).
#   spatial error  at the production dt (450 s): ||X_grid - X_ref|| / ||X_ref||
#                  against a finer run at the SAME dt (default: the p = 12,
#                  256 x 128-leaf run of the fast-leaf rankiso sweep), so the
#                  time error cancels to the extent it is grid-independent;
#                  the grids' own e_t curves show how far that holds.
#
# Error norm: relative L2 on the 320^2 cell-center samples of the npz, eta
# de-meaned (as channel_sslablu_convergence.py), at t = T/2 and T.
#
# Usage (from the repo root; GPU by default, as the sweep):
#   python test/validation/channel_dt_convergence.py 2>&1 | tee dt_convergence.out
# Environment:
#   SSLABLU_DTC_GRIDS    p:npan_y:N per grid, comma-separated (npan_x = 4 npan_y / N)
#                        (default "12:32:8,8:16:8")
#   SSLABLU_DTC_DTS      time steps [s], largest first, each half the previous
#                        (default "900,450,225,112.5,56.25"; 112.5 s is the
#                        Oceananigans dt of reentrant_channel_sslablu.jl)
#   SSLABLU_DTC_T_H      final time [h] (default 100, the sweeps' 800 x 450 s)
#   SSLABLU_DTC_REF      run directory of the spatial reference at dt = 450 s
#                        (default: the fast-leaf rankiso p = 12, 256 x 128 run, under
#                        central_ridge_runs/ or the repo root; skipped if absent)
#   SSLABLU_DTC_OUTDIR   output subdirectory (relative to the repo root)
#                        (default central_ridge_dtconv_<gpu|cpu>_runs)
#   SSLABLU_GPU          1 (default) / 0
# Runs whose directory already holds channel_timestep_ssh.npz are skipped, so an
# interrupted study resumes. Outputs in <outdir>: logs/, dt_convergence.csv,
# dt_convergence.png.
# =============================================================================
import os
import sys
import glob
import time
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "test" / "validation" / "channel_barotropic_timestep.py"

GRIDS = [tuple(int(x) for x in g.split(":"))
         for g in os.environ.get("SSLABLU_DTC_GRIDS", "12:32:8,8:16:8").split(",") if g]
DTS = [float(v) for v in os.environ.get("SSLABLU_DTC_DTS", "900,450,225,112.5,56.25").split(",") if v]
T_H = float(os.environ.get("SSLABLU_DTC_T_H", "100"))
GPU = os.environ.get("SSLABLU_GPU", "1") != "0"
OUT = REPO / os.environ.get("SSLABLU_DTC_OUTDIR", "central_ridge_dtconv_%s_runs" % ("gpu" if GPU else "cpu"))
_REFNAME = "run_sslablu_channel_p12_N8_pan64x128_rb_dt450s_nsteps800_neumann_ridgectr"
_REFCAND = [REPO / d / "central_ridge_rankiso_gpu_runs_fastleaf" / _REFNAME for d in ("central_ridge_runs", ".")]
REFDIR = Path(os.environ.get("SSLABLU_DTC_REF", str(next((c for c in _REFCAND if c.exists()), _REFCAND[0]))))
DT_PROD = 450.0
LOGS = OUT / "logs"

for a, b in zip(DTS, DTS[1:]):
    if abs(a / b - 2.0) > 1e-12:
        sys.exit("SSLABLU_DTC_DTS must halve at each step, got %s" % DTS)
for dt in DTS:
    if abs(T_H * 3600.0 / dt - round(T_H * 3600.0 / dt)) > 1e-9:
        sys.exit("T = %g h is not a whole number of dt = %g s steps" % (T_H, dt))
for p, npy, N in GRIDS:
    if (4 * npy) % N or (4 * npy // N) % 2:
        sys.exit("npan_x = 4 npan_y / N = 4*%d/%d must be an even integer" % (npy, N))
if GPU:
    try:
        import torch
        torch.zeros(1, device="cuda") + 1
    except Exception as exc:
        sys.exit("SSLABLU_GPU=1 but CUDA is not usable here (%s); set SSLABLU_GPU=0" % exc)


def run_dir(p, npy, N, dt, nsteps):
    return OUT / ("run_sslablu_channel_p%d_N%d_pan%dx%d_rb_dt%gs_nsteps%d_neumann_ridgectr"
                  % (p, N, 4 * npy // N, npy, dt, nsteps))


# ---- runs -------------------------------------------------------------------
LOGS.mkdir(parents=True, exist_ok=True)
configs = [(p, npy, N, dt, int(round(T_H * 3600.0 / dt))) for p, npy, N in GRIDS for dt in DTS]
print("dt-convergence: %d runs to T = %g h on %s, into %s/" % (len(configs), T_H, "GPU" if GPU else "CPU",
                                                              OUT.relative_to(REPO)), flush=True)
status = {}
for i, (p, npy, N, dt, ns) in enumerate(configs):
    k = "%d/%d" % (i + 1, len(configs))
    lab = "p=%d %dx%d leaves N=%d dt=%g s (%d steps)" % (p, 2 * npy, npy, N, dt, ns)
    if (run_dir(p, npy, N, dt, ns) / "channel_timestep_ssh.npz").exists():
        print("[%s] %s: skip, already done" % (k, lab), flush=True)
        status[(p, npy, N, dt)] = ("skipped", 0.0)
        continue
    env = dict(os.environ, SSLABLU_N=str(N), SSLABLU_P=str(p), SSLABLU_NPAN_X=str(4 * npy // N),
               SSLABLU_NPAN_Y=str(npy), SSLABLU_SOLVER="rb", SSLABLU_RK="0",
               SSLABLU_DT_H=repr(dt / 3600.0), SSLABLU_NSTEPS=str(ns), SSLABLU_COMPARE_FORMS="0",
               SSLABLU_GPU="1" if GPU else "0", SSLABLU_RIDGE_MIDPANEL="0", PYTHONUNBUFFERED="1")
    env.pop("SSLABLU_RB_RK", None)
    log = LOGS / ("p%d_N%d_pan%dx%d_dt%gs.log" % (p, N, 4 * npy // N, npy, dt))
    print("[%s] %s: running (log: %s)" % (k, lab, log.relative_to(REPO)), flush=True)
    tic = time.time()
    with open(log, "w") as fl:
        rc = subprocess.call([sys.executable, str(SCRIPT)], cwd=OUT, env=env, stdout=fl, stderr=subprocess.STDOUT)
    status[(p, npy, N, dt)] = ("ok" if rc == 0 else "FAILED (exit %d)" % rc, time.time() - tic)
    print("[%s] %s: %s in %.0f s" % (k, lab, status[(p, npy, N, dt)][0], status[(p, npy, N, dt)][1]), flush=True)
    if not (run_dir(p, npy, N, dt, ns) / "channel_timestep_ssh.npz").exists():
        print("     no npz in %s: check the log" % run_dir(p, npy, N, dt, ns).name, flush=True)


# ---- analysis -----------------------------------------------------------------
def load(d, t):
    S = np.load(d / "channel_timestep_ssh.npz")
    return {f: (S["%s_%s_320" % (f, t)] - (S["eta_mean_" + t] if f == "eta" else 0.0)) for f in ("eta", "u", "v")}


rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
ref = None
if (REFDIR / "channel_timestep_ssh.npz").exists():
    _tf = float(np.load(REFDIR / "channel_timestep_ssh.npz")["t_final"])
    if abs(_tf - T_H * 3600.0) < 1e-6:
        ref = {t: load(REFDIR, t) for t in ("mid", "final")}
    else:
        print("spatial reference ends at %g h, not T = %g h: no spatial errors" % (_tf / 3600.0, T_H))
rows, res = [], {}
for p, npy, N in GRIDS:
    have = [dt for dt in DTS if (run_dir(p, npy, N, dt, int(round(T_H * 3600.0 / dt))) / "channel_timestep_ssh.npz").exists()]
    if len(have) < 3:
        print("p=%d %dx%d: only %d of %d time steps finished, skipping the analysis" % (p, 2 * npy, npy, len(have), len(DTS)))
        continue
    for t in ("mid", "final"):
        X = {dt: load(run_dir(p, npy, N, dt, int(round(T_H * 3600.0 / dt))), t) for dt in have}
        for f in ("eta", "u", "v"):
            star = 2.0 * X[have[-1]][f] - X[have[-2]][f]            # Richardson, first order
            d = {dt: rel(X[dt][f], X[dt / 2][f]) for dt in have if dt / 2 in X}
            for dt in have:
                et = rel(X[dt][f], star)
                order = np.log2(d[dt] / d[dt / 2]) if (dt in d and dt / 2 in d) else np.nan
                es = rel(X[dt][f], ref[t][f]) if (ref is not None and dt == DT_PROD) else np.nan
                res[(p, npy, N, t, f, dt)] = (et, order, es)
                rows.append([p, npy, N, {"mid": 0, "final": 1}[t], {"eta": 0, "u": 1, "v": 2}[f], dt, et, order, es])

print("")
print("=============SUMMARY  (dt-convergence to T = %g h; spatial reference %s)============="
      % (T_H, REFDIR.name if ref is not None else "absent"))
print("  time error e_t(dt) = ||X(dt) - X*|| / ||X*|| (X* Richardson); order from successive differences;")
print("  spatial error at dt = %g s vs the reference" % DT_PROD)
for p, npy, N in GRIDS:
    for t in ("mid", "final"):
        for f in ("eta", "u", "v"):
            if (p, npy, N, t, f, DTS[0]) not in res and (p, npy, N, t, f, DTS[1]) not in res:
                continue
            line = "  p=%2d %7s %-5s %-3s |" % (p, "%dx%d" % (2 * npy, npy), t, f)
            for dt in DTS:
                if (p, npy, N, t, f, dt) in res:
                    et, order, _ = res[(p, npy, N, t, f, dt)]
                    line += " %6g s %.1e%s |" % (dt, et, (" (%.2f)" % order) if order == order else "       ")
            es = res.get((p, npy, N, t, f, DT_PROD), (0, 0, np.nan))[2]
            et450 = res.get((p, npy, N, t, f, DT_PROD), (np.nan,))[0]
            line += " spatial %s, time/spatial at %g s: %s" % ("%.1e" % es if es == es else "-", DT_PROD,
                                                                 "%.2g" % (et450 / es) if es == es else "-")
            print(line)
print("  " + "  ".join("%s %s (%.0f s)" % ("p%d/%dx%d/dt%g" % (c[0], 2 * c[1], c[1], c[3]), s, sec)
                       for c, (s, sec) in status.items() if s != "skipped"))
print("=================================================")
if rows:
    np.savetxt(OUT / "dt_convergence.csv", np.array(rows), delimiter=",", fmt="%.6e",
               header="p,npan_y,N,time(0=mid;1=final),field(0=eta;1=u;2=v),dt_s,time_err,order,spatial_err_at_450s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for j, f in enumerate(("eta", "u", "v")):
        a = ax[j]
        for c, (p, npy, N) in enumerate(GRIDS):
            pts = [(dt, res[(p, npy, N, "final", f, dt)][0]) for dt in DTS if (p, npy, N, "final", f, dt) in res]
            if not pts:
                continue
            pts = [q for q in pts if q[1] > 0]                       # X* lies between the finest two
            a.loglog(*zip(*pts), "-o", color=COL[c % 4], label="time error, p=%d %dx%d" % (p, 2 * npy, npy))
            es = res.get((p, npy, N, "final", f, DT_PROD), (0, 0, np.nan))[2]
            if es == es:
                a.axhline(es, color=COL[c % 4], ls="--", lw=1, label="spatial error at %g s" % DT_PROD)
        dts = np.array(DTS)
        a.loglog(dts, 0.5 * pts[0][1] * dts / dts[0], color="0.6", lw=0.8, label="slope 1")
        for x, txt in ((DT_PROD, "SslabLU dt"), (112.5, "Oceananigans dt")):
            if min(DTS) <= x <= max(DTS):
                a.axvline(x, color="0.8", lw=0.8)
                a.text(x, a.get_ylim()[0] * 3 if j == 0 else a.get_ylim()[0] * 3, txt, rotation=90, fontsize=7, color="0.4")
        a.set_title("%s at t = %g h" % (f, T_H)); a.set_xlabel("dt [s]")
        if j == 0:
            a.set_ylabel("relative L2 error")
        a.grid(True, which="major", color="0.92")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "dt_convergence.png", dpi=170)
    print("wrote %s and %s" % ((OUT / "dt_convergence.csv").relative_to(REPO), (OUT / "dt_convergence.png").relative_to(REPO)))
if any(s.startswith("FAILED") for s, _ in status.values()):
    sys.exit(1)
