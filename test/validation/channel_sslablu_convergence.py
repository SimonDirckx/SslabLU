#!/usr/bin/env python3
# =============================================================================
# channel_sslablu_convergence.py
#
# Convergence of the SslabLU barotropic channel (channel_barotropic_timestep.py,
# Neumann walls) against the finest Oceananigans runs
# (reentrant_channel_sslablu.jl), used as the reference solution. Unlike
# channel_ssh_compare.py / channel_wall_convergence.py, the Oceananigans grid is
# FIXED (n_ref x n_ref, the finest present) and SslabLU is refined:
#   channel_sslablu_convergence_h.png    error vs slab count N, one line per p
#   channel_sslablu_convergence_dof.png  error vs SslabLU degrees of freedom
# Each figure has one column per Oceananigans dt at n_ref (every run reaching
# the SslabLU final time) and one row per output time (mid, final).
#
# Error: ||f_S - f_O||_2 / ||f_O||_2 on the n_ref cell centers (the timestep
# script samples SslabLU's leaf interpolant there, so the Oceananigans side gets
# no interpolation). The field is set by SSLABLU_CONV_FIELD:
#   eta      SSH, both de-meaned (default). The volume error is reported apart,
#            as in channel_ssh_compare.py
#   eta_raw  SSH as is: includes SslabLU's volume error (Oceananigans conserves
#            volume to round-off)
#   u, v     velocities, the Oceananigans C-grid averaged to cell centers
#
# The reference has its own discretization error, which floors every curve.
# Where an Oceananigans run at n_ref/2 with the same dt exists, the gray line
# is ||O_nref/2 - R O_nref|| / ||R O_nref||, R = 2x2 cell averaging. That is the
# error of O_nref itself if Oceananigans converges at first order in dx, as the
# Oceananigans-resolution sweeps of channel_ssh_compare.py show.
#
# DoF: unique collocation unknowns of the global leaf tiling,
#   nx ny p^2 + p nx (2 ny + 1),   nx = N npan_x / 2,  ny = npan_y:
# p^2 interior Chebyshev nodes per leaf (p_disc = p + 2 per direction) plus the
# shared non-corner edge nodes (x periodic; the Neumann y-walls are unknowns).
# Leaf corners are not dofs in hpsalt.
#
# Runs that differ only in the slab solver (thomas / rb / rbhbs, any rk)
# collapse into one point, represented by a dense-solver run when there is one;
# the largest relative spread between them is in the SUMMARY.
#
# Usage (from the repo root):
#   python test/validation/channel_sslablu_convergence.py [run_sslablu_channel_*_neumann/ ...]
# Default: every run_sslablu_channel_*_neumann*/ holding a
# channel_timestep_ssh.npz (runs still in progress are skipped) at the SslabLU
# dt SSLABLU_DT_H [h] (default 0.125 = 450 s, as in channel_barotropic_timestep.py).
#   SSLABLU_CONV_NREF    Oceananigans reference n (default: the finest present)
#   SSLABLU_CONV_FIELD   eta | eta_raw | u | v   (default eta)
#
# Outputs (current directory; a _<field> suffix for fields other than eta):
#   channel_sslablu_convergence_h.png, channel_sslablu_convergence_dof.png
#   channel_sslablu_convergence.csv    one row per SslabLU run, reference and time
# =============================================================================

import os
import sys
import glob

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']      # one per line, never cycled
# must agree across all SslabLU runs for one figure to make sense
PHYS = ("H0", "L", "tau0", "rdrag", "steric_amp", "gamma_s", "ridge_xc", "t_mid", "t_final")
YLAB = {"eta": r"$\|\eta'_S - \eta'_O\|_2 \,/\, \|\eta'_O\|_2$",
        "eta_raw": r"$\|\eta_S - \eta_O\|_2 \,/\, \|\eta_O\|_2$",
        "u": r"$\|u_S - u_O\|_2 \,/\, \|u_O\|_2$",
        "v": r"$\|v_S - v_O\|_2 \,/\, \|v_O\|_2$"}

FIELD = os.environ.get("SSLABLU_CONV_FIELD", "eta")
if FIELD not in YLAB:
    raise ValueError("SSLABLU_CONV_FIELD must be one of %s, got %r" % (", ".join(YLAB), FIELD))
dt_S = 3600.0 * float(os.environ.get("SSLABLU_DT_H", "0.125"))
SFX = "" if FIELD == "eta" else "_" + FIELD


def load(f, name):
    """JLD2/HDF5 dataset with the column-major axis reversal undone."""
    a = np.asarray(f[name][()])
    if a.ndim > 1:
        a = a.transpose()
    return np.squeeze(a)


def centered_uv(u, v, n):
    """Oceananigans C-grid (u on x-faces, periodic; v on y-faces, n+1 of
    them) -> cell centers, to match SslabLU's collocated samples."""
    u = u[:n, :n]
    uc = 0.5 * (u + np.roll(u, -1, axis=0))
    vc = 0.5 * (v[:n, :n] + v[:n, 1:n + 1])
    return uc, vc


def restrict(a):
    """2x2 cell average: the n/2 x n/2 finite-volume cell values of an n x n field."""
    m = a.shape[0] // 2
    return a.reshape(m, 2, m, 2).mean(axis=(1, 3))


def ocean_fields(path):
    """{tag: {field: n x n cell-center array}} at the mid and final output times."""
    with h5py.File(path, 'r') as f:
        n = int(f["Nx"][()])
        out = {}
        for tag, sfx in (("mid", "_mid"), ("final", "")):
            eta = load(f, "ssh" + sfx)[:n, :n]
            u, v = centered_uv(load(f, "u" + sfx), load(f, "v" + sfx), n)
            out[tag] = {"eta": eta - eta.mean(), "eta_raw": eta, "u": u, "v": v}
    return out


def sslablu_fields(S, tag, n):
    """SslabLU samples at the n x n Oceananigans cell centers; eta is
    de-meaned with SslabLU's own (quadrature) domain mean."""
    eta = S["eta_%s_%d" % (tag, n)]
    return {"eta": eta - float(S["eta_mean_" + tag]), "eta_raw": eta,
            "u": S["u_%s_%d" % (tag, n)], "v": S["v_%s_%d" % (tag, n)]}


def rel_l2(a, ref):
    return np.linalg.norm(a - ref) / np.linalg.norm(ref)


def dof(N, p, npan_x, npan_y):
    nx, ny = N * npan_x // 2, npan_y
    return nx * ny * p * p + p * nx * (2 * ny + 1)


# ---- SslabLU runs -------------------------------------------------------------
dirs = [a.rstrip("/") + "/" for a in sys.argv[1:]] or glob.glob("run_sslablu_channel_*_neumann*/")
dirs = sorted(dirs, key=lambda d: ("_rbhbs" in d, d))   # dense solvers represent a config
runs = {}          # (p, N, npan_x, npan_y) -> list of npz (first = representative)
phys = {}          # PHYS tuple -> dirs
for d in dirs:
    f = os.path.join(d, "channel_timestep_ssh.npz")
    if not os.path.exists(f):
        print("skip %s: no channel_timestep_ssh.npz (still running?)" % d)
        continue
    S = np.load(f)
    if "walls" not in S or str(S["walls"]) != "neumann":
        print("skip %s: not Neumann walls" % d)
        continue
    if abs(float(S["dt"]) - dt_S) > 1e-9 * dt_S:
        print("skip %s: dt = %g s (SSLABLU_DT_H selects %g s)" % (d, float(S["dt"]), dt_S))
        continue
    phys.setdefault(tuple(float(S[k]) for k in PHYS), []).append(d)
    runs.setdefault((int(S["p"]), int(S["N"]), int(S["npan_x"]), int(S["npan_y"])), []).append(S)
    L, t_mid, t_final, ridge_xc = float(S["L"]), float(S["t_mid"]), float(S["t_final"]), float(S["ridge_xc"])

if not runs:
    sys.exit("no finished Neumann SslabLU runs at dt = %g s" % dt_S)
if len(phys) > 1:
    sys.exit("runs differ in %s; pass a consistent subset of:\n  " % ", ".join(PHYS)
             + "\n  ".join("%s: %s" % (k, " ".join(v)) for k, v in phys.items()))

# ---- Oceananigans runs: references (n_ref) and n_ref/2 for the error estimate --
ocean = {}         # (n, dt) -> path
for c in glob.glob("run_oceananigans_channel_n*/data_final.jld2"):
    with h5py.File(c, 'r') as f:
        if "xc" not in f or "ssh_mid" not in f:
            continue
        ok = (abs(float(f["t"][()]) - t_final) < 1e-6 * t_final
              and abs(float(f["t_mid"][()]) - t_mid) < 1e-6 * t_mid
              and abs((float(f["ridge_xc"][()]) if "ridge_xc" in f else 0.5) - ridge_xc) < 1e-12)
        if ok:
            ocean[(int(f["Nx"][()]), float(f["dt"][()]))] = c
NREF = int(os.environ.get("SSLABLU_CONV_NREF", "0")) or max(n for n, _ in ocean)
refs = sorted((dt for n, dt in ocean if n == NREF), reverse=True)
if not refs:
    sys.exit("no Oceananigans n = %d run reaching t = %g s with ridge x/L = %g" % (NREF, t_final, ridge_xc))
for key in list(runs):
    if "eta_final_%d" % NREF not in runs[key][0]:
        print("skip p=%d N=%d: no SslabLU samples at n = %d (set SSLABLU_SSH_NS)" % (key[0], key[1], NREF))
        del runs[key]
O = {dtO: ocean_fields(ocean[(NREF, dtO)]) for dtO in refs}
Oh = {dtO: ocean_fields(ocean[(NREF // 2, dtO)]) for dtO in refs if (NREF // 2, dtO) in ocean}
TIMES = (("mid", t_mid), ("final", t_final))

# reference error estimate (spatial) and dt -> dt/2 differences (temporal)
est = {(dtO, tag): rel_l2(Oh[dtO][tag][FIELD], restrict(O[dtO][tag][FIELD]))
       for dtO in Oh for tag, _ in TIMES}

# ---- errors -------------------------------------------------------------------
err = {}           # key -> {(dtO, tag): {field: rel L2}}
spread = (0.0, None)
for key, Ss in runs.items():
    err[key] = {}
    for dtO in refs:
        for tag, _ in TIMES:
            F = sslablu_fields(Ss[0], tag, NREF)
            err[key][(dtO, tag)] = {fl: rel_l2(F[fl], O[dtO][tag][fl]) for fl in YLAB}
    for S in Ss[1:]:                     # other slab solvers, same configuration
        e0 = err[key][(refs[0], "final")][FIELD]
        rel = abs(rel_l2(sslablu_fields(S, "final", NREF)[FIELD], O[refs[0]]["final"][FIELD]) - e0) / e0
        if rel > spread[0]:
            spread = (rel, "p=%d N=%d" % key[:2])

# one line per p (and panel rule npan_x, npan_y / N, so h-refinement lines stay uniform)
lines = {}
for key in runs:
    p, N, npx, npy = key
    lines.setdefault((p, npx, npy / N), []).append(key)
lines = {k: sorted(v, key=lambda q: q[1]) for k, v in sorted(lines.items())}
if len(lines) > len(MARKS):
    sys.exit("%d lines but %d markers; pass a subset of the runs" % (len(lines), len(MARKS)))
rules = {k[1:] for k in lines}


def rule_txt(npx, r):
    """panel layout npan_x x npan_y as a function of N, e.g. 4xN"""
    return r"%d$\times$%sN" % (npx, "" if r == 1 else "%g" % r)


def line_label(k):
    if len(rules) == 1:
        return "p = %d" % k[0]
    return "p = %d, panels %s" % (k[0], rule_txt(*k[1:]))


# ---- figures --------------------------------------------------------------------
def convergence_figure(xof, xlabel, out, log2=False, label_n=False):
    nc = len(refs)
    fig, ax = plt.subplots(2, nc, figsize=(4.6 * nc + 0.6, 8.2), sharex=True, sharey=True, squeeze=False)
    for j, dtO in enumerate(refs):
        for i, (tag, t) in enumerate(TIMES):
            a = ax[i, j]
            for k, (lk, keys) in enumerate(lines.items()):
                x = [xof(q) for q in keys]
                a.plot(x, [err[q][(dtO, tag)][FIELD] for q in keys], '-', color=COL[k],
                       marker=MARKS[k], ms=7, lw=1.6, mec="white", mew=0.8, zorder=3)
                if label_n and i == 0 and j == 0:
                    for q, xx in zip(keys, x):
                        a.annotate("N=%d" % q[1], (xx, err[q][(dtO, tag)][FIELD]), xytext=(5, 4),
                                   textcoords="offset points", fontsize=8, color="0.4")
            if (dtO, tag) in est:
                a.axhline(est[(dtO, tag)], color="0.45", ls=":", lw=1.3, zorder=2)
            a.set_xscale("log", base=2 if log2 else 10)
            a.set_yscale("log")
            a.grid(True, which="major", color="0.88", lw=0.6)
            if i == 0:
                a.set_title(r"vs Oceananigans %d$^2$, $\Delta t$ = %g s" % (NREF, dtO), fontsize=11)
            if j == 0:
                a.set_ylabel("t = %.0f h\n" % (t / 3600.0) + YLAB[FIELD])
            if i == 1:
                a.set_xlabel(xlabel)
    if log2:
        Ns = sorted({q[1] for q in runs})
        ax[0, 0].set_xticks(Ns)
        ax[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % v))
        ax[0, 0].xaxis.set_minor_locator(NullLocator())
        lo, hi = Ns[0], Ns[-1]
        ax[0, 0].set_xlim(lo / 1.3, hi * 1.3)
    ax[0, 0].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax[0, 0].yaxis.set_minor_locator(NullLocator())
    h = [Line2D([], [], color=COL[k], marker=MARKS[k], ms=7, lw=1.6, mec="white", mew=0.8,
                label=line_label(lk)) for k, lk in enumerate(lines)]
    if est:
        h.append(Line2D([], [], color="0.45", ls=":", lw=1.3,
                        label=r"est. error of the reference, $\|O_{%d} - R\,O_{%d}\|\,/\,\|R\,O_{%d}\|$"
                        % (NREF // 2, NREF, NREF)))
    fig.suptitle(r"SslabLU (Neumann walls, $\Delta t$ = %g s) vs fixed Oceananigans %d$^2$ reference"
                 % (dt_S, NREF), fontsize=12)
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote " + out)


xrule = ("; panels " + rule_txt(*next(iter(rules)))) if len(rules) == 1 else ""
convergence_figure(lambda q: q[1], "slabs N (h-refinement%s)" % xrule,
                   "channel_sslablu_convergence_h%s.png" % SFX, log2=True)
convergence_figure(lambda q: dof(q[1], q[0], q[2], q[3]), "SslabLU degrees of freedom",
                   "channel_sslablu_convergence_dof%s.png" % SFX, label_n=True)

# ---- CSV ----------------------------------------------------------------------
rows = []
for key in sorted(runs):
    p, N, npx, npy = key
    S = runs[key][0]
    for dtO in refs:
        for tag, t in TIMES:
            e = err[key][(dtO, tag)]
            rows.append([p, N, npx, npy, dof(N, p, npx, npy), dt_S, dtO, NREF, t / 3600.0,
                         e["eta"], e["eta_raw"], e["u"], e["v"], float(S["eta_mean_" + tag]),
                         est.get((dtO, tag), np.nan), len(runs[key])])
rows = np.array(rows)
csv_name = "channel_sslablu_convergence.csv"
with open(csv_name, 'w') as f:
    f.write("p,N,npan_x,npan_y,dof,dt_S,dt_O,n_ref,t_hours,relL2_eta,relL2_eta_raw,relL2_u,"
            "relL2_v,mean_S,ref_err_est_%s,n_solver_variants\n" % FIELD)
    np.savetxt(f, rows, fmt='%.16e', delimiter=',')
print("wrote %s (%d rows)" % (csv_name, rows.shape[0]))

# ---- SUMMARY ------------------------------------------------------------------
print("")
print("=============SUMMARY  (%s, Oceananigans %d^2 reference, SslabLU dt = %g s)============="
      % (FIELD, NREF, dt_S))
print("  p   N  panels       DoF   |<eta>| final   rel L2 at t = %.0f h vs dt_O = %s s   solvers"
      % (t_final / 3600.0, " / ".join("%g" % dtO for dtO in refs)))
for key in sorted(runs):
    p, N, npx, npy = key
    print(" %2d  %2d  %2dx%-3d  %9d    %10.3E     %s   %d"
          % (p, N, npx, npy, dof(N, p, npx, npy), abs(float(runs[key][0]["eta_mean_final"])),
             " / ".join("%9.3E" % err[key][(dtO, "final")][FIELD] for dtO in refs), len(runs[key])))
print("reference self-differences at t = %.0f h (%s):" % (t_final / 3600.0, FIELD))
for dtO in refs:
    if (dtO, "final") in est:
        print("  spatial   ||O_%d - R O_%d|| / ||R O_%d||   dt = %6g s : %9.3E"
              % (NREF // 2, NREF, NREF, dtO, est[(dtO, "final")]))
for d1, d2 in zip(refs[:-1], refs[1:]):
    print("  temporal  ||O_%d(%g s) - O_%d(%g s)|| / ||O_%d(%g s)||  : %9.3E"
          % (NREF, d1, NREF, d2, NREF, d2, rel_l2(O[d1]["final"][FIELD], O[d2]["final"][FIELD])))
if spread[1] is not None:
    print("slab solver variants agree to %.1E relative (worst: %s)" % spread)
print("=================================================")
