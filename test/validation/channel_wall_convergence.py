#!/usr/bin/env python3
# =============================================================================
# channel_wall_convergence.py
#
# True Neumann walls vs the emulated-wall hack, across every SslabLU channel run
# (channel_barotropic_timestep.py) that channel_ssh_compare.py has compared with
# Oceananigans. One figure, three panels:
#   (a), (b) rel L2 of the de-meaned SSH difference SslabLU - Oceananigans vs
#            the Oceananigans grid spacing, at the mid and final output times.
#            FULL domain: Neumann runs have no wall band (their "interior" is
#            the full domain), so it is the only like-for-like mask.
#   (c)      volume error |<eta>|(t) from channel_timestep_diag.csv. The norms
#            in (a), (b) are de-meaned and do not see it; Oceananigans
#            conserves volume to round-off.
# Colour + line style = wall treatment, marker shape = SslabLU resolution
# (N, p, panels), marker fill + line weight = dt: filled markers on thick
# lines for the smallest dt, larger hollow markers on thin lines for the next
# one (half-filled after that). Where two dt agree, the hollow marker rings the
# filled one. Panel (c) shows the smallest dt only.
#
# Runs that differ only in the slab solver (thomas / rb / rbhbs, any rk) are
# collapsed into one configuration, represented by a dense-solver run when
# there is one; the largest relative spread between them is in the SUMMARY.
# The emulated runs carry the wall sponge they need (sponge_rate > 0, the old
# default); it is part of what the "emulated" curves measure.
#
# Usage (from the repo root, after channel_ssh_compare.py has written
# channel_ssh_compare.csv into each run directory):
#   python test/validation/channel_wall_convergence.py [run_sslablu_channel_*/ ...]
# Default: every run_sslablu_channel_*/ holding a channel_ssh_compare.csv.
# The runs must share the physics (forcing, ridge, output times); pass a
# consistent subset otherwise.
#
# Outputs (current directory):
#   channel_wall_convergence.png   the three-panel figure
#   channel_wall_convergence.csv   one row per configuration, n and output time
# =============================================================================

import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator

COL = {"neumann": "#2a78d6", "emulated": "#eb6834"}
LS = {"neumann": "-", "emulated": "--"}
WLAB = {"neumann": "true Neumann walls", "emulated": "emulated walls (old hack)"}
MARKS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']    # one per resolution, never cycled
FILL = ['full', 'none', 'left', 'bottom']           # marker fill per dt, smallest first
# must agree across all runs for one figure to make sense
PHYS = ("H0", "L", "tau0", "rdrag", "steric_amp", "gamma_s", "ridge_xc", "t_mid", "t_final")


def walls_of(S):
    """npz files from before SSLABLU_WALLS carry only wall_noflux (1 was the
    emulated walls, 0 the Dirichlet ones)."""
    if "walls" in S:
        return str(S["walls"])
    return "emulated" if bool(S["wall_noflux"]) else "dirichlet"


def rows_at(C, is_final):
    """compare-CSV rows at one output time, sorted by Oceananigans n."""
    C = C[C[:, 2] == is_final]
    return C[np.argsort(C[:, 0])]


# ---- inputs -----------------------------------------------------------------
dirs = [a.rstrip("/") + "/" for a in sys.argv[1:]] or glob.glob("run_sslablu_channel_*/")
dirs = sorted(dirs, key=lambda d: ("_rbhbs" in d, d))   # dense solvers represent a config

runs = {}          # (walls, N, p, npan_x, npan_y, dt) -> run data
phys = {}          # PHYS tuple -> dirs
sponge = {}        # walls -> set of sponge rates
spread = (0.0, None)
for d in dirs:
    csv = os.path.join(d, "channel_ssh_compare.csv")
    if not os.path.exists(csv):
        print("skip %s: no channel_ssh_compare.csv (run channel_ssh_compare.py on it)" % d)
        continue
    S = np.load(os.path.join(d, "channel_timestep_ssh.npz"))
    walls = walls_of(S)
    if walls not in COL:
        print("skip %s: %s walls" % (d, walls))
        continue
    phys.setdefault(tuple(float(S[k]) for k in PHYS), []).append(d)
    sponge.setdefault(walls, set()).add(float(S["sponge_rate"]))
    key = (walls, int(S["N"]), int(S["p"]), int(S["npan_x"]), int(S["npan_y"]), float(S["dt"]))
    C = np.atleast_2d(np.loadtxt(csv, delimiter=",", skiprows=1))
    if key in runs:                      # another slab solver for the same config
        A = runs[key]["C"]
        if A.shape == C.shape:
            rel = np.max(np.abs(A[:, 3] - C[:, 3]) / np.abs(A[:, 3]))
            if rel > spread[0]:
                spread = (rel, d)
        runs[key]["variants"] += 1
        continue
    D = np.genfromtxt(os.path.join(d, "channel_timestep_diag.csv"), delimiter=",", names=True)
    runs[key] = {"C": C, "dir": d, "variants": 1, "sponge": float(S["sponge_rate"]),
                 "t": D["t_hours"], "mass": D["mass"],
                 "vn": D["wall_vn_max"][-1] if "wall_vn_max" in D.dtype.names else np.nan}
    L, t_mid, t_final = float(S["L"]), float(S["t_mid"]), float(S["t_final"])

if not runs:
    sys.exit("no compared Neumann/emulated runs found")
if len(phys) > 1:
    sys.exit("runs differ in %s; pass a consistent subset of:\n  " % ", ".join(PHYS)
             + "\n  ".join("%s: %s" % (k, " ".join(v)) for k, v in phys.items()))
for walls, rates in sponge.items():
    if len(rates) > 1:
        sys.exit("%s runs with different sponge rates %s; pass a consistent subset"
                 % (walls, sorted(rates)))
    if max(rates) > 0:
        WLAB[walls] = WLAB[walls].replace(" (", " + sponge (")

res = sorted({k[1:5] for k in runs})                  # (N, p, npan_x, npan_y)
if len(res) > len(MARKS):
    sys.exit("%d resolutions but %d markers; pass a subset of the runs" % (len(res), len(MARKS)))
MK = dict(zip(res, MARKS))
dts = sorted({k[5] for k in runs})
if len(dts) > len(FILL):
    sys.exit("%d dt values but %d marker fills; pass a subset of the runs" % (len(dts), len(FILL)))
dt_line = dts[0]
# hollow / half-filled markers take the line colour as their edge colour
DTS = {dt: dict(ms=7, lw=1.6, mec="white", mew=0.8, zorder=3) if i == 0 else
           dict(ms=11, lw=0.9, fillstyle=FILL[i], mew=1.2, zorder=4)
       for i, dt in enumerate(dts)}
ns = sorted({int(n) for r in runs.values() for n in r["C"][:, 0]})

# ---- (a), (b): SSH difference vs Oceananigans dx; (c): volume error ---------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.9))
for k, (is_final, t) in enumerate(((0, t_mid), (1, t_final))):
    a = ax[k]
    best = None                                       # anchors the O(dx) guide
    for key, r in sorted(runs.items()):
        walls, dt = key[0], key[5]
        C = rows_at(r["C"], is_final)
        dx = L / C[:, 0] / 1e3
        a.loglog(dx, C[:, 3], LS[walls], color=COL[walls], marker=MK[key[1:5]], **DTS[dt])
        if dt == dt_line and (best is None or (walls == "neumann", -C[-1, 3])
                              > (best[0] == "neumann", -best[2][-1])):
            best = (walls, dx, C[:, 3])
    _, dxr, er = best
    guide = 0.6 * er[0] * dxr / dxr[0]
    a.loglog(dxr, guide, ":", color="0.35", lw=1.2)
    m = len(dxr) // 2
    a.text(dxr[m], guide[m] * 0.8, r"$O(\Delta x)$", color="0.35", fontsize=9, ha="center", va="top")
    a.set_title("(%s)  t = %.0f h" % ("ab"[k], t / 3600.0), fontsize=11)
    a.set_xlabel(r"Oceananigans $\Delta x$ [km]")
    a.set_xticks([L / n / 1e3 for n in ns])
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    a.xaxis.set_minor_locator(NullLocator())
    a.grid(True, which="major", color="0.88", lw=0.6)
ax[0].set_ylabel(r"$\|\eta'_S - \eta'_O\|_2 \,/\, \|\eta'_S\|_2$  (full domain)")
ylo = min(a.get_ylim()[0] for a in ax[:2])
yhi = max(a.get_ylim()[1] for a in ax[:2])
for a in ax[:2]:
    a.set_ylim(ylo, yhi)
    a.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    a.yaxis.set_minor_locator(NullLocator())

a = ax[2]
top = 0.0
for key, r in sorted(runs.items()):
    if key[5] != dt_line:
        continue
    a.semilogy(r["t"], np.abs(r["mass"]), LS[key[0]], color=COL[key[0]], lw=1.4,
               marker=MK[key[1:5]], ms=6, markevery=max(len(r["t"]) // 8, 1),
               mec="white", mew=0.8)
    top = max(top, np.abs(r["mass"]).max())
a.set_ylim(top * 1e-6, top * 3.0)
a.set_xlim(0, t_final / 3600.0)
a.set_xlabel("t [h]")
a.set_ylabel(r"$|\langle\eta\rangle|$ [m]  (Oceananigans: round-off)")
a.set_title(r"(c)  volume error, $\Delta t$ = %g s" % dt_line, fontsize=11)
a.grid(True, which="major", color="0.88", lw=0.6)

blocks = [[Line2D([], [], color=COL[w], ls=LS[w], lw=1.6, label=WLAB[w]) for w in COL if w in sponge],
          [Line2D([], [], color="0.4", ls="none", marker=MK[q], ms=7,
                  label=r"N = %d, p = %d, panels %d$\times$%d" % q) for q in res],
          [Line2D([], [], color="0.4", marker="o", label=r"$\Delta t$ = %g s" % dt, **DTS[dt])
           for dt in dts] if len(dts) > 1 else [],
          [Line2D([], [], color="0.35", ls=":", lw=1.2, label=r"$O(\Delta x)$")]]
h = []
for b in blocks:                  # pad each block to whole legend columns (2 rows)
    h += b + [Line2D([], [], ls="none", label=" ")] * (-len(b) % 2)
fig.legend(handles=h, loc="lower center", ncol=len(h) // 2, fontsize=9, frameon=False)
fig.tight_layout(rect=[0, 0.12, 1, 1])
out = "channel_wall_convergence.png"
fig.savefig(out, dpi=200)
plt.close(fig)
print("wrote " + out)

# ---- CSV ----------------------------------------------------------------------
rows = []
for key, r in sorted(runs.items()):
    walls, N, p, npx, npy, dt = key
    for c in r["C"]:
        rows.append([1 if walls == "neumann" else 0, N, p, npx, npy, dt, r["sponge"],
                     c[0], c[1], c[3], c[4], c[5], c[9], c[11], c[12], r["variants"]])
rows = np.array(rows)
csv_name = "channel_wall_convergence.csv"
with open(csv_name, 'w') as f:
    f.write("neumann,N,p,npan_x,npan_y,dt,sponge_rate,n,t_hours,relL2_full,relL2_interior,"
            "maxabs_full,mean_S,relL2_u_interior,relL2_v_interior,n_solver_variants\n")
    np.savetxt(f, rows, fmt='%.16e', delimiter=',')
print("wrote %s (%d rows)" % (csv_name, rows.shape[0]))

# ---- SUMMARY ------------------------------------------------------------------
print("")
print("=============SUMMARY  (finest Oceananigans grid, n = %d)=============" % ns[-1])
print("walls      N   p  panels  dt[s]  sponge    relL2 mid / final      |<eta>| final  max|v.n| final  solvers")
for key, r in sorted(runs.items()):
    walls, N, p, npx, npy, dt = key
    mid, fin = rows_at(r["C"], 0)[-1], rows_at(r["C"], 1)[-1]
    print("%-9s %2d %3d  %2dx%-3d %5.0f  %7.1E   %9.3E / %9.3E   %10.3E     %10.3E      %d"
          % (walls, N, p, npx, npy, dt, r["sponge"], mid[3], fin[3],
             abs(r["mass"][-1]), r["vn"], r["variants"]))
if spread[1] is not None:
    print("slab solver variants agree to %.1E relative in relL2_full (worst: %s)" % spread)
print("=================================================")
