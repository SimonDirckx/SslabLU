#!/usr/bin/env python3
# =============================================================================
# channel_ssh_manuscript_fig.py
#
# Manuscript figure for the barotropic channel (Sec. 7.2): the trimmed version
# of channel_ssh_compare.py's diagnostic plot, at the final time and on the
# finest Oceananigans grid present. Three panels, one row (full text width):
#   (a) SslabLU SSH eta - <eta> with bathymetry contours H/H0 (shows the ridge
#       and the gap: the problem itself)
#   (b) SslabLU - Oceananigans on its finest grid (the sanity check; the grid
#       size is printed and belongs in the caption)
#   (c) zonal-mean SSH profiles of both models, with an inset of the relative
#       L2 difference vs the Oceananigans grid spacing (first-order decay:
#       the difference is Oceananigans' discretization error, so it bounds
#       the model difference rather than measuring SslabLU's error)
#
# Inputs, as in channel_ssh_compare.py: a SslabLU run directory (or its
# channel_timestep_ssh.npz) and every run_oceananigans_channel_n*/ whose name
# carries the same dt, nsteps and ridge tokens. Means are removed from both
# fields; the volumes are printed instead (SslabLU's should be ~0).
#
# Usage (from the repo root):
#   python test/validation/channel_ssh_manuscript_fig.py [sslablu_run_dir | ssh.npz] [a.jld2 ...]
# Default run: manuscript_ssh_compare_p16_N8_pan16x32_rb/ (dense red-black,
# p = 16, npan_y = 32, N = 8: the most accurate run of the ridgectr sweeps).
#   SSLABLU_FIG_INSET  1 = draw the convergence inset in (c)  (default 1)
#
# Outputs, into the SslabLU run directory:
#   channel_ssh_manuscript.pdf, channel_ssh_manuscript.png
# =============================================================================

import os
import re
import sys
import glob

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- ridge params: MUST match both model scripts ----------------------------
RIDGE_HR, RIDGE_KB = 0.8, 40.0
GAP_DEPTH, GAP_Y0, GAP_Y1, GAP_W = 1.0, 1.0 / 6.0, 1.0 / 2.0, 0.05
H_LEVELS = [0.3, 0.5, 0.7, 0.9]          # bathymetry contours, H / H0
INSET = os.environ.get("SSLABLU_FIG_INSET", "1") != "0"


def depth_frac(x, y, xc_ridge):
    """H(x,y)/H0 as in channel_barotropic_timestep.py (x, y in units of L)."""
    bump = np.exp(RIDGE_KB * (np.cos(2.0 * np.pi * (x - xc_ridge)) - 1.0))
    gap = 1.0 - 0.5 * GAP_DEPTH * (np.tanh((y - GAP_Y0) / GAP_W) -
                                   np.tanh((y - GAP_Y1) / GAP_W))
    return 1.0 - RIDGE_HR * gap * bump


def load(f, name):
    """JLD2/HDF5 dataset with the column-major axis reversal undone."""
    a = np.asarray(f[name][()])
    if a.ndim > 1:
        a = a.transpose()
    return np.squeeze(a)


# ---- inputs -----------------------------------------------------------------
args = sys.argv[1:]
jl_paths = [a for a in args if a.endswith(".jld2")]
s_args = [a for a in args if not a.endswith(".jld2")]
npz_path = s_args[0] if s_args else "manuscript_ssh_compare_p16_N8_pan16x32_rb"
if os.path.isdir(npz_path):
    npz_path = os.path.join(npz_path, "channel_timestep_ssh.npz")
out_dir = os.path.dirname(npz_path) or "."

S = np.load(npz_path)
L, dt, nsteps = float(S["L"]), float(S["dt"]), int(S["nsteps"])
ridge_xc = float(S["ridge_xc"])
if not jl_paths:
    pattern = "run_oceananigans_channel_n*_dt%gs_nsteps%d%s/data_final.jld2" % (
        dt, nsteps, "_ridgectr" if abs(ridge_xc - 0.5) < 1e-12 else "")
    jl_paths = glob.glob(pattern)
    if not jl_paths:
        sys.exit("no Oceananigans output matching %s; pass .jld2 paths" % pattern)

# every Oceananigans resolution: final-time de-meaned SSH difference
runs = {}
for jl in jl_paths:
    with h5py.File(jl, 'r') as f:
        n = int(f["Nx"][()])
        if (int(f["nsteps"][()]) != nsteps or float(f["dt"][()]) != dt
                or abs(float(f["ridge_xc"][()]) - ridge_xc) > 1e-12):
            sys.exit("%s does not match the SslabLU run (dt, nsteps or ridge crest)" % jl)
        if "eta_final_%d" % n not in S:
            print("skip %s: no SslabLU samples at n = %d" % (jl, n))
            continue
        etaO = load(f, "ssh")[:n, :n]
        runs[n] = {"xc": load(f, "xc") / L, "yc": load(f, "yc") / L,
                   "etaO": etaO - etaO.mean(), "meanO": etaO.mean(), "t": float(f["t"][()])}
    etaS = S["eta_final_%d" % n]
    runs[n]["etaS"] = etaS - float(S["eta_mean_final"])
    d = runs[n]["etaS"] - runs[n]["etaO"]
    runs[n]["rel"] = np.linalg.norm(d) / np.linalg.norm(runs[n]["etaS"])
    if abs(float(S["t_final"]) - runs[n]["t"]) > 1e-6 * runs[n]["t"]:
        sys.exit("final times differ: SslabLU %.0f s, Oceananigans %.0f s"
                 % (float(S["t_final"]), runs[n]["t"]))

ns = sorted(runs)
R = runs[ns[-1]]                         # finest Oceananigans grid
xc, yc = R["xc"], R["yc"]
X, Y = np.meshgrid(xc, yc, indexing='ij')
# SSH in mm, the difference in micrometres (no colorbar offset)
etaS, etaO, d = 1.0e3 * R["etaS"], 1.0e3 * R["etaO"], 1.0e6 * (R["etaS"] - R["etaO"])

print("=============SUMMARY (manuscript figure)=============")
print("SslabLU run              = ", npz_path)
print("  p, N, npan             =  %d, %d, %dx%d" % (int(S["p"]), int(S["N"]),
                                                     int(S["npan_x"]), int(S["npan_y"])))
print("final time               = ", '%.1f h' % (R["t"] / 3600.0))
print("domain-mean SSH S / O    =  %.3E / %.3E m" % (float(S["eta_mean_final"]), R["meanO"]))
for n in ns:
    print("rel L2 diff, O at %4d^2 = " % n, '%10.3E' % runs[n]["rel"])
if len(ns) > 1:
    rate = np.polyfit(np.log(1.0 / np.array(ns)), np.log([runs[n]["rel"] for n in ns]), 1)[0]
    print("observed order in dx     = ", '%.2f' % rate)
print("max |diff| at %d^2       = " % ns[-1], '%.1f um' % np.abs(d).max())
print("======================================================")

# ---- figure -------------------------------------------------------------------
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.55), layout="constrained",
                       gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
Hfrac = depth_frac(X, Y, ridge_xc)


def bathy(a):
    """Unlabeled H/H0 contours (levels H_LEVELS; stated in the caption)."""
    a.contour(X, Y, Hfrac, levels=H_LEVELS, colors="0.2", linewidths=0.4, alpha=0.6)


def map_axes(a, title):
    a.set_aspect("equal")
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
    a.set_xlabel(r"$x/L$")
    a.set_title(title)


# (a) SslabLU SSH + bathymetry
vm = np.abs(etaS).max()
pc = ax[0].pcolormesh(xc, yc, etaS.T, cmap="RdBu_r", vmin=-vm, vmax=vm,
                      shading="auto", rasterized=True)
bathy(ax[0])
map_axes(ax[0], r"(a) SslabLU $\eta-\langle\eta\rangle$ [mm]")
ax[0].set_ylabel(r"$y/L$")
cb = fig.colorbar(pc, ax=ax[0], shrink=0.72, pad=0.02)

# (b) difference to Oceananigans on its finest grid
dm = np.abs(d).max()
pc = ax[1].pcolormesh(xc, yc, d.T, cmap="PuOr_r", vmin=-dm, vmax=dm,
                      shading="auto", rasterized=True)
bathy(ax[1])
map_axes(ax[1], r"(b) SslabLU $-$ Oceananigans [$\mu$m]")
ax[1].set_yticklabels([])
cb = fig.colorbar(pc, ax=ax[1], shrink=0.72, pad=0.02)

# (c) zonal means, plus the convergence inset
ax[2].plot(etaS.mean(axis=0), yc, color="C0", lw=1.4, label="SslabLU")
ax[2].plot(etaO.mean(axis=0), yc, color="C3", lw=1.0, ls="--", label="Oceananigans")
ax[2].axvline(0.0, color="0.7", lw=0.5)
ax[2].set_ylim(0, 1); ax[2].set_yticks([0, 0.5, 1]); ax[2].set_yticklabels([])
ax[2].set_xlabel(r"$\langle\eta\rangle_x$ [mm]")
ax[2].set_title("(c) zonal mean")
ax[2].legend(loc="upper left", frameon=False, handlelength=1.8)
ax[2].set_box_aspect(1.0)
if INSET and len(ns) > 1:
    # lower right, below the profile (which crosses 0 near y/L = 0.45);
    # dotted: first-order reference through the finest point
    ia = ax[2].inset_axes([0.60, 0.14, 0.37, 0.27])
    dx = L / 1.0e3 / np.array(ns)                         # km
    rel = np.array([runs[n]["rel"] for n in ns])
    ia.loglog(dx, rel, "o-", color="k", ms=2.5, lw=0.8)
    ia.loglog(dx, rel[-1] * dx / dx[-1], ":", color="0.5", lw=0.8)
    ia.set_xlabel(r"$\Delta x_{\rm O}$ [km]", fontsize=5.5, labelpad=0.5)
    ia.set_ylabel("rel. diff.", fontsize=5.5, labelpad=0.5)
    ia.tick_params(labelsize=5, length=1.5, pad=1)
    ia.minorticks_off()
    ia.set_xticks(dx); ia.set_xticklabels(["%.1f" % v for v in dx])

for ext in ("pdf", "png"):
    path = os.path.join(out_dir, "channel_ssh_manuscript.%s" % ext)
    fig.savefig(path, dpi=300)
    print("wrote %s" % path)
