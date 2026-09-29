#!/usr/bin/env python3
# =============================================================================
# reentrant_channel_sslablu_analysis.py
#
# Companion analysis for reentrant_channel_sslablu.jl. Loads the Oceananigans
# output (data_final.jld2), depth-averages (u, v) into the BAROTROPIC mode, and
# plots SSH + the zonal-mean meridional tilt in the SAME style/units as the
# SslabLU channel_timestep_* diagnostics, so the two can be compared panel by
# panel.
#
# Why depth-average: SslabLU is barotropic; this Oceananigans run is baroclinic
# (Nz > 1). The apples-to-apples comparison is therefore SslabLU vs the
# barotropic / free-surface mode of the Oceananigans run.
#
# Reading JLD2: a JLD2 file is HDF5 underneath, so h5py can read the plain
# numeric arrays -- BUT JLD2 writes Julia (column-major) arrays, so h5py sees
# every multi-dim array with its axes REVERSED. We transpose them back.
#
# Bathymetry H(x,y) is recomputed here from the SslabLU ridge formula (the model
# script does not save it, and this avoids touching that working file).
#   >>> THESE RIDGE CONSTANTS MUST MATCH reentrant_channel_sslablu.jl <<<
#
# Usage:
#   python reentrant_channel_sslablu_analysis.py [path/to/data_final.jld2]
# Default: the most recently written run_oceananigans_channel_*/data_final.jld2.
# The figure is written next to the .jld2, in that run's directory.
# (needs h5py + matplotlib; e.g. the hpsenv conda env, `pip install h5py`)
# =============================================================================

import os
import sys
import glob

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- ridge params: MUST match reentrant_channel_sslablu.jl ------------------
RIDGE_HR, RIDGE_KB = 0.8, 40.0
GAP_DEPTH, GAP_Y0, GAP_Y1, GAP_W = 1.0, 1.0 / 6.0, 1.0 / 2.0, 0.05


def find_file():
    if len(sys.argv) > 1:
        return sys.argv[1]
    cands = glob.glob("run_oceananigans_channel_*/data_final.jld2")
    if not cands:
        sys.exit("no data_final.jld2 found; pass its path as an argument")
    return max(cands, key=os.path.getmtime)


def load(f, name):
    """Read a JLD2/HDF5 dataset and undo the column-major axis reversal."""
    a = np.asarray(f[name][()])
    if a.ndim > 1:
        a = a.transpose()          # reverse all axes -> back to Julia (Nx,Ny,Nz)
    return np.squeeze(a)


def scalar(f, name):
    return f[name][()]


fn = find_file()
with h5py.File(fn, 'r') as f:
    Nx, Ny, Nz = (int(scalar(f, k)) for k in ("Nx", "Ny", "Nz"))
    Lx, Ly, Lz = (float(scalar(f, k)) for k in ("Lx", "Ly", "Lz"))
    ssh = load(f, 'ssh')                 # (Nx, Ny)
    u = np.nan_to_num(load(f, 'u'))      # (Nx, Ny, Nz)  (land cells = 0)
    v = np.nan_to_num(load(f, 'v'))
print("loaded %s :  Nx,Ny,Nz = %d,%d,%d   L = %.3e x %.3e x %.1f" %
      (fn, Nx, Ny, Nz, Lx, Ly, Lz))

# cell-center coordinates
xc = (np.arange(Nx) + 0.5) * Lx / Nx
yc = (np.arange(Ny) + 0.5) * Ly / Ny
X, Y = np.meshgrid(xc, yc, indexing='ij')

# bathymetry H(x,y) from the SslabLU ridge formula (must match the .jl consts)
bump = np.exp(RIDGE_KB * (np.cos(2.0 * np.pi * (X - Lx / 2) / Lx) - 1.0))
gap = 1.0 - 0.5 * GAP_DEPTH * (np.tanh((Y - GAP_Y0 * Ly) / (GAP_W * Ly)) -
                               np.tanh((Y - GAP_Y1 * Ly) / (GAP_W * Ly)))
H = Lz * (1.0 - RIDGE_HR * gap * bump)   # wet depth > 0, in [0.2*Lz, Lz]

# --- barotropic (depth-averaged) velocity --------------------------------
# land cells are 0, so the vertical sum is the wet-column transport; divide by
# the continuous wet depth H. Staggered Face fields may carry an extra face in
# a Bounded direction -> crop to (Nx, Ny, Nz).
dz = Lz / Nz
if Nz == 1:
    crop = lambda a: a[:Nx, :Ny]
    ubar = crop(u) * dz / H
    vbar = crop(v) * dz / H
else:
    crop = lambda a: a[:Nx, :Ny, :Nz]
    ubar = crop(u).sum(axis=2) * dz / H
    vbar = crop(v).sum(axis=2) * dz / H
ssh = ssh[:Nx, :Ny]

eta_zm = ssh.mean(axis=0)     # <eta>_x (y): the meridional tilt
ubar_zm = ubar.mean(axis=0)

# tilt metric, matching SslabLU (mean N third - mean S third of SSH)
tilt = ssh[:, 2 * Ny // 3:].mean() - ssh[:, :Ny // 3].mean()
print("meridional SSH tilt (N third - S third) = %.4f m" % tilt)
print("max |barotropic u| = %.4f m/s   max |SSH| = %.4f m"
      % (np.abs(ubar).max(), np.abs(ssh).max()))

# ---- figure (SslabLU channel_timestep style: x/L, y/L axes, RdBu_r SSH) -----
xl, yl = xc / Lx, yc / Ly
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

pc = ax[0, 0].pcolormesh(xl, yl, H.T, cmap='viridis', shading='auto')
fig.colorbar(pc, ax=ax[0, 0], shrink=0.85, label='H [m]')
ax[0, 0].set_title('bathymetry H(x,y): ridge + gap')
ax[0, 0].set_aspect('equal')

vm = max(np.abs(ssh).max(), 1e-12)
pc = ax[0, 1].pcolormesh(xl, yl, ssh.T, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='auto')
fig.colorbar(pc, ax=ax[0, 1], shrink=0.85, label=r'$\eta$ [m]')
ax[0, 1].set_title(r'SSH $\eta$(x,y): N-high tilt + gap finger')
ax[0, 1].set_aspect('equal')

vm = max(np.abs(ubar).max(), 1e-12)
pc = ax[1, 0].pcolormesh(xl, yl, ubar.T, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='auto')
fig.colorbar(pc, ax=ax[1, 0], shrink=0.85, label=r'$\bar u$ [m/s]')
ax[1, 0].set_title('depth-averaged zonal velocity (barotropic jet)')
ax[1, 0].set_aspect('equal')

ax[1, 1].plot(eta_zm, yl, 'C3', label=r'$\langle\eta\rangle_x$')
ax[1, 1].set_xlabel(r'$\langle\eta\rangle_x$ [m]', color='C3')
ax[1, 1].tick_params(axis='x', labelcolor='C3')
axt = ax[1, 1].twiny()
axt.plot(ubar_zm, yl, 'C0', label=r'$\langle\bar u\rangle_x$')
axt.set_xlabel(r'$\langle\bar u\rangle_x$ [m/s]', color='C0')
axt.tick_params(axis='x', labelcolor='C0')
ax[1, 1].set_ylabel('y / L')
ax[1, 1].set_title('zonal-mean profiles (tilt emerging)')
ax[1, 1].grid(True, alpha=0.3)

for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.set_xlabel('x / L'); a.set_ylabel('y / L')

fig.suptitle('Oceananigans re-entrant channel (barotropic mode) '
             '-- SslabLU comparison', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(fn) or ".", 'oceananigans_channel_diagnostics.png')
fig.savefig(out, dpi=200)
print("wrote " + out)
