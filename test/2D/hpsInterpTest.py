#!/usr/bin/env python3
"""
Test HPSInterp.interp for the 2D hpsalt case.

Function: sum of 2D Helmholtz Green's functions  Re(H0^(1)(k*r))
with source points located outside the unit square, so the function
is smooth on [0,1]^2.  We tile [0,1]^2 with a 4x4 hpsalt Chebyshev
panel grid, sample the function on those nodes, interpolate to a
uniform grid, and measure the max-norm error against the exact values.

Expected result: spectral (exponential) convergence as p increases.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

import solver.HPSInterp as HPSInterp
import solver.spectralmultidomain.hps.cheb_utils as cheb_utils


# ── exact function ──────────────────────────────────────────────────────────

def hankel_sum(pts, sources, k):
    """Re(H0^(1)(k*r)) summed over all sources, evaluated at pts (N,2)."""
    u = np.zeros(pts.shape[0])
    for src in sources:
        r = np.sqrt((pts[:, 0] - src[0])**2 + (pts[:, 1] - src[1])**2)
        u += np.real(scipy.special.hankel1(0, k * r))
    return u


# ── mock solver ─────────────────────────────────────────────────────────────

def build_mock_solver(npan, p):
    """
    Build a minimal object that HPSInterp.interp_2d accepts.

    Attributes required by interp_2d / construct_boxes_2d / local_interp_2d:
      ndim        – space dimension
      npan_dim    – [nx, ny] number of panels per dimension
      geom        – object with .box_geom = [[xmin,ymin],[xmax,ymax]]
      p           – [p, p] Chebyshev order (indexable)
      _XXfull     – (npan^2 * p^2, 2) array of all Chebyshev grid points

    Within each panel the p*p points are ordered in lexicographic order
    (outer loop x, inner loop y) so that after np.unique the p*p unique
    points map correctly to F[k,l] = f(xpts[k], ypts[l]).
    Shared boundary points between adjacent panels appear multiple times
    in _XXfull; np.unique inside local_interp_2d deduplicates them safely.
    """
    cheb_nodes, _ = cheb_utils.cheb(p)          # ascending on [-1, 1]
    dx = 1.0 / npan

    XX_list = []
    for i in range(npan):
        x0, x1 = i * dx, (i + 1) * dx
        xpts = (cheb_nodes + 1) / 2 * (x1 - x0) + x0   # ascending on [x0, x1]
        for j in range(npan):
            y0, y1 = j * dx, (j + 1) * dx
            ypts = (cheb_nodes + 1) / 2 * (y1 - y0) + y0  # ascending on [y0, y1]
            XX_box = np.empty((p * p, 2))
            XX_box[:, 0] = np.repeat(xpts, p)   # outer index → x
            XX_box[:, 1] = np.tile(ypts, p)     # inner index → y
            XX_list.append(XX_box)

    class _Geom:
        box_geom = np.array([[0.0, 0.0], [1.0, 1.0]])

    class _Solver:
        ndim     = 2
        npan_dim = np.array([npan, npan])
        geom     = _Geom()
        _XXfull  = np.vstack(XX_list)

    _Solver.p = [p, p]
    return _Solver()


# ── problem parameters ──────────────────────────────────────────────────────

kh   = 60.0   # wavenumber
npan = 4      # 4×4 panel grid

# Sources strictly outside [0,1]^2
sources = np.array([
    [-0.60,  0.30],
    [ 1.55,  0.70],
    [ 0.20, -0.55],
    [ 0.80,  1.60],
    [-0.50,  0.85],
    [ 1.40, -0.30],
    [-0.30,  0.60],
    [ 1.20,  0.15],
    [ 0.50, -0.70],
    [ 0.10,  1.45],
    [ 1.60,  0.90],
    [-0.70,  0.10],
    [ 0.90, -0.40],
    [-0.45,  1.20],
    [ 1.35,  0.50],
    [ 0.65,  1.70],
    [-0.20, -0.50],
    [ 1.10,  1.30],
    [ 0.35, -0.65],
    [-0.80,  0.75],
])

# Reference / target: 200×200 uniform grid on [0,1]^2
n_ref = 200
t = np.linspace(0, 1, n_ref)
XX_target = np.column_stack([np.repeat(t, n_ref), np.tile(t, n_ref)])
g_exact   = hankel_sum(XX_target, sources, kh)


# ── convergence study ───────────────────────────────────────────────────────

p_vals = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36]
errors = []

print(f"\nHPS interpolation convergence  (kh={kh}, {npan}×{npan} panels)")
print(f"{'p':>4}   {'max |error|':>14}")
print("-" * 22)
for p in p_vals:
    solver   = build_mock_solver(npan, p)
    f        = hankel_sum(solver._XXfull, sources, kh)
    g_interp = HPSInterp.interp(solver, XX_target, f, 'hpsalt')
    err      = np.max(np.abs(g_interp - g_exact))
    errors.append(err)
    print(f"{p:>4}   {err:>14.3e}")


# ── final visualization (highest p) ────────────────────────────────────────

p_vis    = p_vals[-1]
solver   = build_mock_solver(npan, p_vis)
f        = hankel_sum(solver._XXfull, sources, kh)
g_interp = HPSInterp.interp(solver, XX_target, f, 'hpsalt')

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
kw_sol = dict(origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='RdBu_r')
kw_err = dict(origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')

for ax, data, title, kw in zip(
    axes,
    [g_exact, g_interp, np.log10(np.abs(g_interp - g_exact) + 1e-16)],
    ['Exact  Re Σ H₀⁽¹⁾(kr)', f'HPS interp  (p={p_vis})', 'log₁₀ |error|'],
    [kw_sol, kw_sol, kw_err],
):
    im = ax.imshow(data.reshape(n_ref, n_ref).T, **kw)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.plot(sources[:, 0], sources[:, 1], 'k^', ms=7, label='sources')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

axes[0].legend(fontsize=8)
fig.suptitle(f'HPSInterp 2D  |  kh={kh}, {npan}×{npan} panels, p={p_vis}', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'hpsInterpTest_fields.png'), dpi=150)
plt.show()

# convergence plot
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.semilogy(p_vals, errors, 'o-', lw=2, ms=7)
ax2.set_xlabel('Chebyshev order  p', fontsize=12)
ax2.set_ylabel('max |u_interp − u_exact|', fontsize=12)
ax2.set_title(f'Spectral convergence of HPSInterp 2D  (kh={kh}, {npan}×{npan} panels)', fontsize=11)
ax2.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'hpsInterpTest_convergence.png'), dpi=150)
plt.show()
