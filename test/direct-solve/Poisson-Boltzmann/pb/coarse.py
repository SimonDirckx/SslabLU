"""
Coarse nonlinear solve for c_ref (HANDOFF Sec. 5, Stage 2).

Newton on the coarse condensed slab system.  A Newton step for
-Lap phi + G[phi] = f* has the same form as a chord step with c_ref refreshed:

    phi_{k+1} = M_k^{-1} (f* - G[phi_k] + c[phi_k] phi_k),   M_k = -Lap + c[phi_k],

so each step builds a CondensedSlab with the NODAL coefficient c[phi_k]
(NodalField) and does one solve.  Optional continuation in Lam.
"""
import time
import numpy as np
from scipy.spatial import cKDTree

from pb.local_chord import CondensedSlab
from pb.manufactured import PBManufactured, transform_pdo
from pb.prolong import LeafGrid


class NodalField:
    """Callable c(W) returning stored values at known nodes W (OMS coords), by
    exact coordinate lookup.  hpsmultidomain evaluates the zero-order
    coefficient exactly at the leaf-grid nodes; only leaf-INTERIOR values enter
    the operator (they are unique), so duplicated face nodes are harmless."""

    def __init__(self, W, values, tol=1e-9):
        self.tree = cKDTree(np.asarray(W, float))
        self.values = np.asarray(values, float)
        self.tol = tol * max(1.0, float(np.max(np.abs(W))))

    def __call__(self, W):
        d, i = self.tree.query(np.asarray(W, float))
        if d.max() > self.tol:
            raise ValueError("NodalField queried off its nodes (dist %.2e)" % d.max())
        return self.values[i]


def build_nodal_slab(geom, pg, grid, cvals, p, cyl, use_mumps=True, verbose=True):
    """CondensedSlab for -Lap + c with c given at grid.nodes (a LeafGrid).
    cyl: dict(Rc, s, n, m, nz, flux, device) matching grid."""
    cs = CondensedSlab(geom, transform_pdo(pg, NodalField(grid.nodes, cvals)), p, cyl,
                       use_mumps=use_mumps, verbose=verbose)
    if cs.W.shape != grid.nodes.shape or not np.allclose(cs.W, grid.nodes, atol=1e-12, rtol=0):
        raise RuntimeError("LeafGrid and CylinderSlab node orders differ")
    return cs


def coarse_newton(mms, geom, pg, disc, cyl, lam_schedule=None, tol=1e-10, maxit=30,
                  phi0=None, use_mumps=True, verbose=True):
    """Newton (with optional continuation in Lam) on the coarse slab.

    mms   : PBManufactured at the TARGET Lam (A, B, kappa, theta reused)
    disc  : dict(p, n, m, nz) of the coarse discretization (p int or (p_n, p_m, p_z))
    cyl   : dict(Rc, s, flux, device)
    lam_schedule : increasing Lam values ending at mms.Lam (default: [mms.Lam])
    Returns (phi_c on grid.nodes, grid, log dict)."""
    t_start = time.perf_counter()
    grid = LeafGrid(geom, disc["p"], cyl["Rc"], cyl["s"], disc["n"], disc["m"], disc["nz"])
    W = grid.nodes
    lam_schedule = [mms.Lam] if lam_schedule is None else list(lam_schedule)
    assert np.isclose(lam_schedule[-1], mms.Lam)
    cyl_full = dict(cyl, n=disc["n"], m=disc["m"], nz=disc["nz"])
    phi = np.zeros(len(W)) if phi0 is None else np.asarray(phi0, float).copy()
    steps, t_build, t_solve, N_skel = [], 0.0, 0.0, None
    for Lam in lam_schedule:
        mk = PBManufactured(mms.geom, Lam=Lam, A=mms.A, B=mms.B, kappa=mms.kappa, theta=mms.theta)
        psi = mk.psi(W)
        fst = mk.fstar(W)
        for k in range(maxit):
            c = mk.dg(W, phi, psi)
            t0 = time.perf_counter()
            cs = CondensedSlab(geom, transform_pdo(pg, NodalField(W, c)), disc["p"], cyl_full,
                               use_mumps=use_mumps, verbose=False)
            t_build += time.perf_counter() - t0
            if k == 0 and not np.allclose(cs.W, W, atol=1e-12, rtol=0):
                raise RuntimeError("LeafGrid and CylinderSlab node orders differ")
            N_skel = cs.slab.Aii.shape[0]
            b = mk.phistar(cs.Wb)
            t0 = time.perf_counter()
            phi_new, x = cs.solve(fst - mk.g(W, phi, psi) + c * phi, b)
            t_solve += time.perf_counter() - t0
            inc = np.max(np.abs(phi_new - phi))
            if not np.isfinite(inc):
                raise FloatingPointError("coarse Newton diverged at Lam=%g, step %d" % (Lam, k + 1))
            phi = phi_new
            steps.append(dict(Lam=Lam, step=k + 1, increment=inc))
            if verbose:
                print("    coarse Newton Lam=%-6g step %2d  inc %.2e" % (Lam, k + 1, inc))
            del cs
            if inc < tol:
                break
    log = dict(steps=steps, newton_steps=len(steps), t_total=time.perf_counter() - t_start,
               t_build=t_build, t_solve=t_solve, N_skel=N_skel, N_nodes=len(W))
    return phi, grid, log


def coarse_newton_fast(mms, cs, lam_schedule=None, tol=1e-10, maxit=30, phi0=None, verbose=True):
    """As coarse_newton, on a prebuilt CondensedSlab(..., fast=True): the
    geometry is built once; each Newton step is set_c(c[phi_k]) (batched leaf
    factorizations + CSR scatter + MUMPS refactorization) and one solve.
    Returns (phi_c on cs.W, log dict)."""
    t_start = time.perf_counter()
    W = cs.W
    lam_schedule = [mms.Lam] if lam_schedule is None else list(lam_schedule)
    assert np.isclose(lam_schedule[-1], mms.Lam)
    phi = np.zeros(len(W)) if phi0 is None else np.asarray(phi0, float).copy()
    steps = []
    t_setc = t_fact = t_solve = 0.0
    n_reused = 0
    for Lam in lam_schedule:
        mk = PBManufactured(mms.geom, Lam=Lam, A=mms.A, B=mms.B, kappa=mms.kappa, theta=mms.theta)
        psi = mk.psi(W)
        fst = mk.fstar(W)
        b = mk.phistar(cs.Wb)
        for k in range(maxit):
            c = mk.dg(W, phi, psi)
            t0 = time.perf_counter()
            cs.set_c(c)
            t_setc += cs.t_set_c
            t_fact += time.perf_counter() - t0 - cs.t_set_c
            n_reused += bool(cs.factor.get("analysis_reused"))
            t0 = time.perf_counter()
            phi_new, x = cs.solve(fst - mk.g(W, phi, psi) + c * phi, b)
            t_solve += time.perf_counter() - t0
            inc = np.max(np.abs(phi_new - phi))
            if not np.isfinite(inc):
                raise FloatingPointError("coarse Newton diverged at Lam=%g, step %d" % (Lam, k + 1))
            phi = phi_new
            steps.append(dict(Lam=Lam, step=k + 1, increment=inc))
            if verbose:
                print("    coarse Newton Lam=%-6g step %2d  inc %.2e" % (Lam, k + 1, inc))
            if inc < tol:
                break
    n = len(steps)
    log = dict(steps=steps, newton_steps=n, t_total=time.perf_counter() - t_start,
               t_set_c_per_step=t_setc / n, t_factor_per_step=t_fact / n,
               t_solve_per_step=t_solve / n, analysis_reused=n_reused,
               N_skel=cs.slab.Aii.shape[0], N_nodes=len(W))
    return phi, log
