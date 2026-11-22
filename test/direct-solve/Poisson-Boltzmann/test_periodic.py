"""
test_periodic: Approach 1 on the full periodic twisted elliptic torus.

  1. Global coarse problem: ONE periodic coarse slab over t in [0, L) (end
     faces glued), nested in the fine slabs, coarsened in h and/or p per
     direction; Newton (fast path: geometry once, set_c per step).
  2. Fine OMS for -Lap + c_ref, c_ref = d_s g(P phi_c) per slab: local solver
     'cylhps' fast (CylinderSlabOps, factored once, geometry freed), HBS
     compression of the interface maps, cyclic red-black preconditioner --
     the same machinery as full_twisted_torus_oms.py.
  3. Outer iteration on the interface traces (Algorithm 1):
         u_{k+1} = u_k - P (u_k - E(u_k)),   u_0 = traces of P phi_c,
     E_i = local nonlinear chord solve on double slab i (warm start, local
     tolerance eta |R|), reusing the OMS factors and leaf operators.
     P: 'rb' (one red-black application), 'gmres' (exact (I-K)^{-1} by GMRES on
     the LU operator, red-black preconditioned), 'none' (P = I).

Reports per outer iteration: |R|_inf, interface error vs phi*, local chord
steps; at the end the volume error per slab.  CSV (appended) in --out.

Run from this folder with the repo root on PYTHONPATH, e.g.
    PYTHONPATH=../../.. python test_periodic.py --Ns 32 --nz 2 --nm 8,2 --p 8 --coarsen 2,2,1 --pc 6 --Lam 1
Memory: per fine slab about (leaf factors + Y) + MUMPS factor + HBS blocks,
~0.6 GB at n=8, m=2, nz=2, p=8; the script prints an estimate before building.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, gmres

import geometry.geom_3D.full_twisted_torus as torus
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import direct_solve.omsdirectsolveHBS_torch_new2 as omsdirectHBS_new
from direct_solve.omsdirectsolveHBS_torch_new2 import rkStrat as rkStrat_new
from circInterfaceTree import circInterfaceTree

from pb.manufactured import PBManufactured, transform_pdo
from pb.local_chord import CondensedSlab
from pb.prolong import Prolongation, check_prolongation
from pb.coarse import coarse_newton_fast
from pb.interface_iter import InterfaceIteration

import scipy
from packaging.version import Version

torch.set_default_dtype(torch.double)
torus.s = 0.55
GMRES_TOL = {"rtol": 1e-12} if Version(scipy.__version__) >= Version("1.12") else {"tol": 1e-12}


# ---------------------------------------------------------------- helpers
def ints(s):
    v = tuple(int(x) for x in s.split(","))
    return v[0] if len(v) == 1 else v


def triple(v):
    return (v, v, v) if np.ndim(v) == 0 else tuple(v)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, default=32, help="slabs around the torus (2^k, cyclic red-black)")
    ap.add_argument("--nz", type=int, default=2)
    ap.add_argument("--nm", type=str, default="8,2", help="fine n,m (circInterfaceTree: n = 4m)")
    ap.add_argument("--p", type=ints, default=8, help="fine order: p or p_n,p_m,p_z")
    ap.add_argument("--coarsen", type=ints, default=(2, 2, 1), help="leaf ratios r_n,r_m,r_z")
    ap.add_argument("--pc", type=ints, default=6, help="coarse order: p or p_n,p_m,p_z")
    ap.add_argument("--Lam", type=float, nargs="+", default=[1.0])
    ap.add_argument("--cont", action="store_true", help="continuation in Lam (coarse Newton)")
    ap.add_argument("--newton-tol", type=float, default=1e-10)
    ap.add_argument("--newton-maxit", type=int, default=30)
    ap.add_argument("--precond", type=str, default="rb", choices=["rb", "gmres", "none"])
    ap.add_argument("--tol", type=float, default=1e-10, help="outer tolerance on |R|_inf")
    ap.add_argument("--maxit", type=int, default=50, help="outer iterations")
    ap.add_argument("--eta", type=float, default=0.1, help="local tol = eta |R|")
    ap.add_argument("--local-tol-min", type=float, default=1e-13)
    ap.add_argument("--local-maxit", type=int, default=50)
    ap.add_argument("--rk0", type=int, default=100, help="OMS block compression rank (assembler)")
    ap.add_argument("--rk-start", type=int, default=150, help="red-black rank at stage 0")
    ap.add_argument("--rk-step", type=int, default=50, help="red-black rank increment per stage")
    ap.add_argument("--rk-max", type=int, default=None, help="cap on the red-black rank")
    ap.add_argument("--device", type=str, default="cpu", help="leaf operators")
    ap.add_argument("--compute-device", type=str, default="cuda", help="red-black solver")
    ap.add_argument("--check", action="store_true", help="check (I-K) of P against the local solves")
    ap.add_argument("--out", type=str, default="results_test_periodic")
    return ap.parse_args()


def write_csv(path, rows):
    """Append rows (header once); new time-stamped file if the header differs."""
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    if os.path.exists(path):
        with open(path, newline="") as f:
            old = next(csv.reader(f), None)
        if old != keys:
            root, ext = os.path.splitext(path)
            path = root + time.strftime("_%Y%m%d-%H%M%S") + ext
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        w.writerows(rows)
    print("wrote" if new else "appended to", path)


def random_torus_points(n, seed=1):
    rng = np.random.default_rng(seed)
    r = torus.Rc * np.sqrt(rng.uniform(size=n)); th = rng.uniform(0, 2 * np.pi, n)
    return np.c_[rng.uniform(0, torus.L, n), r * np.cos(th), r * np.sin(th)]


def lam_schedule(Lam, cont):
    if not cont:
        return [Lam]
    return [l for l in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0) if l < Lam * (1 - 1e-12)] + [Lam]


def clock(dev):
    if torch.device(dev).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------- main
def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    Ns, nz = args.Ns, args.nz
    assert Ns >= 4 and not Ns & (Ns - 1), "cyclic red-black needs Ns = 2^k"
    dSlabs, connectivity, H = torus.dSlabs(Ns)
    n, m = map(int, args.nm.split(","))
    rn, rm, rz = triple(args.coarsen)
    p_f, p_c = args.p, args.pc
    nt_f = Ns * nz // 2                                  # fine axial leaves around the torus
    for name, N_, r in (("n", n, rn), ("m", m, rm), ("axial leaves", nt_f, rz)):
        if N_ % r:
            raise ValueError("%s = %d not divisible by %d" % (name, N_, r))
    nt_c = nt_f // rz
    if nt_c % 2:
        raise ValueError("coarse axial leaf count %d must be even" % nt_c)
    pn, pm, pz = triple(p_f)
    n_iface = n * n * (pn - 2) ** 2 + 4 * m * n * (pm - 2) * (pn - 2)
    mem_slab = (n * n + 4 * m * n) * nz * (2 * (pn - 2) ** 6) * 8 / 1e9   # leaf LU + Y, rough
    print("torus: L = %.4f, Ns = %d, H = %.4f, s = %.2f, Rc = %.4f" % (torus.L, Ns, H, torus.s, torus.Rc))
    print("fine  : n=%d m=%d nz=%d p=%s per slab, interface dofs %d, total %d"
          % (n, m, nz, triple(p_f), n_iface, Ns * n_iface))
    print("coarse: periodic, n=%d m=%d axial leaves=%d p=%s"
          % (n // rn, m // rm, nt_c, triple(p_c)))
    print("rough memory: leaf factors %.1f GB total (+ MUMPS factors + HBS blocks)" % (Ns * mem_slab))

    pg = torus.param_geom(jax_avail=False, torch_avail=True, hpsalt=True)
    pdo_geo = transform_pdo(pg, None)                   # geometry only; c via set_c
    dev = torch.device(args.device)

    # ---- coarse geometry (once)
    t0 = time.perf_counter()
    geom_c = [[0.0, -torus.Rc, -torus.Rc], [torus.L, torus.Rc, torus.Rc]]
    cs_c = CondensedSlab(geom_c, pdo_geo, p_c,
                         dict(Rc=torus.Rc, s=torus.s, n=n // rn, m=m // rm, nz=nt_c,
                              flux="conormal", device=dev, periodic=True),
                         fast=True, verbose=False)
    print("coarse geometry: %.2fs, N_skel %d (periodic: Ib = wall only, %d dofs)"
          % (time.perf_counter() - t0, len(cs_c.slab.Ii), len(cs_c.slab.Ib)))
    grid_c = cs_c.slab.grid

    # interface tree / assembler / rank strategy as in full_twisted_torus_oms.py
    max_level = int(np.log2(n ** 2)) - 2

    def make_tree(XXI):
        XXI = XXI.detach().cpu().numpy() if torch.is_tensor(XXI) else np.asarray(XXI)
        return circInterfaceTree(XXI[:, 1:3], Rc=torus.Rc, s=torus.s, n_core=n,
                                 m_arch=m, max_level=max_level)
    leaf_size = n_iface // 2 ** max_level
    gb = lambda p: torus.gb(p, jax_avail=False, torch_avail=True)

    for Lam in args.Lam:
        print("== Lam = %g" % Lam)
        row = dict(Ns=Ns, H=H, n=n, m=m, nz=nz, p=str(triple(p_f)), nc=n // rn, mc=m // rm,
                   ntc=nt_c, pc=str(triple(p_c)), Lam=Lam, precond=args.precond)
        mms = PBManufactured(torus, Lam=Lam)
        mms.calibrate_A(random_torus_points(20000), 3.0)
        row["A"] = mms.A

        # ---- 1. global coarse Newton
        phi_c, clog = coarse_newton_fast(mms, cs_c, lam_schedule=lam_schedule(Lam, args.cont),
                                         tol=args.newton_tol, maxit=args.newton_maxit)
        err_c = np.max(np.abs(phi_c - mms.phistar(grid_c.nodes)))
        print("  coarse: %d Newton steps, %.2fs (set_c %.2fs, factor %.2fs, solve %.2fs per step); err %.2e"
              % (clog["newton_steps"], clog["t_total"], clog["t_set_c_per_step"],
                 clog["t_factor_per_step"], clog["t_solve_per_step"], err_c))
        row.update(newton_steps=clog["newton_steps"], t_coarse=clog["t_total"], err_coarse=err_c)

        # ---- 2. fine OMS for -Lap + c_ref, c_ref from P phi_c per slab
        store_c, store_phi, info = {}, {}, {}

        def c_nodal(ops, geom):
            i = int(round(float(geom[0][0]) / H)) % Ns
            Pr = Prolongation(grid_c, ops.grid, t_offset=i * nz // 2, nt_global=nt_f)
            if i == 0:
                info["P_exact"] = check_prolongation(Pr)
            phf = Pr(phi_c)
            c = mms.dg(ops.nodes(), phf)
            store_c[i], store_phi[i] = c, phf
            return c

        def bc(pts):
            pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
            return mms.phistar(pts)

        cyl_opts = dict(Rc=torus.Rc, s=torus.s, n=n, m=m, nz=nz, flux="conormal",
                        fast=True, c_nodal=c_nodal, keep_geometry=False, device=dev)
        opts = solverWrap.solverOptions("cylhps", p_f, None, use_ctxT=False, mumps_stats=False,
                                        reduced_gpu=False, cyl=cyl_opts)
        assembler = mA.rkHMatAssembler(leaf_size, args.rk0, ndim=3, tree=make_tree)
        t0 = time.perf_counter()
        OMS = oms.oms(dSlabs, pdo_geo, gb, opts, connectivity, stiff_mat_const=False,
                      constructHBS=True, keepLU=True, keepDense=False, offload=False)
        Stot, _ = OMS.construct_Stot_and_rhstot(bc, assembler, dbg=0)
        t_oms = time.perf_counter() - t0
        print("  fine OMS built in %.1fs (Ntot %d); prolongation exactness (slab 0) %.2e"
              % (t_oms, OMS.Ntot, info.get("P_exact", np.nan)))
        row.update(t_oms=t_oms, Ntot=OMS.Ntot, P_exact=info.get("P_exact"))

        t0 = clock(args.compute_device)
        kw = {} if args.rk_max is None else dict(rk_max=args.rk_max)
        strat = rkStrat_new.linear(args.rk_start, args.rk_step, skip_first_level=False, **kw)
        ranks = [int(strat.rank(st)) for st in range(int(np.log2(Ns)))]
        print("  red-black ranks per stage %s (HBS leaf size %d)" % (ranks, leaf_size))
        row.update(rk_asm=args.rk0, rk_start=args.rk_start, rk_step=args.rk_step,
                   rk_max=args.rk_max, rk_stages=str(ranks))
        rb = omsdirectHBS_new.RedBlackSolverHBS(
            strat, OMS, tree=make_tree(OMS.interface_points), quad=False,
            compute_device=args.compute_device, seed=None, dbg=0, normalized=False,
            constant_blocks=False, operator="hbs", cyclic=True)
        rb.factorize()
        t_rb = clock(args.compute_device) - t0
        print("  red-black: setup + factor %.1fs" % t_rb)
        row["t_rb"] = t_rb

        if args.precond == "rb":
            P = rb.solve
        elif args.precond == "none":
            P = lambda v: v
        else:
            M = LinearOperator(Stot.shape, matvec=rb.solve, dtype=np.float64)

            def P(v):
                x, _ = gmres(Stot, v, M=M, restart=50, maxiter=4, **GMRES_TOL)
                return x

        # ---- 3. outer iteration
        it = InterfaceIteration(OMS, mms, bc, store_c, store_phi)
        if args.check:
            print("  check: |(I-K) v - Stot v| / |Stot v| = %.2e (expect ~1e-12)" % it.check_jacobian(Stot))
        u_exact = OMS.interface_values(lambda pts: mms.phistar(np.asarray(pts)))
        u0 = it.traces_of_fields()
        print("  u0 = traces of P phi_c: err_iface %.2e" % np.max(np.abs(u0 - u_exact)))
        t0 = time.perf_counter()
        u, logs = it.run(P, u0, u_exact=u_exact, tol=args.tol, maxit=args.maxit, eta=args.eta,
                         local_tol_min=args.local_tol_min, local_maxit=args.local_maxit)
        t_outer = time.perf_counter() - t0
        ev = it.volume_errors()
        print("  -> %d outer its, |R| %.2e, err_iface %.2e, volume err max %.2e (per slab min %.2e)"
              % (len(logs), logs[-1]["residual"], logs[-1]["err_iface"], ev.max(), ev.min()))
        row.update(outer_its=len(logs), residual=logs[-1]["residual"],
                   err_iface=logs[-1]["err_iface"], err_volume=ev.max(), t_outer=t_outer,
                   local_steps_total=int(sum(l["local_steps_mean"] * Ns for l in logs)),
                   err_iface_u0=np.max(np.abs(u0 - u_exact)))
        write_csv(os.path.join(args.out, "test_periodic.csv"), [row])
        write_csv(os.path.join(args.out, "test_periodic_outer.csv"),
                  [dict(Ns=Ns, p=str(triple(p_f)), Lam=Lam, precond=args.precond, **l) for l in logs])
        OMS.close()
        del OMS, rb, it


if __name__ == "__main__":
    main()
