"""
test_coarse: coarse solve -> c_coarse -> fine local chord iteration (standalone).

Single double slab [0, 2H] x D_Rc of the full twisted elliptic torus,
H = L / Ns, exact Dirichlet data phi* (manufactured solution, model (A)).

  1. Coarse discretization, nested in the fine one (same patch maps), coarsened
     in h and/or p independently per direction:
         tangential (core x, y and arch eta):  n  -> n  / r_n,   p_n -> pc_n
         radial     (arch xi)               :  m  -> m  / r_m,   p_m -> pc_m
         axial      (t)                     :  nz -> nz / r_z,   p_z -> pc_z
     (nz / r_z must stay even: the midplane is a leaf face.)
  2. Coarse Newton (refactor per step, coarse slab only), optional continuation
     in Lam:  phi_c.
  3. Prolongation P (per child-position matrices), c_coarse = d_s g(P phi_c).
  4. Fine slab built ONCE with c_ref = c_coarse; local chord iteration
         phi_{j+1} = M^{-1}(f* - G[phi_j] + c_coarse phi_j),   phi_0 = P phi_c,
     in statically condensed form (reduce / skeleton solve / reconstruct).
     Optional baseline: the same with the constant midpoint c_bar.

Reports: P exactness check; coarse Newton steps, coarse error, error after
prolongation, predicted rho = max|1 - c[phi*]/c_coarse|; fine chord iterations,
observed rate, final errors (nodes / midplane / off-grid); timings.
CSV (appended): test_coarse.csv, test_coarse_iterations.csv in --out.

Paths: default "fast" -- each slab's geometry is built once (CylinderSlabOps);
a new coefficient costs set_c (batched leaf LU + DtN + CSR scatter) plus a MUMPS
refactorization, and each solve does one batched leaf solve.  --reference runs
the old path (slab rebuilt per coefficient, leaf ops re-factored per call);
--check-ops compares the two (operators and one solve) at the first Lam.

Run from this folder with the repo root on PYTHONPATH, e.g.
    # cross-section h-coarsening (nz = 2 cannot be coarsened in t)
    PYTHONPATH=../../.. python test_coarse.py --Ns 64 --nz 2 --nm 8,2 --p 8 --coarsen 2,2,1 --pc 6
    # full 3D h-coarsening
    PYTHONPATH=../../.. python test_coarse.py --Ns 32 --nz 4 --nm 8,2 --p 8 --coarsen 2,2,2 --pc 6
    # p-only coarsening, anisotropic
    PYTHONPATH=../../.. python test_coarse.py --Ns 64 --p 8 --coarsen 1,1,1 --pc 4,4,6
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from scipy.sparse.linalg import norm as spla_norm

import geometry.geom_3D.full_twisted_torus as torus
from pb.manufactured import PBManufactured, transform_pdo
from pb.local_chord import CondensedSlab, chord
from pb.prolong import LeafGrid, Prolongation, check_prolongation
from pb.coarse import coarse_newton, coarse_newton_fast, build_nodal_slab

torch.set_default_dtype(torch.double)
torus.s = 0.55


# ---------------------------------------------------------------- helpers
def ints(s):
    v = tuple(int(x) for x in s.split(","))
    return v[0] if len(v) == 1 else v


def triple(v):
    return (v, v, v) if np.ndim(v) == 0 else tuple(v)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, default=64, help="slabs around the torus: H = L / Ns")
    ap.add_argument("--nz", type=int, default=2)
    ap.add_argument("--nm", type=str, default="8,2", help="fine n,m")
    ap.add_argument("--p", type=ints, default=8, help="fine order: p or p_n,p_m,p_z")
    ap.add_argument("--coarsen", type=ints, default=(2, 2, 1),
                    help="leaf ratios r_n,r_m,r_z (fine / coarse)")
    ap.add_argument("--pc", type=ints, default=None,
                    help="coarse order: p or p_n,p_m,p_z (default: fine order)")
    ap.add_argument("--Lam", type=float, nargs="+", default=[1.0, 10.0, 30.0])
    ap.add_argument("--cont", action="store_true",
                    help="continuation in Lam for the coarse Newton (1, 3, 10, 30, ...)")
    ap.add_argument("--newton-tol", type=float, default=1e-10)
    ap.add_argument("--newton-maxit", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-12, help="fine chord tolerance (increment)")
    ap.add_argument("--maxit", type=int, default=200)
    ap.add_argument("--baseline", action="store_true", help="also run the constant c_bar chord")
    ap.add_argument("--spread", type=float, default=3.0, help="variation of phi*/tau")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no-mumps", action="store_true")
    ap.add_argument("--reference", action="store_true",
                    help="old path: rebuild slabs per coefficient, leaf ops re-factored per call")
    ap.add_argument("--check-ops", action="store_true",
                    help="compare the fast operators/solve against the reference path (first Lam)")
    ap.add_argument("--out", type=str, default="results_test_coarse")
    return ap.parse_args()


def random_slab_points(n, geom, seed=1):
    rng = np.random.default_rng(seed)
    r = torus.Rc * np.sqrt(rng.uniform(size=n)); th = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(geom[0][0], geom[1][0], n)
    return np.c_[t, r * np.cos(th), r * np.sin(th)]


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


def lam_schedule(Lam, cont):
    if not cont:
        return [Lam]
    return [l for l in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0) if l < Lam * (1 - 1e-12)] + [Lam]


def run_chord(cs, mms, cref, phi0, Xoff, args, label):
    """Fine local chord with the slab cs (built with c_ref = cref at cs.W)."""
    W, I = cs.W, cs.interior
    psi = mms.psi(W)
    fst = mms.fstar(W)
    ex = mms.phistar(W)
    b = mms.phistar(cs.Wb)
    Xmid = cs.slab.XX[cs.Imid]
    ex_mid = mms.phistar(Xmid)
    h_of = lambda ph: fst - mms.g(W, ph, psi) + cref * ph
    rho_star = np.max(np.abs(1 - mms.dg(W, ex, psi)[I] / cref[I]))
    rho_it = [np.max(np.abs(1 - mms.dg(W, phi0, psi)[I] / cref[I]))]
    state = dict(prev=None)

    def monitor(j, ph, x, log):
        e_nodes = np.max(np.abs(ph - ex))
        e_mid = np.max(np.abs(x[cs.Imid] - ex_mid))
        rho_it.append(np.max(np.abs(1 - mms.dg(W, ph, psi)[I] / cref[I])))
        rate = np.nan if state["prev"] is None else log["increment"] / state["prev"]
        state["prev"] = log["increment"]
        log.update(err_nodes=e_nodes, err_mid=e_mid, rate=rate, rho_pred=max(rho_star, max(rho_it)))
        print("    [%s] it %3d  inc %.2e  res(prev) %.2e  err %.2e  mid %.2e  rate %.2e"
              % (label, j, log["increment"], log["residual_prev"], e_nodes, e_mid, rate))

    cs.timers = dict(reduce=0.0, skeleton=0.0, reconstruct=0.0)
    print("  [%s] rho* = %.3e" % (label, rho_star))
    # iterate 0 = starting guess (P phi_c).  It lives on the leaf grid only, so
    # there is no skeleton/midplane value; the off-grid error is shown instead.
    err0_nodes = np.max(np.abs(phi0 - ex))
    err0_offgrid = np.max(np.abs(cs.interp(phi0) - mms.phistar(Xoff)))
    print("    [%s] it   0  (start)                     err %.2e  off-grid %.2e"
          % (label, err0_nodes, err0_offgrid))
    t0 = time.perf_counter()
    phi, x, logs = chord(lambda h: cs.solve(h, b), phi0, h_of, tol=args.tol,
                         maxit=args.maxit, monitor=monitor, interior=I)
    t_chord = time.perf_counter() - t0
    k = len(logs)
    rates = [l["rate"] for l in logs if np.isfinite(l["rate"]) and l["increment"] > 1e3 * args.tol]
    if not rates:      # converged within a few steps: use every finite rate
        rates = [l["rate"] for l in logs if np.isfinite(l["rate"]) and l["rate"] > 0]
    res = dict(
        err0_nodes=err0_nodes, err0_offgrid=err0_offgrid,
        iterations=k, converged=bool(logs[-1]["increment"] < args.tol),
        final_increment=logs[-1]["increment"],
        err_nodes=np.max(np.abs(phi - ex)), err_mid=np.max(np.abs(x[cs.Imid] - ex_mid)),
        err_offgrid=np.max(np.abs(cs.interp(phi) - mms.phistar(Xoff))),
        rho_star=rho_star, rho_pred=max(rho_star, max(rho_it)),
        rate_observed=np.exp(np.mean(np.log(rates[-5:]))) if rates else np.nan,
        t_build=cs.t_build, t_set_c=cs.t_set_c if cs.fast else None,
        analysis_reused=cs.factor.get("analysis_reused"),
        t_analysis=cs.factor.get("time_analysis"),
        t_factor=cs.factor.get("time_factor"), nnz_factor=cs.factor.get("nnz_factor"),
        mem_MB=cs.factor.get("mem_MB"), N_skel=cs.slab.Aii.shape[0], t_chord=t_chord,
        t_G_per_it=sum(l["t_G"] for l in logs) / k,
        t_reduce_per_it=cs.timers["reduce"] / k,
        t_skeleton_per_it=cs.timers["skeleton"] / k,
        t_reconstruct_per_it=cs.timers["reconstruct"] / k)
    print("  [%s] -> %d its, err nodes %.2e mid %.2e off-grid %.2e, rate %.3e vs rho_pred %.3e"
          % (label, k, res["err_nodes"], res["err_mid"], res["err_offgrid"],
             res["rate_observed"], res["rho_pred"]))
    return phi, res, logs


# ---------------------------------------------------------------- main
def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    slabs, _, H = torus.dSlabs(args.Ns)
    geom = slabs[0]
    n, m = map(int, args.nm.split(","))
    nz = args.nz
    rn, rm, rz = triple(args.coarsen)
    p_f = args.p
    p_c = p_f if args.pc is None else args.pc
    for name, N, r in (("n", n, rn), ("m", m, rm), ("nz", nz, rz)):
        if N % r:
            raise ValueError("%s = %d not divisible by coarsening ratio %d" % (name, N, r))
    if (nz // rz) % 2:
        raise ValueError("coarse nz = %d must be even (midplane must be a leaf face)" % (nz // rz))
    if any(c > f for c, f in zip(triple(p_c), triple(p_f))):
        print("  note: coarse order exceeds fine order in some direction")
    coarse = dict(p=p_c, n=n // rn, m=m // rm, nz=nz // rz)
    cyl = dict(Rc=torus.Rc, s=torus.s, flux="conormal", device=torch.device(args.device))
    cyl_f = dict(cyl, n=n, m=m, nz=nz)
    cyl_c = dict(cyl, n=coarse["n"], m=coarse["m"], nz=coarse["nz"])
    pg = torus.param_geom(jax_avail=False, torch_avail=True, hpsalt=True)
    use_mumps, fast = not args.no_mumps, not args.reference
    print("slab t in [%.4f, %.4f], H = %.4f (Ns = %d), s = %.2f, Rc = %.4f"
          % (geom[0][0], geom[1][0], H, args.Ns, torus.s, torus.Rc))
    print("fine   : n=%d m=%d nz=%d p=%s" % (n, m, nz, triple(p_f)))
    print("coarse : n=%d m=%d nz=%d p=%s" % (coarse["n"], coarse["m"], coarse["nz"], triple(p_c)))
    print("path   : %s" % ("fast (geometry once, set_c per coefficient)" if fast else "reference"))

    # ---- geometry (once) and prolongation (independent of Lam)
    t_geom_f = t_geom_c = None
    if fast:
        pdo_geo = transform_pdo(pg, None)                 # geometry only; c via set_c
        cs_f = CondensedSlab(geom, pdo_geo, p_f, cyl_f, use_mumps=use_mumps, fast=True)
        cs_c = CondensedSlab(geom, pdo_geo, p_c, cyl_c, use_mumps=use_mumps, fast=True, verbose=False)
        t_geom_f, t_geom_c = cs_f.t_build, cs_c.t_build
        print("  coarse geometry built in %.2fs" % t_geom_c)
        grid_f, grid_c = cs_f.slab.grid, cs_c.slab.grid
    else:
        grid_f = LeafGrid(geom, p_f, cyl["Rc"], cyl["s"], n, m, nz)
        grid_c = LeafGrid(geom, p_c, cyl["Rc"], cyl["s"], coarse["n"], coarse["m"], coarse["nz"])
    t0 = time.perf_counter()
    Pr = Prolongation(grid_c, grid_f)
    t_prolong_setup = time.perf_counter() - t0
    e_P = check_prolongation(Pr)
    print("prolongation: setup %.2fs, exactness on degree p_c-1 polynomials: %.2e (expect ~1e-12)"
          % (t_prolong_setup, e_P))
    Wf = grid_f.nodes

    tag = dict(Ns=args.Ns, H=H, n=n, m=m, nz=nz, p=str(triple(p_f)),
               nc=coarse["n"], mc=coarse["m"], nzc=coarse["nz"], pc=str(triple(p_c)),
               path="fast" if fast else "reference")
    for iLam, Lam in enumerate(args.Lam):
        print("== Lam = %g" % Lam)
        mms = PBManufactured(torus, Lam=Lam)
        mms.calibrate_A(random_slab_points(4000, geom), args.spread)
        sched = lam_schedule(Lam, args.cont)

        # ---- 1-2. coarse Newton
        if fast:
            phi_c, clog = coarse_newton_fast(mms, cs_c, lam_schedule=sched, tol=args.newton_tol,
                                             maxit=args.newton_maxit)
            print("  coarse Newton per step: set_c %.3fs, factor %.3fs, solve %.3fs (analysis reused %d/%d)"
                  % (clog["t_set_c_per_step"], clog["t_factor_per_step"], clog["t_solve_per_step"],
                     clog["analysis_reused"], clog["newton_steps"]))
        else:
            phi_c, _, clog = coarse_newton(mms, geom, pg, coarse, cyl, lam_schedule=sched,
                                           tol=args.newton_tol, maxit=args.newton_maxit,
                                           use_mumps=use_mumps)
        err_coarse = np.max(np.abs(phi_c - mms.phistar(grid_c.nodes)))

        # ---- 3. prolongation, c_coarse
        t0 = time.perf_counter()
        phi_cf = Pr(phi_c)
        t_prolong = time.perf_counter() - t0
        tau_f = mms.tau(mms.psi(Wf))
        err_P = np.max(np.abs(phi_cf - mms.phistar(Wf)))
        rho_est = np.max(np.abs(phi_cf - mms.phistar(Wf)) / tau_f)
        c_coarse = mms.dg(Wf, phi_cf)
        print("  coarse: %d Newton steps, %.2fs (N_skel %d); err coarse %.2e, after P %.2e, "
              "max|P phi_c - phi*|/tau = %.2e"
              % (clog["newton_steps"], clog["t_total"], clog["N_skel"], err_coarse, err_P, rho_est))

        # ---- 4. fine slab with c_coarse, local chord from P phi_c
        if fast:
            cs = cs_f
            cs.set_c(c_coarse)
            fine_setup = cs.t_set_c + (cs.factor.get("time_factor") or 0.0)
        else:
            cs = build_nodal_slab(geom, pg, grid_f, c_coarse, p_f, cyl_f, use_mumps=use_mumps)
            fine_setup = cs.t_build + (cs.factor.get("time_factor") or 0.0)
        if args.check_ops and fast and iLam == 0:
            check_ops(cs, geom, pg, grid_f, c_coarse, p_f, cyl_f, mms, phi_cf, use_mumps)
        Xoff = cs.offgrid_points()
        _, res, logs = run_chord(cs, mms, c_coarse, phi_cf, Xoff, args, "precond with coarse c_ref")
        row = dict(tag, Lam=Lam, A=mms.A, cref="coarse", cont=args.cont,
                   t_geom_fine=t_geom_f, t_geom_coarse=t_geom_c,
                   newton_steps=clog["newton_steps"], t_coarse=clog["t_total"],
                   N_skel_coarse=clog["N_skel"], err_coarse=err_coarse, err_prolong=err_P,
                   rho_est_coarse=rho_est, t_prolong=t_prolong,
                   coarse_over_fine_setup=clog["t_total"] / fine_setup, **res)
        rows = [row]
        iters = [dict(tag, Lam=Lam, cref="coarse", **l) for l in logs]
        if not fast:
            del cs

        # ---- optional baseline: constant midpoint c_bar, same start
        if args.baseline:
            cstar = mms.dg(Wf, mms.phistar(Wf))
            cbar = 0.5 * (cstar.min() + cstar.max())
            if fast:
                cs = cs_f
                cs.set_c(np.full(len(Wf), cbar))
            else:
                pdo = transform_pdo(pg, lambda W, cbar=cbar: cbar + 0.0 * W[:, 0])
                cs = CondensedSlab(geom, pdo, p_f, cyl_f, use_mumps=use_mumps)
            Xoff = cs.offgrid_points()
            _, resb, logsb = run_chord(cs, mms, np.full(len(cs.W), cbar), phi_cf, Xoff, args,
                                       "precond with const c_bar")
            rows.append(dict(tag, Lam=Lam, A=mms.A, cref="const", cont=args.cont, **resb))
            iters += [dict(tag, Lam=Lam, cref="const", **l) for l in logsb]
            if not fast:
                del cs

        write_csv(os.path.join(args.out, "test_coarse.csv"), rows)
        write_csv(os.path.join(args.out, "test_coarse_iterations.csv"), iters)


def check_ops(cs, geom, pg, grid_f, c, p, cyl_f, mms, phi, use_mumps):
    """Fast operators (CylinderSlabOps) vs the reference CylinderSlab with the
    same nodal c: DOF order, Aii, Aib, and one condensed solve.  Expect ~1e-13."""
    print("  check-ops: building the reference slab ...")
    ref = build_nodal_slab(geom, pg, grid_f, c, p, cyl_f, use_mumps=use_mumps, verbose=False)
    S, R = cs.slab, ref.slab
    rel = lambda A, B: spla_norm(A - B) / spla_norm(B)
    W, psi = cs.W, mms.psi(cs.W)
    h = mms.fstar(W) - mms.g(W, phi, psi) + c * phi
    b = mms.phistar(cs.Wb)
    phi1, x1 = cs.solve(h, b)
    phi2, x2 = ref.solve(h, b)
    print("  check-ops: XX %.1e | Ii,Ib equal %s | Aii %.2e | Aib %.2e | solve: x %.2e, phi %.2e"
          % (np.max(np.abs(S.XX - R.XX)),
             np.array_equal(S.Ii, R.Ii) and np.array_equal(S.Ib, R.Ib),
             rel(S.Aii, R.Aii), rel(S.Aib, R.Aib),
             np.max(np.abs(x1 - x2)) / np.max(np.abs(x2)),
             np.max(np.abs(phi1 - phi2)) / np.max(np.abs(phi2))))
    cs.timers = dict(reduce=0.0, skeleton=0.0, reconstruct=0.0)
    del ref


if __name__ == "__main__":
    main()
