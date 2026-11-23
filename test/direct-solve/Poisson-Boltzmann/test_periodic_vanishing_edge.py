"""
test_periodic_vanishing_edge: Approach 1 on the full periodic twisted elliptic
torus for the PHYSICAL problem (no manufactured solution):

    -Lap phi + Lam^2 N(psi) exp(phi / tau(psi)) = 0,   phi = 0 on the wall,
    N = 1 - psi^2 (vanishing edge; --N linear gives 1 - psi/2),  tau = 1 - 0.4 psi.

Same pipeline and performance path as test_periodic.py:
  1. global coarse problem: one periodic coarse slab, fast Newton (geometry
     once, set_c per step), continuation in Lam from phi = 0 (a supersolution:
     Newton on this convex problem then decreases monotonically);
  2. fine OMS for -Lap + c_ref, c_ref = d_s g(P phi_c) per slab ('cylhps' fast,
     factored once, geometry freed), HBS compression, cyclic red-black;
  3. outer iteration u_{k+1} = u_k - P (u_k - E(u_k)) from the traces of P phi_c.

No exact solution, so it reports instead:
  - |R| per outer iteration and local chord steps;
  - coarse vs fine: max |phi_fine - P phi_c| (nodes) and at the probes;
  - overlap consistency of neighbouring slabs (expect ~ the iteration tolerance);
  - departure of phi from a flux function on psi <= --psi-max, and F(psi);
  - phi at fixed probe points, saved to --out; --compare <npz> prints the max
    difference against another run (self-convergence across resolutions).

Run from this folder with the repo root on PYTHONPATH, e.g.
    PYTHONPATH=../../.. python test_periodic_vanishing_edge.py --Ns 32 --nz 2 --nm 8,2 --p 8 --coarsen 2,2,1 --pc 6 --Lam 20
  radially clustered arch leaves (graded toward the wall, ratio 0.5 per leaf):
    ... --nm 16,4 --arch-ratio 0.5        (or --arch-stretch <beta>)
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
import arch_stretch

from pb.manufactured import transform_pdo
from pb.profiles import PBProfiles
from pb.local_chord import CondensedSlab
from pb.prolong import Prolongation, check_prolongation
from pb.coarse import coarse_newton_fast
from pb.interface_iter import InterfaceIteration
from pb import diagnostics as diag
from pb.estimate import ResidualEstimator, COMPONENTS, TIMERS, format_loc, save_dump

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
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arch-stretch", type=float, default=0.0,
                   help="radial clustering of the arch leaves toward the wall (arch_stretch.py): "
                        "beta > 0, leaf widths ~ exp(-beta k / m); 0 = uniform")
    g.add_argument("--arch-ratio", type=float, default=None,
                   help="alternative to --arch-stretch: ratio sigma of successive radial leaf "
                        "widths on the FINE grid (beta = -m ln sigma; e.g. 0.5)")
    ap.add_argument("--p", type=ints, default=8, help="fine order: p or p_n,p_m,p_z")
    ap.add_argument("--coarsen", type=ints, default=(2, 2, 1), help="leaf ratios r_n,r_m,r_z")
    ap.add_argument("--pc", type=ints, default=6, help="coarse order: p or p_n,p_m,p_z")
    ap.add_argument("--Lam", type=float, nargs="+", default=[20.0])
    ap.add_argument("--N", type=str, default="vanishing", choices=["vanishing", "linear"],
                    help="density profile: 1 - psi^2 (vanishing) or 1 - psi/2 (linear)")
    ap.add_argument("--tau-slope", type=float, default=0.4, help="tau = 1 - slope psi")
    ap.add_argument("--no-cont", action="store_true", help="no continuation in Lam (coarse Newton)")
    ap.add_argument("--newton-tol", type=float, default=1e-10)
    ap.add_argument("--newton-maxit", type=int, default=30)
    ap.add_argument("--precond", type=str, default="rb", choices=["rb", "gmres", "none"])
    ap.add_argument("--tol", type=float, default=1e-10, help="outer tolerance on |R|_inf")
    ap.add_argument("--maxit", type=int, default=50, help="outer iterations")
    ap.add_argument("--eta", type=float, default=0.1, help="local tol = eta |R|")
    ap.add_argument("--local-tol-min", type=float, default=1e-13)
    ap.add_argument("--local-maxit", type=int, default=50)
    ap.add_argument("--rk0", type=int, default=100, help="OMS block compression rank (assembler)")
    ap.add_argument("--rk-start", type=int, default=50, help="red-black rank at stage 0")
    ap.add_argument("--rk-step", type=int, default=25, help="red-black rank increment per stage")
    ap.add_argument("--rk-max", type=int, default=None, help="cap on the red-black rank")
    ap.add_argument("--device", type=str, default="cpu", help="leaf operators")
    ap.add_argument("--compute-device", type=str, default="cuda", help="red-black solver")
    ap.add_argument("--check", action="store_true", help="check (I-K) of P against the local solves")
    ap.add_argument("--psi-max", type=float, default=0.8, help="flux-function fit on psi <= psi_max")
    ap.add_argument("--nprobe", type=int, default=200, help="probe points per patch")
    ap.add_argument("--est-samples", type=int, default=3, help="estimator: random points per leaf / leaf face")
    ap.add_argument("--est-node-samples", type=int, default=8, help="estimator check: collocation nodes per leaf")
    ap.add_argument("--no-estimate", action="store_true", help="skip the a-posteriori components")
    ap.add_argument("--est-check-points", type=int, default=16,
                    help="estimator check: face points per batch re-evaluated with the volume polynomial (0 = off)")
    ap.add_argument("--est-dump", action="store_true", help="save every eta sample (coords, psi, patch) to --out")
    ap.add_argument("--compare", type=str, nargs="*", default=[], help="probe .npz files of other runs")
    ap.add_argument("--out", type=str, default="results_test_periodic_vanishing_edge")
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
    beta = (arch_stretch.beta_for_ratio(args.arch_ratio, m) if args.arch_ratio is not None
            else args.arch_stretch)
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
    if beta != 0:
        print("arch radial stretch beta = %.4g\n  fine   %s\n  coarse %s"
              % (beta, arch_stretch.describe(beta, m), arch_stretch.describe(beta, m // rm)))
    print("rough memory: leaf factors %.1f GB total (+ MUMPS factors + HBS blocks)" % (Ns * mem_slab))

    pg = torus.param_geom(jax_avail=False, torch_avail=True, hpsalt=True)
    pdo_geo = transform_pdo(pg, None)                   # geometry only; c via set_c
    dev = torch.device(args.device)
    st_tag = "" if beta == 0 else "_st%.3g" % beta            # output names of stretched runs
    probes = diag.make_probes(torus.L, n_per_patch=args.nprobe)

    # ---- coarse geometry (once)
    t0 = time.perf_counter()
    geom_c = [[0.0, -torus.Rc, -torus.Rc], [torus.L, torus.Rc, torus.Rc]]
    cs_c = CondensedSlab(geom_c, pdo_geo, p_c,
                         dict(Rc=torus.Rc, s=torus.s, n=n // rn, m=m // rm, nz=nt_c,
                              flux="conormal", device=dev, periodic=True, stretch=beta),
                         fast=True, verbose=False)
    print("coarse geometry: %.2fs, N_skel %d (periodic: Ib = wall only, %d dofs)"
          % (time.perf_counter() - t0, len(cs_c.slab.Ii), len(cs_c.slab.Ib)))
    grid_c = cs_c.slab.grid

    # interface tree / assembler / rank strategy as in full_twisted_torus_oms.py
    max_level = int(np.log2(n ** 2)) - 2

    def make_tree(XXI):
        XXI = XXI.detach().cpu().numpy() if torch.is_tensor(XXI) else np.asarray(XXI)
        return circInterfaceTree(XXI[:, 1:3], Rc=torus.Rc, s=torus.s, n_core=n,
                                 m_arch=m, max_level=max_level, stretch=beta)
    leaf_size = n_iface // 2 ** max_level
    gb = lambda p: torus.gb(p, jax_avail=False, torch_avail=True)

    phi_c, prev_Lam = None, None
    for Lam in args.Lam:
        model = PBProfiles(torus, Lam=Lam, N=args.N, tau_slope=args.tau_slope)
        print("== %s" % model.describe())
        row = dict(Ns=Ns, H=H, n=n, m=m, nz=nz, p=str(triple(p_f)), nc=n // rn, mc=m // rm,
                   ntc=nt_c, pc=str(triple(p_c)), Lam=Lam, N=args.N, tau_slope=args.tau_slope,
                   precond=args.precond, arch_stretch=beta)

        # ---- 1. global coarse Newton.  phi = 0 and the solution at any smaller
        # Lam are supersolutions, from which Newton on this convex problem
        # decreases monotonically: warm start only when Lam increases.
        warm = phi_c is not None and Lam >= prev_Lam
        sched = lam_schedule(Lam, not args.no_cont)
        if warm:
            sched = [l for l in sched if l > prev_Lam * (1 + 1e-12)] or [Lam]
        phi_c, clog = coarse_newton_fast(model, cs_c, lam_schedule=sched, tol=args.newton_tol,
                                         maxit=args.newton_maxit, phi0=phi_c if warm else None)
        prev_Lam = Lam
        print("  coarse: %d Newton steps, %.2fs (set_c %.2fs, factor %.2fs, solve %.2fs per step); "
              "phi in [%.3f, %.3f]"
              % (clog["newton_steps"], clog["t_total"], clog["t_set_c_per_step"],
                 clog["t_factor_per_step"], clog["t_solve_per_step"], phi_c.min(), phi_c.max()))
        row.update(newton_steps=clog["newton_steps"], t_coarse=clog["t_total"],
                   phi_c_min=phi_c.min(), phi_c_max=phi_c.max())

        # ---- 2. fine OMS for -Lap + c_ref, c_ref from P phi_c per slab
        store_c, store_phi, info = {}, {}, {}

        def c_nodal(ops, geom):
            i = int(round(float(geom[0][0]) / H)) % Ns
            Pr = Prolongation(grid_c, ops.grid, t_offset=i * nz // 2, nt_global=nt_f)
            if i == 0:
                info["P_exact"] = check_prolongation(Pr)
            phf = Pr(phi_c)
            c = model.dg(ops.nodes(), phf)
            store_c[i], store_phi[i] = c, phf
            return c

        def bc(pts):
            pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
            return model.phi_b(pts)

        cyl_opts = dict(Rc=torus.Rc, s=torus.s, n=n, m=m, nz=nz, flux="conormal",
                        fast=True, c_nodal=c_nodal, keep_geometry=False, device=dev,
                        stretch=beta)
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
        it = InterfaceIteration(OMS, model, bc, store_c, store_phi)
        if args.check:
            print("  check: |(I-K) v - Stot v| / |Stot v| = %.2e (expect ~1e-12)" % it.check_jacobian(Stot))
        u0 = it.traces_of_fields()
        t0 = time.perf_counter()
        u, logs = it.run(P, u0, u_exact=None, tol=args.tol, maxit=args.maxit, eta=args.eta,
                         local_tol_min=args.local_tol_min, local_maxit=args.local_maxit)
        t_outer = time.perf_counter() - t0
        rates = [logs[k + 1]["residual"] / logs[k]["residual"] for k in range(len(logs) - 1)]
        print("  -> %d outer its, |R| %.2e, rates %s, |u - u0| %.2e, %.1fs"
              % (len(logs), logs[-1]["residual"], " ".join("%.1e" % r for r in rates),
                 np.max(np.abs(u - u0)), t_outer))
        row.update(outer_its=len(logs), residual=logs[-1]["residual"], t_outer=t_outer,
                   rate_last=rates[-1] if rates else np.nan,
                   local_steps_total=int(sum(l["local_steps_mean"] * Ns for l in logs)),
                   du_iface=np.max(np.abs(u - u0)))

        # ---- 4. diagnostics (no exact solution)
        S = it.slabs
        d_cf = max(np.max(np.abs(s.phi - store_phi[i])) for i, s in enumerate(S))
        ov, ov_pairs, unmatched = diag.overlap_mismatch(
            [s.W for s in S], [s.phi for s in S], [s.I for s in S],
            [s.ops.zbounds for s in S], torus.L)
        psi_all = np.concatenate([s.psi[s.I] for s in S])
        phi_all = np.concatenate([s.phi[s.I] for s in S])
        res_ff, F, _ = diag.flux_function_fit(psi_all, phi_all, psi_max=args.psi_max)
        pv = diag.eval_probes_slabs(probes, [(s.ops, s.phi) for s in S], H, torus.L)
        pv_c = diag.eval_probes_global(probes, cs_c.slab, phi_c, torus.L)
        print("  phi in [%.4f, %.4f]; coarse vs fine: nodes %.2e, probes %.2e"
              % (phi_all.min(), phi_all.max(), d_cf, np.max(np.abs(pv - pv_c))))
        print("  overlap consistency: max %.2e over %d pairs (%d unmatched nodes)"
              % (ov, Ns, unmatched))
        print("  flux function on psi <= %.2f: max |phi - F(psi)| %.2e (phi range %.2e)"
              % (args.psi_max, res_ff, np.ptp(phi_all)))
        print("  F(psi) at psi = %s:\n    %s"
              % (" ".join("%.2f" % x for x in np.linspace(0, args.psi_max, 9)),
                 " ".join("%.5f" % x for x in F)))
        if not args.no_estimate:
            t0 = time.perf_counter()
            est = ResidualEstimator(pdo_geo, model, torus.Rc, nsamp=args.est_samples,
                                    node_samples=args.est_node_samples, dump=args.est_dump,
                                    check_points=args.est_check_points)
            comp = None
            for i, s in enumerate(S):
                z0 = float(s.ops.zbounds[0])
                comp = ResidualEstimator.merge(
                    comp, est.slab(s.ops, s.phi, (z0 + 0.5 * H, z0 + 1.5 * H), seed=i))
            print("  a-posteriori components (off-node max; %d interior pts, %d leaf-face, "
                  "%d patch-face, %d wall)  [%.1fs]"
                  % (comp["n_eta"], comp["n_face_leaf"], comp["n_face_patch"], comp["n_wall"],
                     time.perf_counter() - t0))
            sections = (("interior", (("eta", "|eta|"),)),
                        ("leaf faces (both leaves in one patch)",
                         (("jump_flux_leaf", "flux jump"), ("jump_val_leaf", "value jump"))),
                        ("patch faces (leaves in two patches)",
                         (("jump_flux_patch", "flux jump"), ("jump_val_patch", "value jump"))),
                        ("wall", (("wall", "|phi - phi_b|"),)),
                        ("checks", (("eta_nodes", "eta at nodes"), ("wall_gauss", "wall at Gauss nodes"))))
            for head, items in sections:
                print("    %s" % head)
                for key, lab in items:
                    print("      %-20s %.2e  at %s" % (lab, comp[key], format_loc(comp["loc"].get(key))))
            print("      patch face map, max physical mismatch %.2e (expect ~1e-15; >1e-9 raises)"
                  % comp["face_map_err"])
            print("      face-wise vs volume evaluation, max rel. difference %.2e (expect ~1e-12)"
                  % comp["face_eval_err"])
            print("    time: interior + node check %.1fs, leaf faces %.1fs, patch faces %.1fs, wall %.1fs"
                  % tuple(comp[t] for t in TIMERS))
            row.update({"est_" + key: comp[key] for key in COMPONENTS})
            row.update({"est_loc_" + key: format_loc(comp["loc"].get(key)) for key in COMPONENTS})
            row["est_face_map_err"] = comp["face_map_err"]
            row["est_face_eval_err"] = comp["face_eval_err"]
            row.update({"est_" + t: comp[t] for t in TIMERS})
            if args.est_dump:
                dpath = os.path.join(args.out, "eta_Ns%d_nz%d_n%d_m%d_p%s_Lam%g_%s%s.npz" % (
                    Ns, nz, n, m, "-".join(map(str, triple(p_f))), Lam, args.N, st_tag))
                save_dump(dpath, comp)
                print("    eta samples saved to", dpath)
        row.update(phi_min=phi_all.min(), phi_max=phi_all.max(), coarse_vs_fine=d_cf,
                   coarse_vs_fine_probes=np.max(np.abs(pv - pv_c)), overlap=ov,
                   overlap_unmatched=unmatched, flux_dev=res_ff, F_axis=F[0])

        tag = "Ns%d_nz%d_n%d_m%d_p%s_Lam%g_%s%s" % (Ns, nz, n, m, "-".join(map(str, triple(p_f))),
                                                    Lam, args.N, st_tag)
        npz = os.path.join(args.out, "probes_%s.npz" % tag)
        np.savez(npz, values=pv, coarse=pv_c, **probes,
                 meta_Ns=Ns, meta_nz=nz, meta_n=n, meta_m=m, meta_p=np.array(triple(p_f)),
                 meta_Lam=Lam, meta_N=args.N, meta_tau_slope=args.tau_slope, meta_stretch=beta)
        print("  probes saved to", npz)
        for path in args.compare:
            try:
                d_abs, d_rel, meta = diag.compare_probes(path, probes, pv)
            except (OSError, ValueError) as e:
                print("  compare %s: %s" % (path, e))
                continue
            print("  vs %s: max |diff| %.2e (rel %.2e)" % (os.path.basename(path), d_abs, d_rel))

        write_csv(os.path.join(args.out, "test_periodic_vanishing_edge.csv"), [row])
        write_csv(os.path.join(args.out, "test_periodic_vanishing_edge_outer.csv"),
                  [dict(Ns=Ns, p=str(triple(p_f)), Lam=Lam, N=args.N, precond=args.precond, **l)
                   for l in logs])
        OMS.close()
        del OMS, rb, it


if __name__ == "__main__":
    main()
