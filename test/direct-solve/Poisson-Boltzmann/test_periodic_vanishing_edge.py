"""
test_periodic_vanishing_edge: Approach 1 on the full periodic torus (twisted
elliptic torus, or a stellarator with --input), with a comparison of solver
variants.

DEFAULT (--problem manufactured): manufactured solution of limited regularity
(pb.manufactured.PBPedestal): phi* = F(psi) + eps (1 - psi) Re[w^m e^{-i n zeta}],
F a pedestal at the flux surface psi0: tanh of width --delta (default; analytic)
or --mms-profile kink (C^{ceil(beta)-1}), load f* = -Lap phi* +
g(phi*), Dirichlet data phi*.  Errors vs phi* are exact (interface and volume);
the flux-function fit and the a-posteriori estimator are skipped.

Variants (--variants, default both), after the baseline (= the pipeline below):
  schwarz : P = I on the SAME OMS, c_ref and start (P phi_c): classical
            nonlinear overlapping Schwarz (additive), --schwarz-maxit;
  dh      : Debye-Hueckel linearization c_ref = d_s g(psi, 0) = Lam^2 N / tau for
            both the local chord solves and the preconditioner (its own OMS and
            compressed red-black), start phi = 0, no coarse problem;
            --dh-maxit, --dh-local-maxit.
Per variant: outer its, |R|, rates (last, asymptotic = geometric mean of the last
5), local chord steps, setup/outer times, c_ref mismatch max|c_ref - c(phi)| /
max|c(phi)| at the final phi, probe difference to the baseline, and (manufactured)
the exact errors.  One CSV row per variant (column 'variant').

--problem physical: the PHYSICAL problem (no manufactured solution):

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

Stellarator geometry (geometry/geom_3D/stellerator.py) with --input, e.g. the
CSX equilibrium --input wout_csx.nc (VMEC flux coordinates: psi = normalized
toroidal flux).  With --input the profile defaults change to
    N = 1 - psi ('vanishing-linear'),  tau = 1,  Lam^2 = 70 per minor radius^2
(--lam-unit minor: code Lam = Lam / a, a = VMEC Aminor_p, else Rc); any of
--N / --tau-slope / --Lam / --lam-unit given explicitly overrides them.

Run from this folder with the repo root on PYTHONPATH, e.g.
    PYTHONPATH=../../.. python test_periodic_vanishing_edge.py --Ns 32 --nz 2 --nm 8,2 --p 8 --coarsen 2,2,1 --pc 6 --Lam 20
  radially clustered arch leaves (graded toward the wall, ratio 0.5 per leaf):
    ... --nm 16,4 --arch-ratio 0.5        (or --arch-stretch <beta>)
  CSX (defaults N = 1 - psi, tau = 1, Lam^2 = 70 / a^2), manufactured + both variants:
    PYTHONPATH=../../.. python test_periodic_vanishing_edge.py --input wout_csx.nc --Ns 64 --nz 2 --nm 8,2 --p 8
  physical problem, baseline only:
    ... --problem physical --variants none
"""
import argparse
import csv
import os
import time
from types import SimpleNamespace

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

from pb.manufactured import transform_pdo, PBPedestal
from pb.profiles import PBProfiles, NKINDS
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
    ap.add_argument("--input", type=str, default=None,
                    help="stellarator geometry (VMEC input or wout_*.nc, e.g. wout_csx.nc) "
                         "instead of the twisted elliptic torus")
    ap.add_argument("--Lam", type=float, nargs="+", default=None,
                    help="Lam values (default 20; with --input sqrt(70))")
    ap.add_argument("--lam-unit", type=str, default=None, choices=["geom", "minor"],
                    help="Lam per geometry length unit (geom, default) or per minor radius "
                         "(minor: code Lam = Lam / a; default with --input)")
    ap.add_argument("--N", type=str, default=None, choices=list(NKINDS),
                    help="density profile: 1 - psi^2 (vanishing, default), 1 - psi "
                         "(vanishing-linear, default with --input) or 1 - psi/2 (linear)")
    ap.add_argument("--tau-slope", type=float, default=None,
                    help="tau = 1 - slope psi (default 0.4; with --input 0, i.e. tau = 1)")
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
    ap.add_argument("--problem", type=str, default="manufactured", choices=["manufactured", "physical"],
                    help="manufactured (default: PBPedestal phi*, exact errors) or physical (f = 0, phi_b = 0)")
    ap.add_argument("--mms-profile", type=str, default="tanh", choices=["tanh", "kink"],
                    help="pedestal H(psi): tanh (analytic, width --delta) or kink (C^ceil(beta)-1)")
    ap.add_argument("--delta", type=float, default=0.5, help="tanh pedestal width in psi")
    ap.add_argument("--mms-amp", type=float, default=3.0, help="phi* = amp [-(1-psi) + kink H(psi)] + ...")
    ap.add_argument("--mms-kink", type=float, default=0.5, help="pedestal height / amp (0: no pedestal)")
    ap.add_argument("--mms-beta", type=float, default=2.5, help="kink profile: H = q^beta (>= 2)")
    ap.add_argument("--mms-psi0", type=float, default=0.7, help="flux surface of the pedestal")
    ap.add_argument("--mms-eps", type=float, default=0.2, help="3D term eps (1-psi) Re[w^m e^{-i n zeta}]")
    ap.add_argument("--mms-mode", type=str, default="2,1", help="m,n of the 3D term (n per field period)")
    ap.add_argument("--variants", type=str, default="schwarz,dh",
                    help="comparisons after the baseline: comma list of schwarz (P = I, same c_ref/start), "
                         "dh (Debye-Hueckel c_ref = d_s g(0), start 0, its own red-black), or none")
    ap.add_argument("--schwarz-maxit", type=int, default=300, help="outer iterations of the P = I variant")
    ap.add_argument("--dh-maxit", type=int, default=100, help="outer iterations of the Debye-Hueckel variant")
    ap.add_argument("--dh-local-maxit", type=int, default=200, help="local chord steps of the Debye-Hueckel variant")
    ap.add_argument("--quiet-outer", action="store_true", help="no per-iteration lines (summaries only)")
    ap.add_argument("--out", type=str, default="results_test_periodic_vanishing_edge")
    args = ap.parse_args()
    v = args.variants.strip().lower()
    args.variants = [] if v in ("", "none") else [x.strip() for x in v.split(",")]
    bad = [x for x in args.variants if x not in ("schwarz", "dh")]
    if bad:
        ap.error("unknown variant(s) %s (choose from schwarz, dh, none)" % bad)
    try:
        args.mms_mode = tuple(int(x) for x in args.mms_mode.split(","))
        assert len(args.mms_mode) == 2 and args.mms_mode[0] >= 0
    except (ValueError, AssertionError):
        ap.error("--mms-mode must be m,n with m >= 0, got %r" % args.mms_mode)
    stell = args.input is not None                   # profile defaults per geometry
    if args.Lam is None:
        args.Lam = [float(np.sqrt(70.0))] if stell else [20.0]
    if args.lam_unit is None:
        args.lam_unit = "minor" if stell else "geom"
    if args.N is None:
        args.N = "vanishing-linear" if stell else "vanishing"
    if args.tau_slope is None:
        args.tau_slope = 0.0 if stell else 0.4
    return args


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
    global torus
    args = parse()
    geo_tag = ""
    if args.input is not None:                       # stellarator (e.g. CSX) instead of the twisted torus
        import geometry.geom_3D.stellerator as stell
        stell.configure(args.input)
        stell.s = 0.55
        torus = stell
        geo_tag = "_" + os.path.splitext(os.path.basename(args.input))[0]
        if args.out == "results_test_periodic_vanishing_edge":
            args.out += geo_tag
    a_ref = 1.0
    if args.lam_unit == "minor":
        a_ref = getattr(getattr(torus, "_geom", None), "aminor", None) or torus.Rc
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
    print("%s: L = %.4f, Ns = %d, H = %.4f, s = %.2f, Rc = %.4f"
          % ("stellarator " + args.input if args.input else "twisted torus", torus.L, Ns, H, torus.s, torus.Rc))
    if args.lam_unit == "minor":
        print("Lam given per minor radius a = %.4f: code Lam = Lam / a" % a_ref)
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
    kw = {} if args.rk_max is None else dict(rk_max=args.rk_max)
    strat = rkStrat_new.linear(args.rk_start, args.rk_step, skip_first_level=False, **kw)
    ranks = [int(strat.rank(st)) for st in range(int(np.log2(Ns)))]
    print("red-black ranks per stage %s (HBS leaf size %d); variants: %s"
          % (ranks, leaf_size, ", ".join(["coarse"] + args.variants)))

    mms_cache = None                                     # -Lap phi* per point set, shared over Lam
    phi_c, prev_Lam = None, None
    for Lam_in in args.Lam:
        Lam = Lam_in / a_ref                             # Lam of the equation in geometry units
        if args.problem == "manufactured":
            model = PBPedestal(torus, pdo_geo, Lam=Lam, N=args.N, tau_slope=args.tau_slope,
                               amp=args.mms_amp, kink=args.mms_kink, beta=args.mms_beta,
                               psi0=args.mms_psi0, eps=args.mms_eps, mode=args.mms_mode,
                               cache=mms_cache, cache_size=Ns + 8, profile=args.mms_profile,
                               delta=args.delta)
            mms_cache = model._cache
        else:
            model = PBProfiles(torus, Lam=Lam, N=args.N, tau_slope=args.tau_slope)
        mms = args.problem == "manufactured"
        print("== %s%s" % (model.describe(), "" if a_ref == 1.0 else
                           "  (Lam a = %g, (Lam a)^2 = %g)" % (Lam_in, Lam_in ** 2)))
        base_row = dict(problem=args.problem, geometry=args.input or "twisted_torus", Ns=Ns, H=H, n=n, m=m,
                        nz=nz, p=str(triple(p_f)), nc=n // rn, mc=m // rm, ntc=nt_c, pc=str(triple(p_c)),
                        Lam=Lam_in, lam_unit=args.lam_unit, Lam_code=Lam, N=args.N,
                        tau_slope=args.tau_slope, arch_stretch=beta, tol=args.tol, eta=args.eta)
        if mms:
            base_row.update(mms_profile=args.mms_profile, mms_delta=args.delta, mms_amp=args.mms_amp, mms_kink=args.mms_kink, mms_beta=args.mms_beta,
                            mms_psi0=args.mms_psi0, mms_eps=args.mms_eps, mms_mode="%d,%d" % args.mms_mode)
        rows, outer_rows = [], []

        def bc(pts):
            pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
            return model.phi_b(pts)

        # ---- 1. global coarse Newton.  phi = 0 and the solution at any smaller
        # Lam are supersolutions of the physical problem, from which Newton on this
        # convex problem decreases monotonically: warm start only when Lam increases.
        warm = phi_c is not None and Lam >= prev_Lam
        sched = lam_schedule(Lam, not args.no_cont)
        if warm:
            sched = [l for l in sched if l > prev_Lam * (1 + 1e-12)] or [Lam]
        phi_c, clog = coarse_newton_fast(model, cs_c, lam_schedule=sched, tol=args.newton_tol,
                                         maxit=args.newton_maxit, phi0=phi_c if warm else None)
        prev_Lam = Lam
        err_c = np.max(np.abs(phi_c - model.phistar(grid_c.nodes))) if mms else np.nan
        print("  coarse: %d Newton steps, %.2fs (set_c %.2fs, factor %.2fs, solve %.2fs per step); "
              "phi in [%.3f, %.3f]%s"
              % (clog["newton_steps"], clog["t_total"], clog["t_set_c_per_step"],
                 clog["t_factor_per_step"], clog["t_solve_per_step"], phi_c.min(), phi_c.max(),
                 "; err vs phi* %.2e" % err_c if mms else ""))

        # ---- fine OMS + red-black for -Lap + c_ref; slab_fields(i, ops, info) -> (c_ref, phi0)
        def build_fine(slab_fields, label):
            store_c, store_phi, info = {}, {}, {}

            def c_nodal(ops, geom):
                i = int(round(float(geom[0][0]) / H)) % Ns
                c, phi0 = slab_fields(i, ops, info)
                store_c[i], store_phi[i] = c, phi0
                return c

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
            t0 = clock(args.compute_device)
            rb = omsdirectHBS_new.RedBlackSolverHBS(
                strat, OMS, tree=make_tree(OMS.interface_points), quad=False,
                compute_device=args.compute_device, seed=None, dbg=0, normalized=False,
                constant_blocks=False, operator="hbs", cyclic=True)
            rb.factorize()
            t_rb = clock(args.compute_device) - t0
            print("  [%s] fine OMS built in %.1fs (Ntot %d), red-black setup + factor %.1fs"
                  % (label, t_oms, OMS.Ntot, t_rb))
            return SimpleNamespace(OMS=OMS, Stot=Stot, rb=rb, store_c=store_c, store_phi=store_phi,
                                   info=info, t_oms=t_oms, t_rb=t_rb)

        # ---- outer iteration on a built OMS, with summary
        def run_outer(fine, P, label, maxit, local_maxit):
            it = InterfaceIteration(fine.OMS, model, bc, fine.store_c, fine.store_phi)
            u_exact = fine.OMS.interface_values(lambda pts: model.phistar(np.asarray(pts))) if mms else None
            u0 = it.traces_of_fields()
            print("  [%s] outer iteration (maxit %d, local maxit %d)%s"
                  % (label, maxit, local_maxit,
                     ": u0 err_iface %.2e" % np.max(np.abs(u0 - u_exact)) if mms else ""))
            t0 = time.perf_counter()
            u, logs = it.run(P, u0, u_exact=u_exact, tol=args.tol, maxit=maxit, eta=args.eta,
                             local_tol_min=args.local_tol_min, local_maxit=local_maxit,
                             verbose=not args.quiet_outer)
            t_outer = time.perf_counter() - t0
            res = np.array([l["residual"] for l in logs])
            rates = res[1:] / res[:-1]
            k5 = min(5, len(rates))
            rate_asym = float(np.prod(rates[-k5:]) ** (1.0 / k5)) if k5 else np.nan
            conv = bool(res[-1] < args.tol)
            if conv:
                its_tol = float(len(logs))
            elif 0 < rate_asym < 1:
                its_tol = len(logs) + float(np.log(args.tol / res[-1]) / np.log(rate_asym))
            else:
                its_tol = np.inf
            smax = max(l["local_steps_max"] for l in logs)
            S_ = it.slabs
            cmis = max(np.max(np.abs(s.c - model.dg(s.W, s.phi, s.psi))) for s in S_) \
                / max(np.max(np.abs(model.dg(s.W, s.phi, s.psi))) for s in S_)
            out = dict(outer_its=len(logs), converged=conv, residual=res[-1],
                       rate_last=rates[-1] if len(rates) else np.nan, rate_asym=rate_asym,
                       its_to_tol=its_tol, local_steps_total=int(round(sum(l["local_steps_mean"] * Ns for l in logs))),
                       local_steps_max=smax, local_capped=bool(smax >= local_maxit),
                       t_outer=t_outer, t_local=sum(l["t_local"] for l in logs),
                       t_precond=sum(l.get("t_precond", 0.0) for l in logs),
                       du_iface=np.max(np.abs(u - u0)), c_mismatch=cmis)
            if mms:
                ev = it.volume_errors()
                out.update(err_iface=np.max(np.abs(u - u_exact)), err_iface_u0=np.max(np.abs(u0 - u_exact)),
                           err_volume=ev.max())
            print("  [%s] -> %d outer its (%s), |R| %.2e, rate last %.1e asym %.1e%s; local steps %d "
                  "(max %d%s); c_ref mismatch %.1e; %.1fs%s"
                  % (label, len(logs), "converged" if conv else "NOT converged", res[-1], out["rate_last"],
                     rate_asym, "" if conv else ", ~%.0f its to tol" % its_tol, out["local_steps_total"],
                     smax, ", CAPPED" if out["local_capped"] else "", cmis, t_outer,
                     "; err iface %.2e, volume %.2e" % (out["err_iface"], out["err_volume"]) if mms else ""))
            outer_rows.extend(dict(variant=label, Ns=Ns, p=str(triple(p_f)), Lam=Lam_in, N=args.N,
                                   problem=args.problem, **l) for l in logs)
            return it, u, out

        # ---- 2. baseline: c_ref = d_s g(P phi_c), start from P phi_c, P = red-black (--precond)
        def fields_coarse(i, ops, info):
            Pr = Prolongation(grid_c, ops.grid, t_offset=i * nz // 2, nt_global=nt_f)
            if i == 0:
                info["P_exact"] = check_prolongation(Pr)
            phf = Pr(phi_c)
            return model.dg(ops.nodes(), phf), phf

        base = build_fine(fields_coarse, "coarse")
        if args.precond == "rb":
            P = base.rb.solve
        elif args.precond == "none":
            P = lambda v: v
        else:
            M = LinearOperator(base.Stot.shape, matvec=base.rb.solve, dtype=np.float64)

            def P(v):
                x, _ = gmres(base.Stot, v, M=M, restart=50, maxiter=4, **GMRES_TOL)
                return x
        if args.check:
            it_chk = InterfaceIteration(base.OMS, model, bc, base.store_c, base.store_phi)
            print("  check: |(I-K) v - Stot v| / |Stot v| = %.2e (expect ~1e-12)" % it_chk.check_jacobian(base.Stot))
            del it_chk
        it, u, summ = run_outer(base, P, "coarse", args.maxit, args.local_maxit)
        rows.append(dict(base_row, variant="coarse", precond=args.precond, newton_steps=clog["newton_steps"],
                         t_coarse=clog["t_total"], err_coarse=err_c, phi_c_min=phi_c.min(),
                         phi_c_max=phi_c.max(), t_oms=base.t_oms, t_rb=base.t_rb, Ntot=base.OMS.Ntot,
                         P_exact=base.info.get("P_exact"), rk_asm=args.rk0, rk_start=args.rk_start,
                         rk_step=args.rk_step, rk_max=args.rk_max, rk_stages=str(ranks), **summ))
        row = rows[-1]

        # ---- 3. diagnostics of the baseline
        S = it.slabs
        d_cf = max(np.max(np.abs(s.phi - base.store_phi[i])) for i, s in enumerate(S))
        ov, ov_pairs, unmatched = diag.overlap_mismatch(
            [s.W for s in S], [s.phi for s in S], [s.I for s in S],
            [s.ops.zbounds for s in S], torus.L)
        phi_all = np.concatenate([s.phi[s.I] for s in S])
        pv = diag.eval_probes_slabs(probes, [(s.ops, s.phi) for s in S], H, torus.L)
        pv_c = diag.eval_probes_global(probes, cs_c.slab, phi_c, torus.L)
        print("  phi in [%.4f, %.4f]; coarse vs fine: nodes %.2e, probes %.2e"
              % (phi_all.min(), phi_all.max(), d_cf, np.max(np.abs(pv - pv_c))))
        print("  overlap consistency: max %.2e over %d pairs (%d unmatched nodes)"
              % (ov, Ns, unmatched))
        row.update(phi_min=phi_all.min(), phi_max=phi_all.max(), coarse_vs_fine=d_cf,
                   coarse_vs_fine_probes=np.max(np.abs(pv - pv_c)), overlap=ov,
                   overlap_unmatched=unmatched)
        if not mms:
            psi_all = np.concatenate([s.psi[s.I] for s in S])
            res_ff, F, _ = diag.flux_function_fit(psi_all, phi_all, psi_max=args.psi_max)
            print("  flux function on psi <= %.2f: max |phi - F(psi)| %.2e (phi range %.2e)"
                  % (args.psi_max, res_ff, np.ptp(phi_all)))
            print("  F(psi) at psi = %s:\n    %s"
                  % (" ".join("%.2f" % x for x in np.linspace(0, args.psi_max, 9)),
                     " ".join("%.5f" % x for x in F)))
            row.update(flux_dev=res_ff, F_axis=F[0])
        if not mms and not args.no_estimate:
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
                dpath = os.path.join(args.out, "eta_Ns%d_nz%d_n%d_m%d_p%s_Lam%g_%s%s%s.npz" % (
                    Ns, nz, n, m, "-".join(map(str, triple(p_f))), Lam_in, args.N, st_tag, geo_tag))
                save_dump(dpath, comp)
                print("    eta samples saved to", dpath)

        tag = "Ns%d_nz%d_n%d_m%d_p%s_Lam%g_%s%s%s%s" % (Ns, nz, n, m, "-".join(map(str, triple(p_f))),
                                                        Lam_in, args.N, st_tag, geo_tag,
                                                        ("_mms-%s%s" % (args.mms_profile, "%g" % args.delta
                                                         if args.mms_profile == "tanh" else "%g" % args.mms_beta))
                                                        if mms else "")
        npz = os.path.join(args.out, "probes_%s.npz" % tag)
        np.savez(npz, values=pv, coarse=pv_c, **probes,
                 meta_Ns=Ns, meta_nz=nz, meta_n=n, meta_m=m, meta_p=np.array(triple(p_f)),
                 meta_Lam=Lam_in, meta_Lam_code=Lam, meta_N=args.N, meta_tau_slope=args.tau_slope,
                 meta_stretch=beta, meta_problem=args.problem)
        print("  probes saved to", npz)
        for path in args.compare:
            try:
                d_abs, d_rel, meta = diag.compare_probes(path, probes, pv)
            except (OSError, ValueError) as e:
                print("  compare %s: %s" % (path, e))
                continue
            print("  vs %s: max |diff| %.2e (rel %.2e)" % (os.path.basename(path), d_abs, d_rel))
        del it

        def probe_diff(it_v):
            pv_v = diag.eval_probes_slabs(probes, [(s.ops, s.phi) for s in it_v.slabs], H, torus.L)
            return np.max(np.abs(pv_v - pv))

        # ---- 4. variant (1): P = I on the SAME OMS / c_ref / start (nonlinear overlapping Schwarz)
        if "schwarz" in args.variants:
            it_s, _, summ = run_outer(base, lambda v: v, "schwarz", args.schwarz_maxit, args.local_maxit)
            summ["probes_vs_coarse"] = probe_diff(it_s)
            print("  [schwarz] probes vs baseline: max |diff| %.2e" % summ["probes_vs_coarse"])
            rows.append(dict(base_row, variant="schwarz", precond="none", t_oms=0.0, t_rb=0.0, **summ))
            del it_s
        base.OMS.close()
        del base, P

        # ---- 5. variant (2): Debye-Hueckel c_ref = d_s g(phi = 0), start phi = 0, P = its red-black
        if "dh" in args.variants:
            def fields_dh(i, ops, info):
                W = ops.nodes()
                z = np.zeros(len(W))
                return model.dg(W, z), z

            dh = build_fine(fields_dh, "dh")
            it_d, _, summ = run_outer(dh, dh.rb.solve, "dh", args.dh_maxit, args.dh_local_maxit)
            summ["probes_vs_coarse"] = probe_diff(it_d)
            print("  [dh] probes vs baseline: max |diff| %.2e" % summ["probes_vs_coarse"])
            rows.append(dict(base_row, variant="dh", precond="rb", t_oms=dh.t_oms, t_rb=dh.t_rb,
                             Ntot=dh.OMS.Ntot, **summ))
            dh.OMS.close()
            del dh, it_d

        # ---- summary table
        print("  summary (Lam = %g):" % Lam_in)
        print("    %-8s %6s %10s %9s %9s %8s %10s %9s%s"
              % ("variant", "its", "|R|", "rate", "steps", "setup s", "outer s", "c mism.",
                 "  err iface  err vol" if mms else ""))
        for r in rows:
            setup = r.get("t_oms", 0.0) + r.get("t_rb", 0.0) + r.get("t_coarse", 0.0)
            print("    %-8s %6d %10.2e %9.1e %9d %8.1f %10.1f %9.1e%s"
                  % (r["variant"], r["outer_its"], r["residual"], r["rate_asym"], r["local_steps_total"],
                     setup, r["t_outer"], r["c_mismatch"],
                     "  %9.2e %8.2e" % (r["err_iface"], r["err_volume"]) if mms else ""))

        write_csv(os.path.join(args.out, "test_periodic_vanishing_edge.csv"), rows)
        write_csv(os.path.join(args.out, "test_periodic_vanishing_edge_outer.csv"), outer_rows)


if __name__ == "__main__":
    main()
