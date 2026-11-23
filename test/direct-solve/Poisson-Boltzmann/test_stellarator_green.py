"""
test_stellarator_green: Laplace / Helmholtz on the stellarator geometry
(geometry/geom_3D/stellerator.py, default W7-X from input.w7x) with an exact
solution: a superposition of free-space Green's functions of point sources
OUTSIDE the domain,

    u(z) = sum_j w_j G(|z - c_j|),   G(r) = cos(kh r) / (4 pi r)   (kh = 0: 1/(4 pi r)),

which solves -Lap u - kh^2 u = 0 inside.  Its trace is the Dirichlet data; the
error is measured on all OMS interface nodes (interior cross-sections spread
over the whole torus).  This checks the map, the forward-map PDO transform
(StellaratorGeometry.transform_helmholtz_pdo) and the periodic OMS pipeline.

Sources: random boundary points (t, theta), pushed outward along the physical
normal by a distance d ~ U(dmin, dmax) (in units of Rc), weights w_j ~ U(-1, 1);
each is accepted only if it is outside the domain and at least dmin Rc away
from a dense sample of the boundary (so "not too close" holds globally, not
only along its own normal).

Run from this folder with the repo root on PYTHONPATH, e.g.
    PYTHONPATH=../../.. python test_stellarator_green.py                      # Laplace, W7-X
    PYTHONPATH=../../.. python test_stellarator_green.py --kh 6 --Ns 128      # Helmholtz
    ... --toy | --input other.vmec | --mmax 4 --nmax 4   (geometry selection)
"""
import argparse
import time

import numpy as np
import scipy
import torch
from packaging.version import Version
from scipy.sparse.linalg import LinearOperator, gmres

import direct_solve.omsdirectsolveHBS_torch_new2 as omsdirectHBS
from direct_solve.omsdirectsolveHBS_torch_new2 import rkStrat
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.solver as solverWrap
import geometry.geom_3D.stellerator as torus
from circInterfaceTree import circInterfaceTree


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kh", type=float, default=0.0, help="wavenumber (0: Laplace)")
    ap.add_argument("--nsrc", type=int, default=6, help="number of exterior point sources")
    ap.add_argument("--dmin", type=float, default=0.5, help="min source distance to the boundary / Rc")
    ap.add_argument("--dmax", type=float, default=1.5, help="max source distance (along the normal) / Rc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Ns", type=int, default=128, help="number of slabs (power of 2, cyclic red-black)")
    ap.add_argument("--nz", type=int, default=2, help="axial leaves per double slab")
    ap.add_argument("--nm", type=str, default="16,4", help="n_core,m_arch cross-section leaves (n = 4 m)")
    ap.add_argument("--p", type=int, default=8, help="hpsmultidomain order")
    ap.add_argument("--s", type=float, default=0.55, help="core half-width / Rc")
    ap.add_argument("--rk0", type=int, default=100, help="OMS block compression rank")
    ap.add_argument("--rk-step", type=int, default=50, help="red-black rank increment per stage")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="red-black compute device")
    ap.add_argument("--tol", type=float, default=1e-10, help="GMRES relative tolerance")
    ap.add_argument("--input", type=str, default=None, help="VMEC input (default: input.w7x)")
    ap.add_argument("--toy", action="store_true", help="toy W7-X-like boundary instead")
    ap.add_argument("--mmax", type=int, default=None, help="truncate boundary spectrum: m <= mmax")
    ap.add_argument("--nmax", type=int, default=None, help="truncate boundary spectrum: |n| <= nmax")
    ap.add_argument("--check-geom", action="store_true",
                    help="run stellerator.check_param first (det J, map and PDO checks)")
    return ap.parse_args()


# ---------------------------------------------------------------- sources
def boundary_points(geo, nt, nth):
    t = np.linspace(0, geo.L, nt, endpoint=False)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    T, TH = np.meshgrid(t, th, indexing="ij")
    p = np.c_[T.ravel(), geo.Rc * np.cos(TH).ravel(), geo.Rc * np.sin(TH).ravel()]
    return np.stack(geo.forward(p), -1)


def outward_normal(geo, p):
    """Physical unit normal at reference wall points p: grad(x^2 + y^2) = 2 x grad y2
    + 2 y grad y3, with grad y_k the rows of J^{-1}."""
    _, J, _ = geo.jet(p)
    Jm = np.stack([np.stack(row, -1) for row in J], -2)          # (n, 3, 3), J[a, b] = dz_a/dp_b
    G = np.linalg.inv(Jm)                                           # G[k, i] = dy_k/dq_i
    nvec = p[:, 1:2] * G[:, 1, :] + p[:, 2:3] * G[:, 2, :]
    return nvec / np.linalg.norm(nvec, axis=1, keepdims=True)


def inside(geo, c, nth=1440):
    """Point-in-polygon test of physical point c in the boundary cross-section
    at its own toroidal angle (exact up to the polygon resolution; no Newton)."""
    ang = np.mod(np.arctan2(c[1], c[0]), 2 * np.pi)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    p = np.c_[np.full(nth, geo.L * ang / (2 * np.pi)), geo.Rc * np.cos(th), geo.Rc * np.sin(th)]
    R, Z = geo.fields(p, order=0)
    R, Z = R[""], Z[""]
    r0, z0 = np.hypot(c[0], c[1]), c[2]
    R2, Z2 = np.roll(R, -1), np.roll(Z, -1)
    cross = ((Z > z0) != (Z2 > z0)) & (r0 < R + (z0 - Z) * (R2 - R) / (Z2 - Z + 1e-300))
    return bool(np.sum(cross) % 2)


def place_sources(geo, nsrc, dmin, dmax, rng, max_tries=10000):
    wall = boundary_points(geo, 1200, 96)
    from scipy.spatial import cKDTree
    tree = cKDTree(wall)
    src = []
    for _ in range(max_tries):
        if len(src) == nsrc:
            break
        t, th = rng.uniform(0, geo.L), rng.uniform(0, 2 * np.pi)
        p = np.array([[t, geo.Rc * np.cos(th), geo.Rc * np.sin(th)]])
        zb = np.stack(geo.forward(p), -1)[0]
        c = zb + rng.uniform(dmin, dmax) * geo.Rc * outward_normal(geo, p)[0]
        if tree.query(c)[0] < dmin * geo.Rc:
            continue                                                # too close to another part
        if inside(geo, c):
            continue
        src.append(c)
    if len(src) < nsrc:
        raise RuntimeError("could only place %d of %d sources" % (len(src), nsrc))
    return np.array(src), tree


def main():
    args = parse()
    torus.configure("toy" if args.toy else args.input, mmax=args.mmax, nmax=args.nmax)
    torus.s = args.s
    geo = torus._geom
    n_core, m_arch = (int(v) for v in args.nm.split(","))
    Ns, kh, p = args.Ns, args.kh, args.p
    assert Ns >= 4 and not Ns & (Ns - 1), "cyclic red-black needs Ns = 2^k slabs"
    print("geometry: nfp = %d, R00 = %.4f, Rc = %.4f, L = %.4f, %d boundary modes, s = %.2f"
          % (geo.nfp, geo.R00, geo.Rc, geo.L, len(geo.modes), torus.s))
    if args.check_geom:
        chk = torus.check_param()
        if chk["det_min"] * chk["det_max"] <= 0:
            raise RuntimeError("the map folds (det J changes sign); not solving")

    # ---- exact solution
    rng = np.random.default_rng(args.seed)
    SRC, wtree = place_sources(geo, args.nsrc, args.dmin, args.dmax, rng)
    W = rng.uniform(-1, 1, args.nsrc)
    dist = wtree.query(SRC)[0] / geo.Rc
    print("%s, kh = %g, %d sources (distance to boundary / Rc: min %.2f, max %.2f), weights %s"
          % ("Laplace" if kh == 0 else "Helmholtz", kh, args.nsrc, dist.min(), dist.max(),
             np.array2string(W, precision=2)))
    if kh > 0:
        print("  wavelength / Rc = %.2f" % (2 * np.pi / kh / geo.Rc))

    def bc(pts):
        pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
        z = np.stack(geo.forward(pts), -1)
        r = np.linalg.norm(z[:, None, :] - SRC[None, :, :], axis=-1)
        return (np.cos(kh * r) / (4 * np.pi * r)) @ W

    def bfield(q, kh_):
        return -kh_ * kh_ * torch.ones_like(q[:, 0])

    param_geom = torus.param_geom(jax_avail=False, torch_avail=True, hpsalt=True)
    pdo = param_geom.transform_helmholtz_pdo(bfield, kh)

    # ---- OMS
    dSlabs, connectivity, H = torus.dSlabs(Ns)
    print("Ns = %d, H = %.4f (H/Rc = %.2f), nz = %d, n = %d, m = %d, p = %d"
          % (Ns, H, H / geo.Rc, args.nz, n_core, m_arch, p))
    cyl_opts = dict(Rc=torus.Rc, s=torus.s, n=n_core, m=m_arch, nz=args.nz, flux="conormal")
    opts = solverWrap.solverOptions("cylhps", p, None, use_ctxT=False, mumps_stats=False,
                                    reduced_gpu=True, cyl=cyl_opts)
    gb = lambda q: torus.gb(q, jax_avail=False, torch_avail=True)
    max_level = int(np.log2(n_core ** 2)) - 2

    def make_tree(XXI):
        XXI = XXI.detach().cpu().numpy() if torch.is_tensor(XXI) else np.asarray(XXI)
        return circInterfaceTree(XXI[:, 1:3], Rc=torus.Rc, s=torus.s, n_core=n_core,
                                 m_arch=m_arch, max_level=max_level)

    n_iface = (n_core ** 2 + 4 * m_arch * n_core) * (p - 2) ** 2
    leaf_size = n_iface // 2 ** max_level
    assembler = mA.rkHMatAssembler(leaf_size, args.rk0, ndim=3, tree=make_tree)
    tic = time.perf_counter()
    OMS = oms.oms(dSlabs, pdo, gb, opts, connectivity, stiff_mat_const=False,
                  constructHBS=True, keepLU=True, keepDense=False)
    Stot, rhs = OMS.construct_Stot_and_rhstot(bc, assembler, dbg=0)
    t_oms = time.perf_counter() - tic
    print("OMS built in %.1fs, Ntot = %d" % (t_oms, OMS.Ntot))

    tic = time.perf_counter()
    strat = rkStrat.linear(args.rk0 + args.rk_step, args.rk_step, skip_first_level=False)
    rb = omsdirectHBS.RedBlackSolverHBS(strat, OMS, tree=make_tree(OMS.interface_points), quad=False,
                                        compute_device=args.device, seed=None, dbg=0, normalized=False,
                                        constant_blocks=False, operator="hbs", cyclic=True)
    rb.factorize()
    t_rb = time.perf_counter() - tic
    print("red-black setup + factor %.1fs" % t_rb)

    # ---- solve and compare with the exact trace on all interface nodes
    M = LinearOperator(Stot.shape, matvec=rb.solve, dtype=np.float64)
    tol_kw = {"rtol": args.tol} if Version(scipy.__version__) >= Version("1.12") else {"tol": args.tol}
    its = []
    tic = time.perf_counter()
    x, info = gmres(Stot, rhs, M=M, restart=100, maxiter=1,
                    callback=lambda r: its.append(r), callback_type="pr_norm", **tol_kw)
    t_gm = time.perf_counter() - tic
    u_ex = OMS.interface_values(bc)
    err = np.max(np.abs(x - u_ex))
    scale = np.max(np.abs(u_ex))
    res = np.linalg.norm(Stot @ x - rhs) / np.linalg.norm(rhs)
    print("GMRES: %d its (info %d), %.1fs, relative residual %.2e" % (len(its), info, t_gm, res))
    print("interface error: max %.3e, relative %.3e  (max |u| = %.3e)" % (err, err / scale, scale))


if __name__ == "__main__":
    main()
