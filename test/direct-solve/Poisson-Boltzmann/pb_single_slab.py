"""
Single-slab first pass for nonlinear Poisson-Boltzmann (HANDOFF_PB_single_slab.md,
Sec. 6), Approach 1 with the statically condensed slab system only.

Problem: double slab [0, 2H] x D_Rc of the full twisted elliptic torus
(H = L / Ns, Ns = --Ns, default 32), exact Dirichlet data phi* on the end faces and the wall,
so the exact local solution is phi*.

Stages (select with --stage):
  0  building-block checks: torus map, transform_pdo vs transform_helmholtz_pdo
  1  linear verification: c_ref = Lam^2 N/tau, h = (-Lap + c_ref) phi*;
     error vs phi* at leaf nodes, at the midplane (skeleton), off-grid; p sweep
  4  local chord  phi_{j+1} = M^{-1}(f* - G[phi_j] + c_ref phi_j)  for
     Lam in --Lam and c_ref in --cref:
        const    : midpoint constant of c[phi*] over the slab
        pert<d>  : c[phi_c], phi_c = phi* + d tau eta (eta smooth, |eta| <= 1);
                   stand-in for the prolongated coarse solution (Stage 2), rho ~ d
     started from phi_c (const: from phi* + 0.3 tau eta).
Output: CSV files in --out.

Run from this folder with the repo root on PYTHONPATH, e.g.
    PYTHONPATH=../../.. python pb_single_slab.py --stage 0 1 4
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

import geometry.geom_3D.full_twisted_torus as torus
from pb.manufactured import PBManufactured, transform_pdo, check_transform_pdo
from pb.local_chord import CondensedSlab, chord

torch.set_default_dtype(torch.double)
torus.s = 0.55


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, nargs="+", default=[0, 1, 4])
    ap.add_argument("--p1", type=int, nargs="+", default=[6, 8, 10], help="orders, stage 1")
    ap.add_argument("--p4", type=int, default=8, help="order, stage 4")
    ap.add_argument("--nm", type=str, nargs="+", default=["8,2"], help="n,m pairs")
    ap.add_argument("--nz", type=int, default=2)
    ap.add_argument("--Ns", type=int, default=32, help="slabs around the torus: H = L / Ns")
    ap.add_argument("--Lam", type=float, nargs="+", default=[1.0, 10.0, 30.0])
    ap.add_argument("--cref", type=str, nargs="+", default=["const", "pert0.3", "pert0.1"])
    ap.add_argument("--spread", type=float, default=3.0, help="variation of phi*/tau")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--maxit", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no-mumps", action="store_true")
    ap.add_argument("--out", type=str, default="results_pb_single_slab")
    return ap.parse_args()


def slab_geom(Ns):
    slabs, _, H = torus.dSlabs(Ns)
    return slabs[0], H


def random_slab_points(n, geom, seed=1):
    rng = np.random.default_rng(seed)
    r = torus.Rc * np.sqrt(rng.uniform(size=n)); th = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(geom[0][0], geom[1][0], n)
    return np.c_[t, r * np.cos(th), r * np.sin(th)]


def eta(W):
    """Smooth perturbation shape, |eta| <= 1 (reference coordinates)."""
    Rc = torus.Rc
    return np.cos(1.3 * W[:, 0] + 0.4) * np.sin(2.0 * W[:, 1] / Rc + 0.2) * np.cos(1.7 * W[:, 2] / Rc)


def write_csv(path, rows):
    """Append rows to path (header written once).  If path exists with a
    different header, write to a new time-stamped file instead."""
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


def build(geom, c_fun, p, n, m, args):
    pg = torus.param_geom(jax_avail=False, torch_avail=True, hpsalt=True)
    cyl = dict(Rc=torus.Rc, s=torus.s, n=n, m=m, nz=args.nz, flux="conormal",
               device=torch.device(args.device))
    return CondensedSlab(geom, transform_pdo(pg, c_fun), p, cyl, use_mumps=not args.no_mumps)


def errors(cs, mms, phi, x, Xoff):
    ex = mms.phistar(cs.W)
    e_nodes = np.max(np.abs(phi - ex))
    e_mid = np.max(np.abs(x[cs.Imid] - mms.phistar(cs.slab.XX[cs.Imid])))
    e_off = np.max(np.abs(cs.interp(phi) - mms.phistar(Xoff)))
    return e_nodes, e_mid, e_off


# ---------------------------------------------------------------- stage 0
def stage0(args, geom):
    print("== stage 0: building blocks")
    torus.check_param()
    pg =torus.param_geom(jax_avail=False, torch_avail=True, hpsalt=True)
    print("  transform_pdo vs transform_helmholtz_pdo, max diff = %.2e  (expect 0)"
          % check_transform_pdo(pg))


# ---------------------------------------------------------------- stage 1
def stage1(args, geom):
    print("== stage 1: linear verification, c_ref = Lam^2 N / tau")
    rows = []
    for nm in args.nm:
        n, m = map(int, nm.split(","))
        for Lam in args.Lam:
            mms = PBManufactured(torus, Lam=Lam)
            mms.calibrate_A(random_slab_points(4000, geom), args.spread)
            c_fun = lambda W, mms=mms: mms.Lam ** 2 * mms.N(mms.psi(W)) / mms.tau(mms.psi(W))
            for p in args.p1:
                cs = build(geom, c_fun, p, n, m, args)
                Xoff = cs.offgrid_points()
                h = -mms.lap_phistar(cs.W) + c_fun(cs.W) * mms.phistar(cs.W)
                b = mms.phistar(cs.Wb)
                t0 = time.perf_counter()
                phi, x = cs.solve(h, b)
                t_solve = time.perf_counter() - t0
                # Dirichlet path: h = 0 gives -Aii^{-1} Aib b (no body load)
                phi0, x0 = cs.solve(np.zeros_like(h), b)
                e_dir = np.max(np.abs(x0[cs.slab.Ii] - cs._solve_ii(-(cs.slab.Aib @ b))))
                e_nodes, e_mid, e_off = errors(cs, mms, phi, x, Xoff)
                r = dict(Ns=args.Ns, H=(geom[1][0] - geom[0][0]) / 2, n=n, m=m, nz=args.nz, p=p, Lam=Lam, A=mms.A,
                         N_skel=cs.slab.Aii.shape[0], N_nodes=len(cs.W),
                         err_nodes=e_nodes, err_mid=e_mid, err_offgrid=e_off,
                         h0_vs_dirichlet=e_dir, t_build=cs.t_build, t_solve=t_solve,
                         t_analysis=cs.factor.get("time_analysis"),
                         t_factor=cs.factor.get("time_factor"),
                         nnz_factor=cs.factor.get("nnz_factor"), mem_MB=cs.factor.get("mem_MB"))
                print("  n=%d m=%d p=%2d Lam=%4g: err nodes %.2e  mid %.2e  off-grid %.2e"
                      "  (h=0 vs Dirichlet %.1e)" % (n, m, p, Lam, e_nodes, e_mid, e_off, e_dir))
                rows.append(r)
                del cs
    write_csv(os.path.join(args.out, "stage1_linear.csv"), rows)


# ---------------------------------------------------------------- stage 4
def stage4(args, geom):
    print("== stage 4: local chord iteration (condensed path)")
    rows, iters = [], []
    p = args.p4
    for nm in args.nm:
        n, m = map(int, nm.split(","))
        for Lam in args.Lam:
            mms = PBManufactured(torus, Lam=Lam)
            mms.calibrate_A(random_slab_points(4000, geom), args.spread)
            for kind in args.cref:
                if kind == "const":
                    Wr = random_slab_points(20000, geom, seed=2)
                    cs_ = mms.dg(Wr, mms.phistar(Wr))
                    cbar = 0.5 * (cs_.min() + cs_.max())
                    c_fun = lambda W, cbar=cbar: cbar + 0.0 * W[:, 0]
                    delta0 = 0.3
                elif kind.startswith("pert"):
                    delta0 = float(kind[4:])
                    c_fun = lambda W, d=delta0, mms=mms: mms.dg(
                        W, mms.phistar(W) + d * mms.tau(mms.psi(W)) * eta(W))
                else:
                    raise ValueError(kind)
                cs = build(geom, c_fun, p, n, m, args)
                Xoff = cs.offgrid_points()
                W, I = cs.W, cs.interior
                psi = mms.psi(W)
                fst = mms.fstar(W)
                cref = c_fun(W)
                ex = mms.phistar(W)
                b = mms.phistar(cs.Wb)
                phi0 = ex + delta0 * mms.tau(psi) * eta(W)
                h_of = lambda ph: fst - mms.g(W, ph, psi) + cref * ph
                rho_star = np.max(np.abs(1 - mms.dg(W, ex, psi)[I] / cref[I]))
                rho_it = [np.max(np.abs(1 - mms.dg(W, phi0, psi)[I] / cref[I]))]
                state = dict(prev_inc=None)

                def monitor(j, ph, x, log):
                    e_nodes = np.max(np.abs(ph - ex))
                    e_mid = np.max(np.abs(x[cs.Imid] - mms.phistar(cs.slab.XX[cs.Imid])))
                    rho_it.append(np.max(np.abs(1 - mms.dg(W, ph, psi)[I] / cref[I])))
                    rate = np.nan if state["prev_inc"] is None else log["increment"] / state["prev_inc"]
                    state["prev_inc"] = log["increment"]
                    log.update(err_nodes=e_nodes, err_mid=e_mid, rate=rate,
                               rho_pred=max(rho_star, max(rho_it)))
                    print("    it %3d  inc %.2e  res(prev) %.2e  err %.2e  mid %.2e  rate %.3f"
                          % (j, log["increment"], log["residual_prev"], e_nodes, e_mid, rate))

                cs.timers = dict(reduce=0.0, skeleton=0.0, reconstruct=0.0)
                print("  n=%d m=%d p=%d Lam=%g c_ref=%s: rho* = %.3f" % (n, m, p, Lam, kind, rho_star))
                t0 = time.perf_counter()
                phi, x, logs = chord(lambda h: cs.solve(h, b), phi0, h_of,
                                     tol=args.tol, maxit=args.maxit, monitor=monitor, interior=I)
                t_tot = time.perf_counter() - t0
                e_nodes, e_mid, e_off = errors(cs, mms, phi, x, Xoff)
                k = len(logs)
                rates = [l["rate"] for l in logs if np.isfinite(l["rate"]) and l["increment"] > 1e3 * args.tol]
                r = dict(Ns=args.Ns, H=(geom[1][0] - geom[0][0]) / 2, n=n, m=m, nz=args.nz, p=p, Lam=Lam, cref=kind, A=mms.A,
                         iterations=k, converged=logs[-1]["increment"] < args.tol,
                         final_increment=logs[-1]["increment"],
                         err_nodes=e_nodes, err_mid=e_mid, err_offgrid=e_off,
                         rho_star=rho_star, rho_pred=max(rho_star, max(rho_it)),
                         rate_observed=np.exp(np.mean(np.log(rates[-5:]))) if rates else np.nan,
                         t_build=cs.t_build, t_analysis=cs.factor.get("time_analysis"),
                         t_factor=cs.factor.get("time_factor"),
                         nnz_factor=cs.factor.get("nnz_factor"), mem_MB=cs.factor.get("mem_MB"),
                         t_chord=t_tot,
                         t_G_per_it=sum(l["t_G"] for l in logs) / k,
                         t_reduce_per_it=cs.timers["reduce"] / k,
                         t_skeleton_per_it=cs.timers["skeleton"] / k,
                         t_reconstruct_per_it=cs.timers["reconstruct"] / k)
                print("  -> %d its, err nodes %.2e mid %.2e off-grid %.2e, rate %.3f vs rho_pred %.3f"
                      % (k, e_nodes, e_mid, e_off, r["rate_observed"], r["rho_pred"]))
                rows.append(r)
                iters += [dict(Ns=args.Ns, n=n, m=m, nz=args.nz, p=p, Lam=Lam, cref=kind, **l) for l in logs]
                del cs
    write_csv(os.path.join(args.out, "stage4_chord.csv"), rows)
    write_csv(os.path.join(args.out, "stage4_iterations.csv"), iters)


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    geom, H = slab_geom(args.Ns)
    print("slab t in [%.4f, %.4f], H = %.4f, s = %.2f, Rc = %.4f"
          % (geom[0][0], geom[1][0], H, torus.s, torus.Rc))
    if 0 in args.stage:
        stage0(args, geom)
    if 1 in args.stage:
        stage1(args, geom)
    if 4 in args.stage:
        stage4(args, geom)
