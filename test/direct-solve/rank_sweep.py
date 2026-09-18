"""
rank_sweep.py

Does the HBS compression error of the source-to-target block decay with rank
at fixed kh, or is it floored?

  * the system is built ONCE (one slab discretization, one MUMPS factorization)
  * the operator is sampled ONCE at the largest s in the sweep; every smaller
    rank reuses a column prefix of the same Om / Psi / Y / Z
  * the reference matvecs for the error check are computed ONCE, out of sample
  * the csv is rewritten after EVERY rank, so an OOM later in the sweep never
    costs a result already paid for
  * an OOM on one rank is recorded as a nan row and the sweep continues

The sample count follows matAssembler's rkHBS path exactly:
    s(rank) = 2*max(rank, leaf_size) + rank + 20

Memory notes:
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set below, before torch
    is imported, which is the only point at which it takes effect
  * construct() runs with fast=True, matching matAssembler's production path
  * the HBSMAT and the torch caching allocator are released between ranks

Usage:
    python rank_sweep.py                                  # production defaults
    python rank_sweep.py --ranks 400,600 --leaf_size 1600
    python rank_sweep.py --N 9 --p 6 --kh 20 --leaf 0.125 --leaf_size 128 \
                         --ranks 16,32,64                 # small smoke test
"""

import os

# must precede `import torch`, or the allocator is already configured
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import gc
import sys
import time

import numpy as np
import torch

import solver.solver as solverWrap
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import geometry.geom_3D.cube as cube
import matAssembly.HBS.HBStorch as HBStorch
import matAssembly.HBS.slabTree as slabTree


# ----------------------------------------------------------------- logging --

class Tee(object):
    """Write to console and to a log file at the same time."""

    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, msg):
        sys.__stdout__.write(msg)
        sys.__stdout__.flush()
        self.f.write(msg)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def nsamples(rank, leaf_size):
    """Exactly matAssembler's rule for the rkHBS path."""
    return 2 * max(rank, leaf_size) + rank + 20


def write_csv(rows, path):
    """Plain rewrite of the whole file; called after every rank."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cuda_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_GB():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


# ------------------------------------------------------------------- setup --

def build(args):
    kh = args.kh

    def c11(p): return torch.ones_like(p[:, 0])
    def c22(p): return torch.ones_like(p[:, 1])
    def c33(p): return torch.ones_like(p[:, 2])
    def c(p):   return -kh * kh * torch.ones_like(p[:, 0])

    Helm = pdoalt.PDO_3d(c11=c11, c22=c22, c33=c33, c=c)

    src = np.array([-.5, -.2, 1.])

    def bc(p):
        p = to_np(p)
        rr = np.linalg.norm(p - src.T, axis=1)
        return np.real(np.exp(1j * kh * rr) / (4 * np.pi * rr))

    dSlabs, connectivity, H = cube.dSlabs(args.N)
    p_disc = args.p + 2                      # hpsalt convention
    a = np.array([H / 4, args.leaf / 2, args.leaf / 2])
    opts = solverWrap.solverOptions("hpsalt", [p_disc, p_disc, p_disc], a,
                                    reduced_gpu=args.reduced_gpu)

    OMS = oms.oms(dSlabs, Helm,
                  lambda q: cube.gb(q, jax_avail=False, torch_avail=True),
                  opts, connectivity, stiff_mat_const=True)
    return OMS, bc, H


def get_stmap(OMS, slab=0):
    """One uncompressed source-to-target map, plus the slab solver."""
    s, XXb, XXi, _ = OMS._slab_solver(slab)
    idx = OMS._slab_indices(slab, s, XXb, XXi)
    Il, Ir, Ic, Igb, XXi2, XXb2 = idx[:6]
    st_l, st_r = OMS.compute_stmaps(Il, Ic, Ir, XXi2, XXb2, s)
    st = st_r if (st_r is not None and len(Ir)) else st_l
    return s, st, XXi2, XXb2, Ic


def power_norm(A, iters=20, seed=1):
    """Largest singular value of A by power iteration on A^T A.

    The Frobenius estimate over a few Gaussian columns reports the RMS of the
    spectrum and is blind to an isolated resonant mode, which is precisely the
    mode that matters here.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((A.shape[1], 1))
    x /= np.linalg.norm(x)
    sig = float("nan")
    for _ in range(iters):
        y = to_np(A @ x)
        x = to_np(A.T @ y)
        nx = np.linalg.norm(x)
        if nx == 0:
            break
        sig = np.sqrt(nx)
        x = x / nx
    return sig


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=33)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--kh", type=float, default=99.7)
    ap.add_argument("--leaf", type=float, default=0.03125)   # discr. leaf in y,z
    ap.add_argument("--leaf_size", type=int, default=800)    # HBS leaf
    ap.add_argument("--ranks", type=str, default="32,64,128,256,400,500")
    ap.add_argument("--ntest", type=int, default=16)         # out-of-sample cols
    ap.add_argument("--reduced_gpu", type=int, default=1)
    ap.add_argument("--slab", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", type=int, default=1,
                    help="construct(fast=...); 1 matches matAssembler")
    ap.add_argument("--power_iters", type=int, default=20,
                    help="0 disables the power-iterated ||A||")
    args = ap.parse_args()
    args.reduced_gpu = bool(args.reduced_gpu)
    args.fast = bool(args.fast)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = args.out or ("rank_sweep_kh%.4g_N%d_p%d_ls%d_%s"
                        % (args.kh, args.N, args.p, args.leaf_size, stamp))
    csv_path = base + ".csv"
    log_path = base + ".log"

    tee = Tee(log_path)
    sys.stdout = tee

    ranks = sorted(int(r) for r in args.ranks.replace(" ", "").split(","))
    rank_max = ranks[-1]
    s_max = nsamples(rank_max, args.leaf_size)

    print("rank_sweep  %s" % stamp)
    print("kh=%.4f  N=%d  p=%d  leaf=%.5f  HBS leaf_size=%d  reduced_gpu=%s"
          % (args.kh, args.N, args.p, args.leaf, args.leaf_size,
             args.reduced_gpu))
    print("ranks: %s   fast=%s" % (ranks, args.fast))
    print("s(rank) = 2*max(rank, leaf_size) + rank + 20;  s_max = %d" % s_max)
    print("PYTORCH_CUDA_ALLOC_CONF = %s" % os.environ["PYTORCH_CUDA_ALLOC_CONF"])
    print("log -> %s" % log_path)
    print("csv -> %s" % csv_path)

    # ---- build once -------------------------------------------------------
    t0 = time.time()
    OMS, bc, H = build(args)
    slv, st, XXi, XXb, Ic = get_stmap(OMS, args.slab)
    A = st.A
    XXI = st.XXI
    n, m = A.shape[0], A.shape[1]
    print("\nsystem built in %.1f s;  block shape = (%d, %d)"
          % (time.time() - t0, n, m))
    print("sample storage for Om/Psi/Y/Z at s_max: %.2f GB"
          % (4 * n * s_max * 8 / 1e9))

    # ---- local discretization error, no compression anywhere --------------
    try:
        ue = bc(slv.XXi)
        uh = -(slv.solver_ii @ (slv.Aib @ bc(slv.XXb)))
        disc_err = np.linalg.norm(to_np(uh).ravel() - ue.ravel()) \
            / np.linalg.norm(ue.ravel())
        print("local Dirichlet solve vs exact: %.4e   <-- discretization floor"
              % disc_err)
    except Exception as ex:
        disc_err = float("nan")
        print("local solve probe failed: %r" % (ex,))

    # ---- sample once ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    print("\nsampling once at s = %d ..." % s_max)
    t0 = time.time()
    Om = rng.standard_normal((n, s_max))
    Psi = rng.standard_normal((m, s_max))
    Y = A @ Om
    Z = A.T @ Psi
    t_sample = time.time() - t0
    print("sampling done in %.1f s  (%.3f s per column)"
          % (t_sample, t_sample / s_max))

    # ---- reference matvecs, out of sample, computed once ------------------
    print("computing %d out-of-sample reference columns ..." % args.ntest)
    t0 = time.time()
    V = rng.standard_normal((m, args.ntest))
    W = rng.standard_normal((n, args.ntest))
    AV = to_np(A @ V)
    AtW = to_np(A.T @ W)
    nAV, nAtW = np.linalg.norm(AV), np.linalg.norm(AtW)
    norm_rms = nAV / np.linalg.norm(V)
    print("reference done in %.1f s;  ||A V||_F = %.4e" % (time.time() - t0, nAV))
    print("  RMS estimate of ||A||  : %.3f   (blind to isolated modes)"
          % norm_rms)

    if args.power_iters > 0:
        t0 = time.time()
        norm_2 = power_norm(A, iters=args.power_iters)
        print("  power-iterated ||A||_2 : %.3f   (%d iters, %.1f s)"
              % (norm_2, args.power_iters, time.time() - t0))
        if norm_2 > 5 * norm_rms:
            print("  >>> spectrum has an isolated large mode: near-resonant <<<")
    else:
        norm_2 = float("nan")

    # ---- tree once --------------------------------------------------------
    quad = False
    tree = slabTree.slabTree(XXI, quad, args.leaf_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device: %s" % device)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print("gpu free / total: %.2f / %.2f GB" % (free / 1e9, total / 1e9))

    # ---- sweep ------------------------------------------------------------
    rows = []

    def base_row(rk, s_rk, **kw):
        r = dict(rank=rk, s=s_rk, fwd_rel=float("nan"), adj_rel=float("nan"),
                 t_construct=float("nan"), mem_MB=float("nan"),
                 peak_GB=float("nan"), status="ok")
        r.update(kw)
        r.update(dict(kh=args.kh, N=args.N, p=args.p, leaf=args.leaf,
                      leaf_size=args.leaf_size, fast=args.fast, n=n, m=m,
                      norm_A_rms=norm_rms, norm_A_2=norm_2,
                      disc_err=disc_err, t_sample=t_sample))
        return r

    print("\n%8s %8s %12s %12s %12s %10s %9s"
          % ("rank", "s", "fwd_rel", "adj_rel", "t_constr", "mem_MB", "peak_GB"))
    print("-" * 78)

    for rk in ranks:
        s_rk = nsamples(rk, args.leaf_size)
        if s_rk > s_max:
            print("%8d  skipped (s=%d > s_max=%d)" % (rk, s_rk, s_max))
            continue

        cuda_reset()
        Om_r, Psi_r = Om[:, :s_rk], Psi[:, :s_rk]
        Y_r, Z_r = Y[:, :s_rk], Z[:, :s_rk]
        M = None

        try:
            M = HBStorch.HBSMAT(device=device, tree=tree, quad=quad)
            t0 = time.time()
            M.construct(rk, Om_r, Psi_r, Y_r, Z_r, fast=args.fast)
            t_constr = time.time() - t0

            fwd = np.linalg.norm(to_np(M.matmat(V)) - AV) / nAV
            try:
                adj = np.linalg.norm(to_np(M.rmatmat(W)) - AtW) / nAtW
            except Exception:
                adj = float("nan")
            mem = getattr(M, "nbytes", 0) / 1e6
            peak = cuda_peak_GB()

            print("%8d %8d %12.4e %12.4e %12.2f %10.1f %9.2f"
                  % (rk, s_rk, fwd, adj, t_constr, mem, peak))
            rows.append(base_row(rk, s_rk, fwd_rel=fwd, adj_rel=adj,
                                 t_construct=t_constr, mem_MB=mem,
                                 peak_GB=peak, status="ok"))

        except torch.OutOfMemoryError as ex:
            peak = cuda_peak_GB()
            print("%8d %8d   OOM at peak %.2f GB -- %s"
                  % (rk, s_rk, peak, str(ex).split(".")[0]))
            print("           try a smaller --leaf_size, or chunk the leaf")
            print("           batch in HBStorch.compute_UV_pair's qr")
            rows.append(base_row(rk, s_rk, peak_GB=peak, status="OOM"))

        except Exception as ex:
            print("%8d %8d   FAILED: %r" % (rk, s_rk, ex))
            rows.append(base_row(rk, s_rk, status="fail:%s" % type(ex).__name__))

        finally:
            M = None
            cuda_reset()
            write_csv(rows, csv_path)      # after every rank, ok or not

    # ---- summary ----------------------------------------------------------
    good = [r for r in rows if r["status"] == "ok"]
    print("\nwrote %d rows (%d ok) to %s" % (len(rows), len(good), csv_path))

    if len(good) >= 2:
        f0, r0 = good[0]["fwd_rel"], good[0]["rank"]
        f1, r1 = good[-1]["fwd_rel"], good[-1]["rank"]
        print("fwd_rel %.3e (rank %d) -> %.3e (rank %d):  factor %.1f"
              % (f0, r0, f1, r1, f0 / max(f1, 1e-300)))
        print("\ndecay rate, decades per unit rank:")
        for a, b in zip(good[:-1], good[1:]):
            d = (np.log10(a["fwd_rel"]) - np.log10(b["fwd_rel"])) \
                / (b["rank"] - a["rank"])
            print("  %4d -> %4d : %.4f" % (a["rank"], b["rank"], d))
        print("\na steady rate means rank is the knob; a collapsing rate means")
        print("the knee is real (or leaf_size is too close to rank).")
    if good:
        print("compression cannot beat the %.3e discretization floor." % disc_err)

    sys.stdout = sys.__stdout__
    tee.close()


if __name__ == "__main__":
    main()
