"""
precision_probe.py

Small, fast diagnostic for the ~3e-7 floor in RIGHT ERR.

Runs one double-slab discretization at N=9, p=6, kh=20 (seconds, not hours) and
answers four questions in order:

  [1] What dtype do the local solver objects carry?
  [2] Is the source-to-target operator a deterministic linear map to 1e-15,
      or does the same column give different answers in different batch shapes?
  [3] How accurate is a single local Dirichlet solve against the exact solution?
      (no compression anywhere in this path)
  [4] What compression error does the assembler report at this size?

Expected if everything is float64:
    [2] ~1e-15        [3] small (sets the discretization floor)     [4] << 3e-7
Expected if the leaf solves are single precision:
    [2] ~1e-7         [3] ~1e-7                                     [4] ~3e-7

Usage:
    python precision_probe.py                # defaults below
    python precision_probe.py --kh 99.7 --N 33 --p 10     # full-size, slow
"""

import argparse
import inspect
import numpy as np
import torch

import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import geometry.geom_3D.cube as cube


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def to_np(x):
    """Torch tensor or numpy array -> numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def describe(name, o):
    """Print whatever type/dtype/device information an object exposes."""
    bits = [type(o).__name__]
    for a in ("dtype", "device", "shape"):
        if hasattr(o, a):
            bits.append("%s=%s" % (a, getattr(o, a)))
    print("    %-14s %s" % (name, "  ".join(bits)))


def rel(a, b):
    a, b = to_np(a).ravel(), to_np(b).ravel()
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ----------------------------------------------------------------------------
# problem setup (mirrors ULVtest, torch/hpsalt branch, shrunk)
# ----------------------------------------------------------------------------

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

    p_disc = args.p + 2                       # hpsalt convention, as in ULVtest
    a = np.array([H / 4, args.leaf / 2, args.leaf / 2])
    opts = solverWrap.solverOptions("hpsalt", [p_disc, p_disc, p_disc], a,
                                    reduced_gpu=args.reduced_gpu)
    assembler = mA.rkHMatAssembler(args.leaf_size, args.rank, ndim=3)

    OMS = oms.oms(dSlabs, Helm,
                  lambda q: cube.gb(q, jax_avail=False, torch_avail=True),
                  opts, connectivity, stiff_mat_const=True)

    return OMS, assembler, bc, dSlabs, H


# ----------------------------------------------------------------------------
# [1] dtype probe
# ----------------------------------------------------------------------------

def probe_dtypes(OMS, s):
    print("\n[1] dtype / device of the local solver objects")
    for attr in ("solver_ii", "Aib", "Aii", "XX", "XXi", "XXb", "Ii", "Ib"):
        if hasattr(s, attr):
            describe(attr, getattr(s, attr))
        else:
            print("    %-14s (absent)" % attr)

    # solver_ii is usually a factorization / LinearOperator with no dtype of its
    # own. Its OUTPUT dtype is the number that matters.
    try:
        n = s.Aib.shape[0]
        y = s.solver_ii @ np.random.standard_normal((n, 2))
        describe("solver_ii out", y)
        if hasattr(y, "dtype") and str(getattr(y, "dtype")).endswith("float32"):
            print("    >>> SINGLE PRECISION on the local solve output <<<")
    except Exception as e:
        print("    solver_ii probe failed:", repr(e))

    print("    torch default dtype:", torch.get_default_dtype())
    print("    construct() signature:", end=" ")
    try:
        print(inspect.signature(type(s).construct))
    except Exception:
        print("(unavailable)")


# ----------------------------------------------------------------------------
# [2] determinism of the source-to-target operator
# ----------------------------------------------------------------------------

def probe_determinism(A, label):
    """Same column, three batch widths. float64 => ~1e-15."""
    print("\n[2] determinism of %s (same column, different batch shape)" % label)
    n = A.shape[1]
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n, 1))

    a1 = to_np(A @ v)
    out = {}
    for w in (4, 64):
        V = rng.standard_normal((n, w))
        V[:, 0:1] = v
        out[w] = to_np(A @ V)[:, 0:1]
        print("    batch 1 vs %-3d : %.3e" % (w, rel(out[w], a1)))

    # and a straight repeat of the identical call
    print("    batch 1 vs 1   : %.3e" % rel(to_np(A @ v), a1))
    print("    (float64 linear map ~1e-15;  ~1e-7 means single precision)")


# ----------------------------------------------------------------------------
# [3] local discretization error, no compression in the path
# ----------------------------------------------------------------------------

def probe_local_solve(s, bc):
    print("\n[3] single local Dirichlet solve vs exact solution")
    try:
        ue = bc(s.XXi)
        ui = -(s.solver_ii @ (s.Aib @ bc(s.XXb)))
        e = rel(ui, ue)
        print("    ||u_h - u|| / ||u|| = %.3e" % e)
        print("    (this is the discretization floor; compression cannot beat it)")
        return e
    except Exception as ex:
        print("    failed:", repr(ex))
        print("    -> check whether XXb/XXi match XX[Ib]/XX[Ii] for this wrapper")
        return None


# ----------------------------------------------------------------------------
# [4] assembler compression error at this size
# ----------------------------------------------------------------------------

def probe_compression(OMS, assembler, bc):
    print("\n[4] assembler compression error (LEFT ERR / RIGHT ERR below)")
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=1)
    print("    nc = %d   Ntot = %d   blocks = %d" % (nc, Ntot, len(S_rk_list)))
    return S_rk_list


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=9)
    ap.add_argument("--p", type=int, default=6)
    ap.add_argument("--kh", type=float, default=20.0)
    ap.add_argument("--leaf", type=float, default=1.0 / 8)   # leaf width in y,z
    ap.add_argument("--leaf_size", type=int, default=128)    # HBS leaf
    ap.add_argument("--rank", type=int, default=64)          # HBS rank
    ap.add_argument("--reduced_gpu", type=int, default=1)
    ap.add_argument("--skip_compression", action="store_true")
    args = ap.parse_args()
    args.reduced_gpu = bool(args.reduced_gpu)

    print("N=%d  p=%d  kh=%.4f  leaf=%.4f  HBS(leaf=%d, rank=%d)  reduced_gpu=%s"
          % (args.N, args.p, args.kh, args.leaf, args.leaf_size, args.rank,
             args.reduced_gpu))
    print("points per wavelength in y,z ~ %.1f"
          % ((2 * np.pi / args.kh) / args.leaf * args.p))

    OMS, assembler, bc, dSlabs, H = build(args)

    # one slab is enough for [1]-[3]; this also populates OMS._ref_solver
    s, XXb, XXi, _ = OMS._slab_solver(0)

    probe_dtypes(OMS, s)
    probe_local_solve(s, bc)

    # source-to-target map for [2], uncompressed
    try:
        Il, Ir, Ic, Igb, XXi2, XXb2 = OMS._slab_indices(0, s, XXb, XXi)[:6]
        st_l, st_r = OMS.compute_stmaps(Il, Ic, Ir, XXi2, XXb2, s)
        A = st_r.A if len(Ir) else st_l.A
        probe_determinism(A, "st.A (uncompressed source->target)")
    except Exception as ex:
        print("\n[2] skipped:", repr(ex))
        print("    _slab_indices/compute_stmaps signature differs; print with")
        print("    inspect.signature(oms.oms._slab_indices) and adjust")

    if not args.skip_compression:
        probe_compression(OMS, assembler, bc)

    print("\nRead the results together:")
    print("  [2] ~1e-7 and [3] ~1e-7  -> single precision in the leaf solves")
    print("  [2] ~1e-15 and [3] ~1e-7 -> genuine discretization error, raise p")
    print("  [2] ~1e-15 and [3] small but [4] ~3e-7 -> look inside HBStorch's")
    print("      error measurement rather than at the operator")


if __name__ == "__main__":
    main()
