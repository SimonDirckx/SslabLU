"""
Is cublasDgeqrfBatched worth a ctypes wrapper, at the shapes this solver
actually produces?

Measured so far, all float64 on a V100:

  square sweep, Nb=512, S=P
      256x256    0.2635 ms/mat      <- fast batched path
      320x320    1.4690 ms/mat      <- dispatch boundary, 5.6x jump
      512x512    2.6259 ms/mat

  non-square, Nb=512
      394x512    2.0683 ms/mat      torch
      394x256    1.3716 ms/mat      torch  -- NOT the fast path either, so
                                      splitting W into two 394x256 halves
                                      costs 2.74 ms vs 2.07 ms: a net loss
      394x512    1.4620 ms/mat      cublasDgeqrfBatched  (1.41x)
      394x512    1.3460 ms/mat      MKL on host (689 ms / 512)

So the boundary keys on max(m,n), not on the product, and s ~ 3*rk keeps
every level above it.  cuBLAS is the only device-side thing that has moved
the number.  This script measures it across the real shapes, weighted by how
often each occurs, to decide whether ~1.4x on 82% of compression justifies
the wrapper.

The shapes come from the solver's own trace at Ntot=1048576, rk=128, nl=128,
with compute_UV_pair merging the Om/Y and Psi/Z factorizations:

    (1024, 394, 256)   leaf, p = n + ny = 128 + 128
    (512,  394, 512)   \
    (256,  394, 512)    |  every level above: p = 256 + 256
    ...                 |  Nb halves each level
    (4,    394, 512)   /

Each of these is issued once per compute_UV_pair call, and there are 37
blocks x 2 constructors' worth in a full factorization -- but the relative
weighting within one construct is what matters for the decision.

LAYOUT.  cuBLAS is column-major.  A contiguous torch (Nb, P, S) tensor read
column-major is a batch of S-by-P matrices with lda = S.  Getting this
backwards still returns info=0 while factoring the transpose, which is why
correctness is checked at every shape before timing is reported.
"""

import ctypes
import time

import torch

DTYPE = torch.float64
DEV = "cuda"

# (Nb, s, p) exactly as compute_UV_pair builds W, one entry per level.
LEVEL_SHAPES = [
    (1024, 394, 256),
    (512, 394, 512),
    (256, 394, 512),
    (128, 394, 512),
    (64, 394, 512),
    (32, 394, 512),
    (16, 394, 512),
    (8, 394, 512),
    (4, 394, 512),
]

# Rank 256 is what Helmholtz needs; s = 2*max(rk,nl) + rk + 10 = 778,
# p = n + ny = 256 + 512 = 768 once rkm saturates.  Included to see whether
# the picture changes at the ranks that matter, since both dimensions grow.
LEVEL_SHAPES_RK256 = [
    (512, 778, 512),
    (256, 778, 768),
    (128, 778, 768),
    (64, 778, 768),
    (32, 778, 768),
]

INCLUDE_CPU = True     # MKL uses all cores; see note at the end of the output


# ---------------------------------------------------------------------------
# cuBLAS
# ---------------------------------------------------------------------------
def load_cublas():
    """PyTorch dlopens cuBLAS once CUDA is initialised, so the process symbol
    table is the most portable handle.  'libcublas.so' is a toolkit-only devel
    symlink absent from pip wheels, which ship 'libcublas.so.12' under
    site-packages/nvidia/cublas/lib."""
    torch.zeros(1, device=DEV)

    try:
        lib = ctypes.CDLL(None)
        lib.cublasDgeqrfBatched
        print("cublas: process symbol table")
        return lib
    except AttributeError:
        pass

    import glob
    import os

    roots = [
        os.path.join(os.path.dirname(torch.__file__), "lib"),
        os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia"),
    ]
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "libcublas.so*"),
                              recursive=True):
            try:
                lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                lib.cublasDgeqrfBatched
                print(f"cublas: {path}")
                return lib
            except (OSError, AttributeError):
                continue

    for name in ("libcublas.so.12", "libcublas.so.11", "libcublas.so"):
        try:
            lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            lib.cublasDgeqrfBatched
            print(f"cublas: {name}")
            return lib
        except (OSError, AttributeError):
            continue

    raise RuntimeError("could not locate libcublas with cublasDgeqrfBatched")


cublas = load_cublas()

cublas.cublasCreate_v2.restype = ctypes.c_int
cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cublas.cublasDestroy_v2.restype = ctypes.c_int
cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
cublas.cublasDgeqrfBatched.restype = ctypes.c_int
cublas.cublasDgeqrfBatched.argtypes = [
    ctypes.c_void_p,                # handle
    ctypes.c_int,                   # m
    ctypes.c_int,                   # n
    ctypes.c_void_p,                # Aarray   (device array of double*)
    ctypes.c_int,                   # lda
    ctypes.c_void_p,                # TauArray (device array of double*)
    ctypes.POINTER(ctypes.c_int),   # info     (HOST-side argument check)
    ctypes.c_int,                   # batchCount
]

CUBLAS_STATUS = {
    0: "SUCCESS", 1: "NOT_INITIALIZED", 3: "ALLOC_FAILED", 7: "INVALID_VALUE",
    8: "ARCH_MISMATCH", 11: "MAPPING_ERROR", 13: "EXECUTION_FAILED",
    14: "INTERNAL_ERROR", 15: "NOT_SUPPORTED",
}


def _ptr_array(t, stride_bytes):
    return (torch.arange(t.shape[0], device=t.device, dtype=torch.int64)
            * stride_bytes + t.data_ptr())


def geqrf_batched(handle, A, tau=None, a_ptrs=None, t_ptrs=None):
    """In-place batched QR.  A is contiguous (Nb, P, S), read column-major as
    a batch of S-by-P matrices with lda = S.

    tau / a_ptrs / t_ptrs may be passed in to keep the pointer-array
    construction out of a timing loop; a real wrapper would cache them per
    shape the same way."""
    nb, p, s = A.shape
    assert A.is_contiguous()
    k = min(s, p)

    if tau is None:
        tau = torch.zeros(nb, k, dtype=A.dtype, device=A.device)
    if a_ptrs is None:
        a_ptrs = _ptr_array(A, p * s * A.element_size())
    if t_ptrs is None:
        t_ptrs = _ptr_array(tau, k * tau.element_size())
    info = ctypes.c_int(0)

    status = cublas.cublasDgeqrfBatched(
        handle,
        ctypes.c_int(s), ctypes.c_int(p),
        ctypes.c_void_p(a_ptrs.data_ptr()), ctypes.c_int(s),
        ctypes.c_void_p(t_ptrs.data_ptr()),
        ctypes.byref(info), ctypes.c_int(nb),
    )
    if status != 0:
        raise RuntimeError(f"cublasDgeqrfBatched status={status} "
                           f"({CUBLAS_STATUS.get(status, '?')}) "
                           f"at shape {tuple(A.shape)}")
    if info.value != 0:
        raise RuntimeError(f"cublasDgeqrfBatched bad argument at {-info.value}")
    return A, tau


def r_from_packed(A):
    """R from the packed geqrf output.  The torch view A[b] is the transpose
    of the cuBLAS matrix, so the column-major upper triangle appears as the
    torch lower triangle: R^T = tril(A[b])[:, :k]."""
    nb, p, s = A.shape
    k = min(s, p)
    return torch.tril(A)[:, :, :k].mT.contiguous()


def qr_flops(m, n):
    """Householder QR, 2mn^2 - (2/3)n^3 with n the smaller dimension."""
    a, b = max(m, n), min(m, n)
    return 2.0 * a * b * b - (2.0 / 3.0) * b ** 3


def timed(fn, sync, reps):
    fn()
    sync()
    t = time.time()
    for _ in range(reps):
        fn()
    sync()
    return (time.time() - t) / reps


def check(handle, s, p):
    """Relative |R| agreement on a 2-matrix case.  |R| rather than R because
    QR factors agree only up to column signs, and every consumer downstream
    is sign-invariant."""
    g = torch.Generator(device=DEV).manual_seed(0)
    W = torch.randn(2, s, p, generator=g, dtype=DTYPE, device=DEV)
    A = W.mT.contiguous().clone()
    geqrf_batched(handle, A)
    Rt = torch.linalg.qr(W, mode="r").R
    Rc = r_from_packed(A)[:, :Rt.shape[1], :]
    return ((Rt.abs() - Rc.abs()).abs().max() / Rt.abs().max()).item()


def run(handle, shapes, label):
    print(f"\n{'='*74}\n{label}\n{'='*74}")
    print(f"{'Nb':>5} {'s':>5} {'p':>5} {'torch':>9} {'cublas':>9} "
          f"{'mkl':>9}  {'speedup':>8}  {'err':>9}")
    print(f"{'':>17} {'ms/mat':>9} {'ms/mat':>9} {'ms/mat':>9}")

    tot_torch = tot_cublas = tot_mkl = 0.0

    for nb, s, p in shapes:
        try:
            err = check(handle, s, p)
        except RuntimeError as e:
            print(f"{nb:5d} {s:5d} {p:5d}   cublas unavailable: {e}")
            continue

        W = torch.randn(nb, s, p, dtype=DTYPE, device=DEV)

        t_torch = timed(lambda: torch.linalg.qr(W, mode="r"),
                        torch.cuda.synchronize, reps=5)

        # geqrf overwrites its input, so the loop refactors already-factored
        # data.  Identical flop count, so the timing is unaffected.
        A = W.mT.contiguous()
        k = min(s, p)
        tau = torch.zeros(nb, k, dtype=DTYPE, device=DEV)
        a_ptrs = _ptr_array(A, p * s * A.element_size())
        t_ptrs = _ptr_array(tau, k * tau.element_size())
        t_cublas = timed(
            lambda: geqrf_batched(handle, A, tau, a_ptrs, t_ptrs),
            torch.cuda.synchronize, reps=5)

        if INCLUDE_CPU:
            Wc = W.cpu()
            t_mkl = timed(lambda: torch.linalg.qr(Wc, mode="r"),
                          lambda: None, reps=3)
        else:
            t_mkl = float("nan")

        tot_torch += t_torch
        tot_cublas += t_cublas
        tot_mkl += t_mkl if INCLUDE_CPU else 0.0

        f = lambda t: t / nb * 1000
        print(f"{nb:5d} {s:5d} {p:5d} {f(t_torch):9.4f} {f(t_cublas):9.4f} "
              f"{f(t_mkl):9.4f}  {t_torch/t_cublas:7.2f}x  {err:9.1e}")

        del W, A, tau, a_ptrs, t_ptrs
        torch.cuda.empty_cache()

    print(f"\n  per-construct totals (sum over levels):")
    print(f"    torch   {tot_torch*1000:8.1f} ms")
    print(f"    cublas  {tot_cublas*1000:8.1f} ms   "
          f"({tot_torch/tot_cublas:.2f}x)")
    if INCLUDE_CPU:
        print(f"    mkl     {tot_mkl*1000:8.1f} ms   "
              f"({tot_torch/tot_mkl:.2f}x)")

    print(f"\n  If null_qr is 130.0 s of a 235 s factorization, a "
          f"{tot_torch/tot_cublas:.2f}x on the QR gives")
    qr_new = 130.0 / (tot_torch / tot_cublas)
    print(f"  null_qr {qr_new:.1f} s and a total near "
          f"{235.0 - 130.0 + qr_new:.0f} s "
          f"({(235.0)/(235.0 - 130.0 + qr_new):.2f}x end to end).")


def main():
    handle = ctypes.c_void_p()
    if cublas.cublasCreate_v2(ctypes.byref(handle)) != 0:
        raise RuntimeError("cublasCreate_v2 failed")
    try:
        run(handle, LEVEL_SHAPES, "rk=128, nl=128  (Ntot=1048576, measured)")
        run(handle, LEVEL_SHAPES_RK256, "rk=256, nl=256  (Helmholtz target)")
        print("\nNote: MKL uses all cores here.  Running the host QR "
              "concurrently with\nGPU work will not reach this throughput.")
    finally:
        cublas.cublasDestroy_v2(handle)


if __name__ == "__main__":
    main()
