"""
Batched QR of many SMALL matrices: torch/cuSOLVER vs cublasDgeqrfBatched vs MKL.

Context.  At (Nb, 394, 512) every CUDA path in PyTorch and CuPy loops over the
batch calling an unbatched geqrf -- 2.07 ms/matrix regardless of Nb.
cublasDgeqrfBatched does batch: 4.64 ms/matrix at Nb=128, 1.46 at Nb=512,
1.27 at Nb=1024, then 1.68 at Nb=2048, i.e. it bottoms out around Nb=1024 and
degrades past it.  Best case 1.62x over torch, while MKL on the host does the
same Nb=512 batch faster than either.

NVIDIA documents geqrfBatched as tuned for SMALL matrices, and 394x512 is not
small.  This script asks whether the picture changes at 100x100, where the
routine is in its intended regime and more matrices fit on the device at once.

Two questions:
  1. Does cuBLAS's optimum move to a larger Nb?  Smaller matrices occupy fewer
     resources per thread block, so the Nb=1024 turning point should shift up.
     If Nb=2048 is still improving, extend the sweep.
  2. Does the CPU still win?  If MKL beats cuBLAS at 100x100 as well as at
     394x512, then no CUDA path is competitive at any shape of interest and
     host-QR-overlapped-with-GPU-work is the answer rather than the fallback.

LAYOUT.  cuBLAS is column-major.  A contiguous torch (Nb, P, S) tensor read
column-major is a batch of S-by-P matrices with lda = S.  Getting this
backwards still returns info=0 while factoring the transpose, which is why
correctness is checked before any timing is reported.
"""

import ctypes
import time

import torch

S = P = 256                 # matrix dimensions
BATCHES = (128,256,512)
DTYPE = torch.float64
DEV = "cuda"


# ---------------------------------------------------------------------------
# cuBLAS
# ---------------------------------------------------------------------------
def load_cublas():
    """PyTorch has already dlopen'd cuBLAS once CUDA is initialised, so the
    process symbol table is the most portable handle.  'libcublas.so' is a
    toolkit-only devel symlink absent from pip wheels, which ship
    'libcublas.so.12' under site-packages/nvidia/cublas/lib."""
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
    """Device-resident array of pointers to each matrix in a batch."""
    return (torch.arange(t.shape[0], device=t.device, dtype=torch.int64)
            * stride_bytes + t.data_ptr())


def geqrf_batched(handle, A):
    """In-place batched QR.  A is contiguous (Nb, P, S) torch, read
    column-major as a batch of S-by-P matrices with lda = S."""
    nb, p, s = A.shape
    assert A.is_contiguous()
    k = min(s, p)

    tau = torch.zeros(nb, k, dtype=A.dtype, device=A.device)
    a_ptrs = _ptr_array(A, p * s * A.element_size())
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
                           f"({CUBLAS_STATUS.get(status, '?')})")
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


# ---------------------------------------------------------------------------

def timed(fn, sync, reps):
    fn()
    sync()
    t = time.time()
    for _ in range(reps):
        fn()
    sync()
    return (time.time() - t) / reps


def main():
    handle = ctypes.c_void_p()
    if cublas.cublasCreate_v2(ctypes.byref(handle)) != 0:
        raise RuntimeError("cublasCreate_v2 failed")

    try:
        g = torch.Generator(device=DEV).manual_seed(0)

        # ---- correctness at this shape, before trusting any timing -------
        # |R| rather than R: QR factors agree only up to column signs, and
        # every downstream consumer here is sign-invariant.
        Wr = torch.randn(2, S, P, generator=g, dtype=DTYPE, device=DEV)
        A = Wr.mT.contiguous().clone()
        geqrf_batched(handle, A)
        Rt = torch.linalg.qr(Wr, mode="r").R
        Rc = r_from_packed(A)[:, :Rt.shape[1], :]
        rel = ((Rt.abs() - Rc.abs()).abs().max()
               / Rt.abs().max()).item()
        print(f"\n{S}x{P}, relative |R| error: {rel:.2e}")
        if rel > 1e-10:
            print("MISMATCH -- layout is wrong; timings below are meaningless.")

        # ---- timing ------------------------------------------------------
        print(f"\n{'Nb':>6} {'torch-gpu':>11} {'cublas':>11} {'mkl-cpu':>11}"
              f"   ms/matrix")
        for nb in BATCHES:
            W = torch.randn(nb, S, P, generator=g, dtype=DTYPE, device=DEV)
            Wc = W.cpu()

            t_gpu = timed(lambda: torch.linalg.qr(W, mode="r"),
                          torch.cuda.synchronize, reps=5)

            # geqrf overwrites its input, so the loop refactors already
            # factored data.  Identical flop count, so timing is unaffected.
            A = W.mT.contiguous()
            t_cub = timed(lambda: geqrf_batched(handle, A),
                          torch.cuda.synchronize, reps=5)

            t_cpu = timed(lambda: torch.linalg.qr(Wc, mode="r"),
                          lambda: None, reps=3)

            f = lambda t: t / nb * 1000
            print(f"{nb:6d} {f(t_gpu):8.4f}    {f(t_cub):8.4f}    "
                  f"{f(t_cpu):8.4f}")

        print("\nMKL uses all cores here; running host QR concurrently with "
              "GPU work will not reach this throughput.")

    finally:
        cublas.cublasDestroy_v2(handle)


if __name__ == "__main__":
    main()
