import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.linalg as tla
import matAssembly.HBS.ULVsparse_torch as ULVsparse

# ---------------------------------------------------------------------------
# module knobs
# ---------------------------------------------------------------------------
_UV_TQR    = [0.0]      # factorization of the sample pair (Cholesky or QR)
_UV_TBASIS = [0.0]      # extraction of U from the projected samples
_UV_TSETUP = [0.0]      # forming the Gram/cross products, or the QR arena
_UV_NCALL  = [0]
_UV_SYNC   = [False]
_MATMAT_CHUNK = [256]
_EIGH_CHUNK   = [128]   # upper bound on the eigh batch; 0 means "no cap"
_EIGH_BYTES   = [1 << 30]  # and a byte budget for it, which is what actually
                        # binds: syevd's workspace grows with the batch size
                        # AND with the matrix size, so a fixed count that fits
                        # at one ny asks for gigabytes at twice that ny
_UV_ORTH_PASSES = [2]   # CholeskyQR passes in the fast path; 1 skips the
                        # reorthogonalization and reinstates a cond(Om)^2 leak
_PIN_HOST     = [True]  # page-lock host masters; set False if host RAM is tight
_SOLVE_CHUNK  = [512]   # columns per ULV solve in HBSMAT.solve; 0 disables
_UV_BASIS     = ['svd_dev']  # fast-path basis extraction:
                        #   'svd_host'  SVD of L = RB^T (ny x ny) on the host,
                        #               batch-parallel, one LAPACK thread each
                        #   'svd_dev'   same SVD on the device (torch.linalg.svd)
                        #   'jw'        previous Jordan-Wielandt eigh (2ny x 2ny)
                        #               on the device, kept for A/B comparison
_SVD_WORKERS  = [0]     # host threads for 'svd_host'; 0 means the CPUs this process may use
_SVD_POOL     = [None, 0]   # (executor, its worker count), created lazily
_SVD_DEV_DRIVER = ['gesvd']  # for 'svd_dev': 'gesvd' or 'gesvdj', never 'gesvda'
_UV_CHUNK_ELEMS = [1<<28]  # cap on c*max(n,ny)*s per compute_UV_pair call,
                        # i.e. per batched cuSOLVER/cuBLAS/MAGMA launch.  Blocks
                        # are independent, so chunking is exact.  0 disables.


def _dev_eq(a, b):
    """Device equality that treats 'cuda' and 'cuda:<current>' as equal.
    torch.device('cuda') != torch.device('cuda:0') under ==, and the RB solver
    assigns an indexed compute_device to blocks built with an unindexed one."""
    a, b = torch.device(a), torch.device(b)
    if a.type != b.type:
        return False
    if a.type != 'cuda':
        return True
    cur = torch.cuda.current_device()
    return ((a.index if a.index is not None else cur) ==
            (b.index if b.index is not None else cur))


class _HostBuffer:
    """Flat host buffer holding one group's master copy, page-locked in place.

    Allocated pageable at the exact size and registered with cudaHostRegister
    instead of torch.empty(..., pin_memory=True): the caching host allocator
    rounds each request up to a power of two and keeps freed blocks cached, so
    for multi-GB factor groups pin_memory can nearly double host usage.

    Lifetime: __del__ unregisters while self.t still holds the storage, so the
    memory is never freed while registered.  Views that outlive this object
    stay valid; they are merely no longer page-locked.  If registration is
    unavailable (ROCm, driver refusal, _PIN_HOST off) the buffer is pageable:
    evicts stay copy-free, uploads become synchronous.
    """
    __slots__ = ('t', 'registered')

    def __init__(self, numel, dtype):
        self.t = torch.empty(numel, dtype=dtype)
        self.registered = False
        if _PIN_HOST[0] and numel > 0 and torch.cuda.is_available():
            try:
                rc = torch.cuda.cudart().cudaHostRegister(
                    self.t.data_ptr(), self.t.numel() * self.t.element_size(), 0)
                self.registered = (rc is None) or (int(rc) == 0)
            except Exception:
                self.registered = False

    def __del__(self):
        if getattr(self, 'registered', False):
            try:
                torch.cuda.cudart().cudaHostUnregister(self.t.data_ptr())
            except Exception:
                pass


def uv_timers_reset():
    _UV_TQR[0] = _UV_TBASIS[0] = _UV_TSETUP[0] = 0.0
    _UV_NCALL[0] = 0


def uv_timers():
    return dict(qr=_UV_TQR[0], basis=_UV_TBASIS[0],
                setup=_UV_TSETUP[0], ncall=_UV_NCALL[0])


def _uv_sync(device):
    if _UV_SYNC[0] and torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# block arithmetic
# ---------------------------------------------------------------------------

def to_block_tensor(M, n, b):
    """(n*b, s) -> (n, b, s) block tensor."""
    return M.reshape(n, b, M.shape[1])


def _rsolve(P, B, rtol=None):
    """Batched min-residual solution of  X @ B = P   (X = P B^+).

    B: (Nb, n, s) with s >= n.  P: (Nb, ny, s).  Returns (Nb, ny, n).
    SVD applied factor-by-factor with an explicit relative cutoff; the (s, n)
    pseudo-inverse is never built."""
    assert B.shape[-1] >= B.shape[-2], "undersampled: s < n, a different problem"
    U, S, Vh = tla.svd(B, full_matrices=False)
    if rtol is None:
        rtol = max(B.shape[-1], B.shape[-2]) * torch.finfo(B.dtype).eps
    Sinv = torch.where(S > rtol * S[..., :1], S.reciprocal(), torch.zeros_like(S))
    return torch.bmm(torch.bmm(P, Vh.mT) * Sinv.unsqueeze(-2), U.mT)


def block_solve_r(A, B, device=None, fast=False):
    """X[i] = A[i] @ pinv(B[i]), batched over the block dim.

    `device` and `fast` are accepted for call-site compatibility and ignored:
    the tensors carry their own device, and the QR variant of the pinv was
    never enabled."""
    return _rsolve(A, B)


def block_mult(A, B, device=None, mode='N'):
    """A: (Nb, n, k), B: (Nb, k, m)."""
    if mode == 'N':
        return torch.bmm(A, B)
    elif mode == 'T':
        return torch.bmm(A.mT, B)
    raise ValueError("mode not recognized")


def block_matvec(A, B, device=None, mode='N'):
    """A: (Nb, n, k) block diagonal; B: (Nb*nB, col) flat."""
    Nb  = A.shape[0]
    col = B.shape[1]
    Bm  = B.reshape(Nb, B.shape[0] // Nb, col)
    if mode == 'N':
        return torch.bmm(A, Bm).reshape(Nb * A.shape[1], col)
    elif mode == 'T':
        return torch.bmm(A.mT, Bm).reshape(Nb * A.shape[2], col)
    raise ValueError("mode not recognized")


def block_mult_and_reduce(A, B, fac, device=None, mode='N'):
    """Block-wise product, then group fac consecutive blocks into one.

    mode='N': A (Nb, n, rk), B (Nb, rk, s) -> (Nb//fac, fac*n,  s)
    mode='T': A (Nb, n, rk), B (Nb, n,  s) -> (Nb//fac, fac*rk, s)"""
    Nb = A.shape[0]
    if mode == 'N':
        C = torch.bmm(A, B)
        return C.reshape(Nb // fac, fac * A.shape[1], B.shape[2])
    elif mode == 'T':
        C = torch.bmm(A.mT, B)
        return C.reshape(Nb // fac, fac * A.shape[2], B.shape[2])
    raise ValueError("mode not recognized")


def _svd_worker_init():
    # Per-thread state: with an OpenMP backend the intra-op thread count is a
    # per-thread ICV, so each pool thread pins its own to 1 once.  These
    # threads do nothing else, so it is never restored.
    torch.set_num_threads(1)


def _svd_pool():
    nw = _SVD_WORKERS[0] or len(os.sched_getaffinity(0))
    if _SVD_POOL[0] is None or _SVD_POOL[1] != nw:
        if _SVD_POOL[0] is not None:
            _SVD_POOL[0].shutdown(wait=True)
        _SVD_POOL[0] = ThreadPoolExecutor(max_workers=nw,
                                          initializer=_svd_worker_init)
        _SVD_POOL[1] = nw
    return _SVD_POOL[0]


def _left_sv_jw_1(Rb, k):
    """Fallback for one block: Jordan-Wielandt eigh on the host, same as the
    'jw' path.  Returns the top halves (u/sqrt(2)); the renormalization and
    reorthogonalization in compute_UV_pair restore unit columns."""
    ny = Rb.shape[-1]
    H = Rb.new_zeros((2 * ny, 2 * ny))
    H[:ny, ny:] = Rb.mT
    H[ny:, :ny] = Rb
    return tla.eigh(H).eigenvectors[:, -k:].flip(-1)[:ny, :]


def _left_sv_host(RB, k):
    """Top-k left singular vectors of L = RB^T, per block, on the host, in
    torch.

    RB: (Nb, ny, ny) upper triangular with Bp Bp^T = RB^T RB, so the left
    singular vectors of Bp are exactly those of L.  Returns (Nb, ny, k) on
    RB's device, columns ordered by decreasing sigma.

    Golub-Kahan bidiagonalization of L (torch CPU svd = LAPACK gesdd) is
    backward stable to O(u ||L||), the same accuracy class as eigh of the
    Jordan-Wielandt form, at size ny instead of 2ny.

    Threading.  torch releases the GIL inside ops, so Python threads run the
    SVDs concurrently.  The batch is split into one contiguous chunk per
    worker, and each worker issues one batched CPU svd on its chunk (torch
    loops the chunk internally, one LAPACK call per block).  One LAPACK
    thread per block:
      - main thread: torch.set_num_threads(1) around the map, which also
        sets MKL's process-wide count; restored afterwards;
      - worker threads: torch.set_num_threads(1) once, in _svd_worker_init,
        for the per-thread OpenMP setting.

    If a chunk raises LinAlgError (gesdd non-convergence), that chunk is
    redone block by block, and any block that fails again falls back to the
    host Jordan-Wielandt eigh.
    """
    Rh = RB.detach().to('cpu')                # ~ny^2*8 bytes per block
    Nb, ny, _ = Rh.shape
    out = torch.empty((Nb, ny, k), dtype=Rh.dtype)

    def work(lo, hi):
        L = Rh[lo:hi].mT
        try:
            out[lo:hi] = tla.svd(L, full_matrices=False).U[..., :k]
        except torch.linalg.LinAlgError:
            for i in range(lo, hi):
                try:
                    out[i] = tla.svd(Rh[i].mT, full_matrices=False).U[:, :k]
                except torch.linalg.LinAlgError:
                    out[i] = _left_sv_jw_1(Rh[i], k)

    pool = _svd_pool()
    nw = min(_SVD_POOL[1], Nb)
    bounds = [(Nb * j) // nw for j in range(nw + 1)]
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        futs = [pool.submit(work, bounds[j], bounds[j + 1])
                for j in range(nw) if bounds[j + 1] > bounds[j]]
        for f in futs:
            f.result()
    finally:
        torch.set_num_threads(prev)
    return out.to(RB.device)


def _left_sv_dev(RB, k):
    """Same quantity as _left_sv_host, computed on RB's device with
    torch.linalg.svd.  Driver from _SVD_DEV_DRIVER: 'gesvd' (QR iteration,
    most robust) or 'gesvdj' (Jacobi, also accurate).  Never 'gesvda', which
    is approximate.  The driver argument is CUDA-only, so it is ignored on CPU.
    """
    drv = _SVD_DEV_DRIVER[0] if RB.device.type == 'cuda' else None
    U = tla.svd(RB.mT, full_matrices=False, driver=drv).U
    return U[..., :k].contiguous()


def _cholqr2(X, need_q=True):
    """CholeskyQR of X^T, repeated.  X: (Nb, m, s) with s >= m.

    Returns (Q, R) with R (Nb, m, m) upper triangular and Q (Nb, m, s) having
    orthonormal ROWS, so that  X = R^T Q  and  X X^T = R^T R.  Q is the
    transpose of the usual thin factor of X^T; keeping it row-major in the
    s direction avoids transposing an (s, m) tensor at every use.

    Two passes give ||Q Q^T - I|| = O(u) and R accurate to O(u ||X||) as long
    as cond(X) <= u^{-1/2}, i.e. the same backward stability as Householder
    QR, at GEMM/TRSM throughput instead of batched geqrf.  A failed Cholesky
    means cond(X) is past that bound; the Gram is then shifted by
    11(ms + m(m+1)) u ||X||_F^2 and a third pass is run, which extends the
    bound to cond(X) <= 1/u.

    need_q=False skips the final triangular solve when only R is wanted,
    saving one m^2 s TRSM.

    Costs one D2H sync per pass, from the Cholesky info check.
    """
    Nb, m, s = X.shape
    assert s >= m, f"_cholqr2: need s >= m, got s={s}, m={m}"
    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    npass = max(1, _UV_ORTH_PASSES[0])
    Q, R, p = X, None, 0
    while p < npass:
        # G is symmetric to roundoff by construction and cholesky_ex reads one
        # triangle, so the usual 0.5*(G + G^T) buys nothing and costs a second
        # full (Nb, m, m) tensor -- 786 MB at the leaf with Nb=128, m=800,
        # twice per call.
        G = torch.bmm(Q, Q.mT)
        Lc, info = torch.linalg.cholesky_ex(G)
        if bool(info.any()):
            sh = (11 * (m * s + m * (m + 1)) * torch.finfo(X.dtype).eps
                  * torch.diagonal(G, dim1=-2, dim2=-1).sum(-1))
            Lc, info = torch.linalg.cholesky_ex(G + sh[:, None, None] * eye)
            if bool(info.any()):
                raise torch.linalg.LinAlgError(
                    "_cholqr2: shifted Cholesky failed; the sample block is "
                    "rank-deficient past cond ~ 1/u")
            npass = max(npass, p + 3)     # shifted CholeskyQR3
        del G
        Ri = Lc.mT                        # G = Ri^T Ri, Ri upper triangular
        del Lc
        p += 1
        if need_q or p < npass:
            # After the first pass Q is this function's own buffer, so the
            # solve writes into it instead of allocating another (Nb, m, s):
            # 1.6 GB at the leaf with s=1916.  TRSM is in-place on its
            # right-hand side anyway, so passing out=Q is the intended path
            # and not a copy.  The first pass must NOT do this -- Q is still
            # the caller's X there, and for the A side that is Om or Psi,
            # which the rest of the level reads.
            Q = torch.linalg.solve_triangular(
                Ri.mT, Q, upper=False, left=True)
        R = Ri if R is None else torch.bmm(Ri, R)
    return Q, R


def construct_D(U, V, M_om, M_psi):
    """D = (I - U U*) Y Om^+  +  U [ (I - V V*) Z Psi^+ ]* U*

    with M_om = Y Om^+ and M_psi = Z Psi^+ supplied by compute_UV_pair, so
    this is four GEMMs and nothing else -- no triangular solves, no SVD
    fallback, no width-s intermediate.  Both branches of compute_UV_pair
    return M in this form; the QR branch gets there via
    R_oy^T R_oo^{-T} = Y Om^T (R_oo^T R_oo)^{-1} = Y Om^+."""
    P  = M_om - torch.bmm(U, torch.bmm(U.mT, M_om))     # (Nb, ny, n)
    Gk = torch.bmm(M_psi, U)                            # (Nb, ny, k)
    Gk = Gk - torch.bmm(V, torch.bmm(V.mT, Gk))
    return P + torch.bmm(U, Gk.mT)


def compute_UV_pair(Om, Y, Psi, Z, rk, device=None, fast=False):
    """Both halves of a level's basis computation.  `device` is accepted for
    call-site compatibility and ignored; the tensors carry their own.

    Returns ((U, M_om), (V, M_psi)) with U the leading k left singular
    vectors of Y (I - Om^+ Om) and M_om = Y Om^+ (and likewise V, M_psi from
    Psi, Z).

    fast=True   CholeskyQR2 on Om^T for the projector, then the singular
                vectors of the projected samples via the Jordan-Wielandt
                eigenproblem:

                    Om^T = Q1 R          CholeskyQR2, ||Q1^T Q1 - I|| = O(u)
                    M    = (Y Q1) R^{-T}                        = Y Om^+
                    Bp   = Y - (Y Q1) Q1^T                      = Y P_tau P_tau^T
                    Bp   = RB^T QB       CholeskyQR2, R factor only
                    U    = top-k left singular vectors of RB^T
                           (_UV_BASIS 'svd_host': host SVD, default), or
                           top half of top-k eigenvectors of
                           [[0, RB^T], [RB, 0]]  (_UV_BASIS 'jw')

                Nothing is squared: the Grams are formed only to orthogonalize,
                never to have their spectrum read.  The extraction resolves
                sigma_j down to u sigma_1 rather than sqrt(u) sigma_1.

                The second CholeskyQR pass on Om^T is not decorative.  With Q1
                orthonormal only to eps, I - Q1 Q1^T is not a projector and Bp
                keeps a leak of size eps ||Y|| lying in row(Om) -- the very
                component the projection removes.  One pass gives
                eps ~ u cond(Om)^2 ~ 100u; two give eps = O(u).  Set
                _UV_ORTH_PASSES[0] = 1 to measure what that is worth.

                What remains is the cancellation floor u ||Y||/||Bp||, common
                to every formulation in fixed precision (Householder QR on
                [Om^T | Y^T] included) and expected to be O(10) u here.

    fast=False  Householder QR of W = [Om^T | Y^T] = Q R, then

                    M = R_oy^T R_oo^{-T},  L = R[n:, n:]^T,  U = svd(L).U

                Q is never formed: nothing downstream needs it.  The SVD runs
                on the host, which is why this path is slow.  Kept as the
                reference the fast path is validated against.

    The two sides run sequentially rather than stacked into a 2Nb batch.
    Stacking was measured at zero speedup (null_qr 130.01 s either way, since
    a per-matrix loop just loops twice as long) and it doubles peak memory.
    """
    Nb, ny, s = Y.shape
    n = Om.shape[1]
    k = min(rk, ny)
    assert Psi.shape == Om.shape and Z.shape == Y.shape, \
        "compute_UV_pair needs matching shapes"
    assert s >= n + k, \
        "undersampled: not enough columns left after projecting off Om"
    _UV_NCALL[0] += 2
    dev = Y.device
    out = []

    for A, B in ((Om, Y), (Psi, Z)):
        if fast:
            # ---- P_tau as a projector: Q1 = orth(A^T), never a null basis --
            _uv_sync(dev); _t = time.time()
            QA, RA = _cholqr2(A)                  # A = RA^T QA, QA rows orthon.
            _uv_sync(dev); _UV_TQR[0] += time.time() - _t

            # ---- B P_tau P_tau^T and M = B A^+ ----------------------------
            _uv_sync(dev); _t = time.time()
            C  = torch.bmm(B, QA.mT)              # (Nb, ny, n) = Y Q1
            # A^+ = Q1 RA^{-T}, so M = C RA^{-T}: one triangular solve, no
            # normal equations and no cond(A)^2.
            M  = torch.linalg.solve_triangular(
                RA.mT, C, upper=False, left=False).contiguous()
            Bp = torch.baddbmm(B, C, QA, beta=1.0, alpha=-1.0)   # B - C Q1^T
            del C, QA, RA
            _uv_sync(dev); _UV_TSETUP[0] += time.time() - _t

            # ---- top-k left singular vectors of Bp, without squaring -------
            _uv_sync(dev); _t = time.time()
            _, RB = _cholqr2(Bp, need_q=False)    # Bp Bp^T = RB^T RB
            del Bp
            if _UV_BASIS[0] == 'svd_host':
                # Left singular vectors of Bp are those of L = RB^T (ny x ny).
                # Direct SVD of L never forms the Gram, so like the JW form
                # below it resolves sigma_j down to ~u sigma_1, but the dense
                # reduction is at size ny instead of 2ny and runs on the host
                # (see _left_sv_host for why host + batch-parallel).
                UU = _left_sv_host(RB, k)
            elif _UV_BASIS[0] == 'svd_dev':
                UU = _left_sv_dev(RB, k)
            else:
                # Left singular vectors of Bp are those of L = RB^T, and they live
                # in R^ny: the s direction is already gone.  Extract them from the
                # Jordan-Wielandt form
                #     H = [[0, L], [L^T, 0]],   eig(H) = +-sigma,
                #     eigenvectors (u, +-v)/sqrt(2),
                # which is assembled by block placement -- exactly, with no
                # arithmetic -- and has ||H|| = sigma_1.  A backward-stable eigh
                # therefore resolves directions down to sigma_j ~ u sigma_1, where
                # eigh(Bp Bp^T) loses everything below sqrt(u) sigma_1 to the
                # rounding error of forming the Gram.
                # The eigenproblem is 2ny, so its workspace is 4x the ny version's
                # at equal batch size -- dividing a fixed chunk COUNT by 4 does not
                # track that, because the count itself was tuned at some other ny.
                # Budget bytes: ~6x the batch covers H, the eigenvector output and
                # cusolver's own scratch.  At ny=400 this gives the old chunk back;
                # at ny=800 it backs off to a quarter of it, which is the whole
                # point.
                per = 6 * (2 * ny) ** 2 * RB.element_size()
                c   = min(_EIGH_CHUNK[0] or Nb, Nb)
                if _EIGH_BYTES[0]:
                    c = min(c, _EIGH_BYTES[0] // per)
                c = max(1, c)
                UU = torch.empty((Nb, ny, k), dtype=RB.dtype, device=dev)
                H  = torch.zeros((min(c, Nb), 2 * ny, 2 * ny),
                                 dtype=RB.dtype, device=dev)
                for j in range(0, Nb, c):
                    blk = RB[j:j+c]
                    Hv  = H[:blk.shape[0]]
                    Hv[:, :ny, ny:] = blk.mT
                    Hv[:, ny:, :ny] = blk
                    UU[j:j+c] = tla.eigh(Hv).eigenvectors[..., -k:].flip(-1)[:, :ny, :]
                del H
            del RB
            # Kept unchanged for both paths.  For 'svd_host' the columns are
            # already orthonormal to O(u) and this is a near no-op (the
            # optional fix #4 would drop it there).  For 'jw':
            # each column arrives as the top half of (u, v)/sqrt(2).  The two
            # halves separate cleanly for distinct positive sigma; a +-pair is
            # degenerate only where sigma ~ 0, so clamp before dividing and
            # then restore orthonormality outright -- ULVsparse.compute_QRW_sparse
            # builds the complement W1 from V and needs V^T V = I.
            UU = UU / UU.norm(dim=-2, keepdim=True).clamp_min(
                torch.finfo(UU.dtype).tiny)
            Gk = torch.bmm(UU.mT, UU)             # (Nb, k, k), cheap
            Lk, info = torch.linalg.cholesky_ex(Gk)   # symmetric by construction
            if not bool(info.any()):
                UU = torch.linalg.solve_triangular(Lk.mT, UU, upper=True,
                                                   left=False)
            del Gk, Lk
            UU = UU.contiguous()
            _uv_sync(dev); _UV_TBASIS[0] += time.time() - _t
        else:
            _uv_sync(dev); _t = time.time()
            W = torch.empty((Nb, s, n + ny), dtype=B.dtype, device=dev)
            W[:, :, :n].copy_(A.mT)
            W[:, :, n:].copy_(B.mT)
            _uv_sync(dev); _UV_TSETUP[0] += time.time() - _t

            _uv_sync(dev); _t = time.time()
            R = tla.qr(W, mode='r').R          # (Nb, r, n+ny), r = min(s, n+ny)
            del W                              # largest tensor here; drop it
            # R_oo is upper triangular and well conditioned for Gaussian Om,
            # so the pinv of Om^T is a triangular solve.
            M = torch.linalg.solve_triangular(
                R[:, :n, :n].mT, R[:, :n, n:].mT, upper=False,
                left=False).contiguous()       # (Nb, ny, n) = Y Om^+
            _uv_sync(dev); _UV_TQR[0] += time.time() - _t

            _uv_sync(dev); _t = time.time()
            L = R[:, n:, n:].mT                # (Nb, ny, r-n)
            UU = tla.svd(L.to('cpu'), full_matrices=False).U[..., :k]
            UU = UU.to(dev).contiguous()
            del R, L
            _uv_sync(dev); _UV_TBASIS[0] += time.time() - _t

        out.append((UU, M))

    return out[0], out[1]


def compute_UV_pair_chunked(Om, Y, Psi, Z, rk, device=None, fast=False):
    """compute_UV_pair split along the block (batch) dimension.

    Every quantity in compute_UV_pair is computed per block, so running it on
    slices of the batch gives the same result.  Keeping each batched library
    call below _UV_CHUNK_ELEMS elements avoids 32-bit size/offset limits in
    batched potrf/trsm at large Nb, and also bounds the peak of the
    Q / Bp / Gram temporaries.
    """
    Nb, ny, s = Y.shape
    n = Om.shape[1]
    cap = _UV_CHUNK_ELEMS[0]
    c = Nb if not cap else max(1, min(Nb, cap // max(1, max(n, ny) * s)))
    if c >= Nb:
        return compute_UV_pair(Om, Y, Psi, Z, rk, device, fast=fast)
    U = V = Mo = Mp = None
    for j in range(0, Nb, c):
        sl = slice(j, j + c)
        (u, mo), (v, mp) = compute_UV_pair(Om[sl], Y[sl], Psi[sl], Z[sl],
                                           rk, device, fast=fast)
        if U is None:
            U  = u.new_empty((Nb,) + tuple(u.shape[1:]))
            V  = v.new_empty((Nb,) + tuple(v.shape[1:]))
            Mo = mo.new_empty((Nb,) + tuple(mo.shape[1:]))
            Mp = mp.new_empty((Nb,) + tuple(mp.shape[1:]))
        U[sl], V[sl], Mo[sl], Mp[sl] = u, v, mo, mp
        del u, v, mo, mp
    return (U, Mo), (V, Mp)


class HBSMAT:
    """HBS matrix: compression, apply, and ULV solve.

    @init:      linear operator A, tree on DOFs (symmetric), target rank k
    @constructs HBS approximation to the source-target map
    @implements matvec/matmat (normal and transpose) and solve

    Device policy
    -------------
    Every tensor the object stores lives on self.compute_device.  There is no
    per-level exception, so any consumer that takes a whole list
    (ULVsparse.solve, ULVsparse.compute_ULV) sees a list on one device.  Use
    to()/cpu() to relocate the whole object; that is the only supported way to
    trade VRAM for host memory.

    Level ordering: the construction loop runs L-1 down to 0, so Dmats[0] is
    the leaf level and Dmats[-1] is the ROOT.
    """

    # every attribute holding a list of tensors; to() walks these.
    # Nbvec is a list of ints, so it stays put.
    _GROUPS = {
        'core': ('Umats', 'Vmats', 'Dmats'),
        'ulv' : ('Qlist', 'Wlist', 'Rlist', 'Uulist'),
    }

    def __init__(self, A=None, device=None, tree=None, quad=False):
        # perm lives in a one-element box so that .T views (shallow __dict__
        # copies) see the same object when residency rebinds it.
        self._permbox = [None]
        self.Umats  = []
        self.Vmats  = []
        self.Dmats  = []
        self.Qlist  = []
        self.Rlist  = []
        self.Wlist  = []
        self.Uulist = []
        torch.set_default_dtype(torch.float64)
        self.dtype   = np.float64
        self.dtype_t = torch.float64

        self.mode  = 'N'
        self._tree = None

        dev = torch.device(device) if device is not None else torch.device('cpu')
        self.compute_device = dev
        self.home           = torch.device('cpu')
        self._resident = {'core': dev, 'ulv': dev}
        # host master per group: None until the first evict snapshots it
        self._host     = {'core': None, 'ulv': None}
        self.strict    = False
        self.nFill = self.nSpill = 0
        self.bytesH2D = self.bytesD2H = 0

        if A is not None:
            self.A     = A
            self.shape = A.shape
            self.dtype = A.dtype

        if tree is not None:
            self.tree  = tree
            self.perm  = tree.perm_leaf
            self.Nb    = tree.nleaves
            self.nl    = len(self.perm) // self.Nb
            self.L     = tree.nlevels
            self.shape = (len(self.perm), len(self.perm))

        self.blockSolveTime = 0
        self.nullTime   = 0
        self.setupTime  = 0
        self.DTime      = 0
        self.tSample    = 0
        self.tConstruct = 0
        self.tULV       = 0
        self.tCompress  = 0
        self.Nbvec = []
        self.quad  = quad
        self.fac   = 4 if quad else 2

    def set_Nbvec(self, Nbvec):
        self.Nbvec = Nbvec
        self.Nb = Nbvec[0]
        self.nl = self.A.shape[0] // self.Nb
        self.L  = len(Nbvec)
        self.perm = torch.arange(self.A.shape[0], dtype=torch.int64)

    def set_mats(self, Umats, Dmats, Vmats, Nbvec, fac=4):
        self.Umats = Umats
        self.Dmats = Dmats
        self.Vmats = Vmats
        self._host['core'] = None
        self._resident['core'] = Dmats[0].device if torch.is_tensor(Dmats[0]) else None
        self.perm  = torch.arange(Dmats[0].shape[0])
        self.fac   = fac
        self.Nb    = Nbvec[0]
        self.shape = torch.tensor([Dmats[0].shape[0], Dmats[0].shape[0]],
                                  dtype=torch.int64)
        self.dtype = Dmats[0][0].dtype

    @property
    def nbytes(self):
        return (sum(U.nbytes for U in self.Umats)
                + sum(V.nbytes for V in self.Vmats)
                + sum(D.nbytes for D in self.Dmats))

    @property
    def device(self):
        return str(self.compute_device)

    def _as_local(self, X):
        dev = self.compute_device
        if torch.is_tensor(X):
            return X.to(device=dev, dtype=self.dtype_t, non_blocking=True)
        return torch.from_numpy(X).to(device=dev, dtype=self.dtype_t)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def construct(self, rk, Om=None, Psi=None, Y=None, Z=None,
                  compute_ULV=False, fast=False):
        if Om is None:
            if self.A is None:
                raise ValueError("Samples and LinOP cannot both be None")
            # hard requirement max(fac*rk, nl) + rk at the worst level, plus
            # oversampling p = rk there (every other level has more)
            s = max(self.fac * rk, self.nl) + 2 * rk
            Om  = np.random.standard_normal(size=(self.A.shape[1], s))
            Psi = np.random.standard_normal(size=(self.A.shape[0], s))
            Y   = self.A @ Om
            Z   = self.A.T @ Psi
        self.constructHBS(rk, Om, Psi, Y, Z, fast=fast, with_ulv=compute_ULV)

    def constructHBS(self, rk, Om0, Psi0, Y0, Z0, fast=False, with_ulv=False):
        """Compress from samples Y = A Om, Z = A^T Psi.

        with_ulv=True interleaves the ULV factorization into the same sweep,
        which is what makes it cheap: the level-(lvl) factors are still live
        when the ULV step needs them.  The two used to be separate methods
        with identical compression bodies.
        """
        # A sample may arrive boxed -- as a one-element list -- which is the
        # caller's way of handing over its only reference.  The leaf
        # permutation below is a gather, so it allocates a second full (m, s)
        # copy while the source is still live; four of those is 5 GB at
        # m = 100k, s = 1550.  Unboxing here and clearing the source right
        # after each gather means at most one source and one copy coexist,
        # and a boxed source is freed outright instead of waiting for the
        # caller's frame to end.  Plain tensors still work: they are simply
        # not freed early.
        Om0  = Om0.pop()  if isinstance(Om0,  list) else Om0
        Psi0 = Psi0.pop() if isinstance(Psi0, list) else Psi0
        Y0   = Y0.pop()   if isinstance(Y0,   list) else Y0
        Z0   = Z0.pop()   if isinstance(Z0,   list) else Z0

        s  = Om0.shape[1]
        Nb = self.Nb
        nl = self.nl
        self.Nbvec = [Nb]
        self.nSamples = s
        self.NNvec = np.zeros(shape=(0,), dtype=np.int64)
        self.NNvec = np.append(self.NNvec, 0)
        self._reset_residency()

        tic = time.time()
        if torch.is_tensor(self.perm):
            self.perm = self.perm.to(self.compute_device)
        else:
            self.perm = torch.as_tensor(self.perm, dtype=torch.int64,
                                        device=self.compute_device)
        # advanced indexing returns a fresh contiguous tensor, so the reshape
        # to (Nb, nl, s) is a view and the deflation below may write into it.
        # Each source is dropped immediately after its gather, so the peak
        # here is one source plus one copy, not four of each.
        Om  = self._as_local(Om0)[self.perm, :].reshape(Nb, nl, s)
        Om0 = None
        Psi = self._as_local(Psi0)[self.perm, :].reshape(Nb, nl, s)
        Psi0 = None
        Y   = self._as_local(Y0)[self.perm, :].reshape(Nb, nl, s)
        Y0  = None
        Z   = self._as_local(Z0)[self.perm, :].reshape(Nb, nl, s)
        Z0  = None
        self.setupTime += time.time() - tic
        self.tSample   += time.time() - tic

        tic_compress = time.time()
        for lvl in range(self.L - 1, -1, -1):

            if lvl == self.L - 1:
                Om_ell, Psi_ell, Y_ell, Z_ell = Om, Psi, Y, Z
                rkm = min(rk, nl)
            else:
                d = self.device
                Y_ell  -= block_mult(D_ell, Om_ell, d)
                Y_ell   = block_mult_and_reduce(U_ell, Y_ell, self.fac, d, mode='T')
                Z_ell  -= block_mult(D_ell, Psi_ell, d, mode='T')
                Z_ell   = block_mult_and_reduce(V_ell, Z_ell, self.fac, d, mode='T')
                Om_ell  = block_mult_and_reduce(V_ell, Om_ell, self.fac, d, mode='T')
                Psi_ell = block_mult_and_reduce(U_ell, Psi_ell, self.fac, d, mode='T')
                Nb  = Nb // self.fac
                rkm = min(rk, nl * (self.fac ** (self.L - 1 - lvl)))
            self.Nbvec += [Nb]

            if lvl > 0:
                tic = time.time()
                (U_ell, M_om), (V_ell, M_psi) = compute_UV_pair_chunked(
                    Om_ell, Y_ell, Psi_ell, Z_ell, rkm, self.device, fast=fast)
                self.nullTime += time.time() - tic
                tic = time.time()
                D_ell = construct_D(U_ell, V_ell, M_om, M_psi)
                self.DTime += time.time() - tic
                self.Dmats += [D_ell]
                self.Umats += [U_ell]
                self.Vmats += [V_ell]
            else:
                # root: D = Y Om^+, one block, so the explicit SVD pinv with a
                # relative cutoff is affordable
                tic = time.time()
                D_ell = block_solve_r(Y_ell, Om_ell, self.device, fast=fast)
                self.blockSolveTime += time.time() - tic
                self.Dmats += [D_ell]
                if with_ulv:
                    U_ell = torch.eye(D_ell.shape[1], dtype=D_ell.dtype,
                                      device=self.compute_device)[None, :, :]

            if with_ulv:
                tic = time.time()
                dev = self.compute_device
                if lvl == self.L - 1:
                    Rhat = D_ell
                else:
                    Rhat = ULVsparse.sparse_block_mult_tens(Uhat, D_ell, device=dev)
                    Rhat = ULVsparse.block_diag_add_tens(Rhat, R_22, device=dev)

                Q, W, Ru, R_22, NN = ULVsparse.compute_QRW_sparse(
                    Rhat, V_ell, self.Nbvec[-1], device=dev)
                self.NNvec = np.append(self.NNvec, self.NNvec[-1] + NN)
                self.Qlist += [Q]
                if W is not None:
                    # W = [W1 | V_ell] and the V_ell half is Vmats[-1]
                    # verbatim.  Store only the complement; ULVsparse.solve
                    # reads V from Vmats.
                    W = W[:, :, :W.shape[2] - V_ell.shape[2]].contiguous()
                self.Wlist += [W]
                self.Rlist += [Ru]

                if lvl == self.L - 1:
                    Uhat = U_ell
                else:
                    Uhat = ULVsparse.sparse_block_mult_tens(Uhat, U_ell, device=dev)
                Uu = ULVsparse.sparse_block_mult_tens(Q[:, :, :-rkm], Uhat,
                                                      device=dev, mode='T')
                Ud = ULVsparse.sparse_block_mult_tens(Q[:, :, -rkm:], Uhat,
                                                      device=dev, mode='T')
                self.Uulist += [Uu]
                Uhat = Ud
                self.tULV += time.time() - tic

        if self.compute_device.type == 'cuda':
            torch.cuda.synchronize()
        self.tCompress = time.time() - tic_compress

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

    def __matmul__(self, v):
        if self.mode == 'N':
            return self.matmat(v)
        elif self.mode == 'T':
            return self.rmatmat(v)
        raise ValueError("mode not recognized")

    def matvec(self, v):
        return self._apply(v, transpose=False)

    def rmatvec(self, v):
        return self._apply(v, transpose=True)

    def matmat(self, v, chunk=None):
        return self._apply(v, transpose=False, chunk=chunk)

    def rmatmat(self, v, chunk=None):
        return self._apply(v, transpose=True, chunk=chunk)

    def _apply(self, v, transpose, chunk=None):
        """A v (transpose=False) or A^T v (transpose=True).

        All input normalization happens BEFORE chunking, so the chunk loop
        only ever sees a 2-D tensor on compute_device with dtype_t, and the
        return type does not depend on the column count:
            numpy in  -> numpy out (host)
            tensor in -> tensor on compute_device
            1-D in    -> 1-D out
        """
        self._require_resident('rmatmat' if transpose else 'matmat')
        if v.ndim not in (1, 2):
            raise ValueError(f"expected 1-D or 2-D input, got ndim={v.ndim}")
        numpy_input = isinstance(v, np.ndarray)
        was_vector  = (v.ndim == 1)

        V = self._as_local(v)
        if was_vector:
            V = V[:, None]

        n_in  = int(self.shape[0] if transpose else self.shape[1])
        n_out = int(self.shape[1] if transpose else self.shape[0])
        if V.shape[0] != n_in:
            raise ValueError(
                f"{'rmatmat' if transpose else 'matmat'}: expected {n_in} rows, "
                f"got {V.shape[0]}")

        c = _MATMAT_CHUNK[0] if chunk is None else chunk
        if c and V.shape[1] > c:
            out = torch.empty((n_out, V.shape[1]), dtype=V.dtype, device=V.device)
            for j0 in range(0, V.shape[1], c):
                out[:, j0:j0+c] = self._apply_2d(V[:, j0:j0+c], transpose)
        else:
            out = self._apply_2d(V, transpose)

        if was_vector:
            out = out[:, 0]
        if numpy_input:
            out = out.cpu().numpy()
        return out

    def _apply_2d(self, V, transpose):
        """Core HBS apply.  V: 2-D tensor on compute_device, original ordering.

        matmat and rmatmat differ only in which basis list runs the upward
        (restriction) sweep and in the mode used on D, so they share one body.
        """
        dmode = 'T' if transpose else 'N'
        down  = self.Umats if transpose else self.Vmats
        up    = self.Vmats if transpose else self.Umats

        VV = [V[self.perm, :]]
        for M in down:
            VV.append(block_matvec(M, VV[-1], mode='T'))
        u = block_matvec(self.Dmats[-1], VV[-1], mode=dmode)
        for lvl in range(len(up) - 1, -1, -1):
            u = block_matvec(up[lvl], u) \
              + block_matvec(self.Dmats[lvl], VV[lvl], mode=dmode)
        out = torch.zeros_like(u)
        out[self.perm, :] = u
        return out

    # ------------------------------------------------------------------
    # tree / perm
    # ------------------------------------------------------------------

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, t):
        self._tree = t

    @property
    def perm(self):
        return self._permbox[0]

    @perm.setter
    def perm(self, p):
        self._permbox[0] = p

    # ------------------------------------------------------------------
    # device placement
    # ------------------------------------------------------------------

    def _lists(self, groups):
        for g in groups:
            for name in self._GROUPS[g]:
                yield g, getattr(self, name)

    def _entries(self, g):
        """(list_name, index) of every tensor slot in group g.  perm rides
        with 'core' through its one-element box."""
        names = self._GROUPS[g] + (('_permbox',) if g == 'core' else ())
        return [(name, i) for name in names
                for i, t in enumerate(getattr(self, name)) if torch.is_tensor(t)]

    def _reset_residency(self, groups=None):
        """Forget host masters: the live tensors are about to be rebuilt on
        compute_device, so any snapshot of the old ones is stale."""
        for g in (groups if groups is not None else tuple(self._GROUPS)):
            self._host[g] = None
            self._resident[g] = self.compute_device

    def _snapshot(self, g):
        """Build the immutable host master of group g from its live tensors.

        The only device-to-host copy a group ever pays.  All tensors land in
        one flat page-locked buffer per dtype; afterwards evict() rebinds to
        views of it and prefetch() uploads from it asynchronously.  Correct
        because factors are never modified in place after construction --
        whatever rebuilds a group must drop its master first."""
        ents  = self._entries(g)
        sizes = {}
        for name, i in ents:
            t = getattr(self, name)[i]
            sizes[t.dtype] = sizes.get(t.dtype, 0) + t.numel()
        bufs = {dt: _HostBuffer(n, dt) for dt, n in sizes.items()}
        offs = dict.fromkeys(sizes, 0)
        views, devs, nD2H = [], set(), 0
        for name, i in ents:
            t  = getattr(self, name)[i]
            o  = offs[t.dtype]
            offs[t.dtype] = o + t.numel()
            hv = bufs[t.dtype].t[o:o + t.numel()].view(t.shape)
            hv.copy_(t, non_blocking=True)
            if t.is_cuda:
                devs.add(t.device)
                nD2H += t.nbytes
            views.append(hv)
        for d in devs:                  # one sync per snapshot, not per tensor
            torch.cuda.current_stream(d).synchronize()
        rec = {'ents': ents, 'views': views, 'bufs': bufs}
        self._host[g] = rec
        return rec, nD2H

    def _group_resident(self, g):
        # A group with no tensors (e.g. ulv on a with_ulv=False block) is
        # vacuously resident -- there is nothing to stage.
        if not any(len(getattr(self, n)) for n in self._GROUPS[g]):
            return True
        r = self._resident[g]
        return r is not None and _dev_eq(r, self.compute_device)

    def _require_resident(self, what, need_ulv=False):
        groups = ('core', 'ulv') if need_ulv else ('core',)
        missing = [g for g in groups if not self._group_resident(g)]
        if not missing:
            return
        if self.strict:
            raise RuntimeError(
                f"{what}() on an HBS block whose {'+'.join(missing)} factors "
                f"are not resident on {self.compute_device} (resident="
                f"{self._resident}). The block was evicted and nothing staged "
                "it back. Under strict=True this is an error instead of a "
                "silent fallback to host arithmetic."
            )
        self.prefetch(groups=missing)

    def prefetch(self, groups=('core',), non_blocking=True):
        """Stage a device mirror for the named groups. No-op if resident.

        With a host master, each upload comes from page-locked memory and is
        asynchronous.  Without one (never evicted) the live tensors are moved
        directly; the master is created on the first evict instead."""
        need = [g for g in groups if not self._group_resident(g)]
        if not need:
            return self
        dev, n = self.compute_device, 0
        for g in need:
            rec = self._host[g]
            if rec is not None:
                for (name, i), hv in zip(rec['ents'], rec['views']):
                    getattr(self, name)[i] = hv.to(dev, non_blocking=non_blocking)
                    n += hv.nbytes
            else:
                for name, i in self._entries(g):
                    t = getattr(self, name)[i]
                    if not _dev_eq(t.device, dev):
                        getattr(self, name)[i] = t.to(dev, non_blocking=non_blocking)
                        n += t.nbytes
            self._resident[g] = dev
        self.nFill += 1
        self.bytesH2D += n
        return self

    def evict(self, groups=('core', 'ulv')):
        """Drop the device mirror for the named groups.

        The first evict of a group after it is built snapshots it into a
        page-locked host master (one D2H copy, one sync).  Every later evict
        rebinds the tensor lists to views of that master: no copy, no sync,
        and the device memory goes straight back to the caching allocator.
        bytesD2H therefore counts real transfers only.

        compute_device is NOT touched.  An evicted block that is applied again
        must be re-staged, never silently demoted to host arithmetic; that is
        the whole point of this method existing instead of to('cpu').
        """
        if _dev_eq(self.compute_device, self.home):
            return self
        for g in groups:
            if self._resident[g] is None:
                continue
            rec = self._host[g]
            if rec is None:
                rec, nD2H = self._snapshot(g)
                self.bytesD2H += nD2H
            for (name, i), hv in zip(rec['ents'], rec['views']):
                getattr(self, name)[i] = hv
            self._resident[g] = None
            self.nSpill += 1
        return self

    def release_ulv(self):
        """Send the ULV factors home, keep the apply factors on device."""
        return self.evict(groups=('ulv',))

    def to(self, device, non_blocking=False):
        device = torch.device(device)
        moved = 0
        for g in self._GROUPS:
            for name, i in self._entries(g):
                t = getattr(self, name)[i]
                if not _dev_eq(t.device, device):
                    moved += t.nbytes
                    getattr(self, name)[i] = t.to(device, non_blocking=non_blocking)
            self._resident[g] = device
            self._host[g]     = None    # relocation: home changes, master is moot

        # perm rides with 'core' in _entries, but normalize it here so a
        # tree-supplied numpy perm becomes a tensor on the target rather than
        # staying host-side and forcing an implicit H2D on every fancy-index.
        if not torch.is_tensor(self.perm):
            self.perm = torch.as_tensor(self.perm, dtype=torch.int64,
                                        device=device)

        self.home           = device
        self.compute_device = device

        # Counted separately from bytesH2D/bytesD2H so residency_report()
        # keeps measuring the spill/fill schedule and is not polluted by
        # one-off relocations.
        self.bytesRelocated = getattr(self, 'bytesRelocated', 0) + moved
        return self

    def cpu(self):
        """Relocate to host: this becomes a host object that computes on the
        host.  NOT the same as evict(), which keeps compute_device on the
        accelerator and merely drops the mirror."""
        return self.to('cpu')

    def cuda(self, index=None):
        """Relocate to a CUDA device."""
        return self.to('cuda' if index is None else f'cuda:{index}')

    def device_nbytes(self, include_ulv=True):
        groups = ('core', 'ulv') if include_ulv else ('core',)
        tot = sum(t.nbytes for _, lst in self._lists(groups)
                  for t in lst if torch.is_tensor(t))
        if torch.is_tensor(self.perm):
            tot += self.perm.nbytes
        return tot

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def compute_ULV(self):
        """Factorize an already-compressed block.  constructHBS(with_ulv=True)
        does this inline and more cheaply, since the level factors are still
        live there; this is for blocks built with compute_ULV=False.

        The `device` argument was missing in the previous version, so every
        call raised TypeError before reaching ULVsparse."""
        self._require_resident('compute_ULV')
        tic = time.time()
        (self.Qlist, self.Wlist, self.Uulist,
         self.Rlist, self.NNvec) = ULVsparse.compute_ULV(
            self.Umats, self.Dmats, self.Vmats, self.Nbvec, self.compute_device)
        self.tULV = time.time() - tic
        self._resident['ulv'] = self.compute_device
        self._host['ulv']     = None    # rebuilt: any old master is stale

    def solve(self, b, mode='N', chunk=None, overwrite_b=False):
        """Apply the inverse (mode='N') or inverse transpose (mode='T') of the
        HBS operator via its ULV factorization.

        Permutation convention
        ----------------------
        The stored factors represent  A_p = P A P^T,  where P is the leaf
        permutation, (P v)[i] = v[perm[i]].  Therefore

            A^{-1} = P^T A_p^{-1} P      and      A^{-T} = P^T A_p^{-T} P,

        so the gather / scatter around the ULV solve is identical in both
        modes; only the triangular sweep in ULVsparse.solve differs.

        Memory
        ------
        The ULV sweeps hold roughly 7 rhs-sized tensors at their peak.  For the
        2s-column fused solves of the red-black factorization that is the
        device-memory peak, so columns are processed in chunks of `chunk`
        (default _SOLVE_CHUNK[0]; 0 disables chunking).  Columns are
        independent, so the result is identical up to BLAS kernel selection.

        overwrite_b=True writes the solution into b itself (when b is a torch
        tensor already on compute_device with the factor dtype, or a numpy
        array aliased by a CPU compute device) and returns it, so no output
        buffer is allocated.  Safe because each chunk's columns are gathered
        (copied) before that chunk's solution is scattered back.  If b had to
        be converted anyway, the private copy is always reused as output.
        """
        if not self.Qlist:
            raise RuntimeError(
                "solve() requires the ULV factorization, but this HBSMAT has "
                "an empty Qlist.  Build it with construct(..., compute_ULV=True) "
                "or call compute_ULV() first."
            )
        if mode not in ('N', 'T'):
            raise NotImplementedError(f"mode '{mode}' not recognized. Use 'N' or 'T'.")

        self._require_resident('solve', need_ulv=True)

        input_is_numpy = isinstance(b, np.ndarray)
        if not (input_is_numpy or torch.is_tensor(b)):
            raise TypeError("b must be either a numpy.ndarray or a torch.Tensor")
        if b.ndim not in (1, 2):
            raise ValueError(f"b must have ndim 1 or 2, got ndim={b.ndim}")

        dtype_t = self.Dmats[0].dtype
        if input_is_numpy:
            src_ptr = b.__array_interface__['data'][0] if b.size else None
            b_torch = torch.as_tensor(b, dtype=dtype_t, device=self.compute_device)
        else:
            src_ptr = b.data_ptr() if b.numel() else None
            b_torch = b.to(device=self.compute_device, dtype=dtype_t)

        # Did the conversion produce a private copy, or does b_torch alias b?
        aliased = (src_ptr is not None) and (b_torch.data_ptr() == src_ptr)

        was_vector = (b_torch.ndim == 1)
        if was_vector:
            b_torch = b_torch[:, None]

        nrow, ncol = b_torch.shape
        if nrow != self.perm.shape[0]:
            raise ValueError(f"solve: expected {self.perm.shape[0]} rows, got {nrow}")

        # Output buffer: reuse b_torch when allowed (caller opted in) or when
        # it is our own copy anyway; otherwise allocate.
        if overwrite_b or not aliased:
            out = b_torch
        else:
            out = torch.empty((nrow, ncol), dtype=dtype_t, device=b_torch.device)

        c = _SOLVE_CHUNK[0] if chunk is None else chunk
        if not c or c > ncol:
            c = max(ncol, 1)

        # u = P^T ( ULV^{-1} ( P b ) ), one column chunk at a time
        for j0 in range(0, ncol, c):
            j1 = min(j0 + c, ncol)
            bperm = b_torch[self.perm, j0:j1]      # advanced-index gather: a copy
            uhat = ULVsparse.solve(
                self.Umats, self.Dmats, self.Qlist, self.Wlist, self.Uulist,
                self.Rlist, self.NNvec, bperm, device=self.compute_device, mode=mode,
                Vmats=self.Vmats,
            )
            del bperm                              # free before the scatter
            out[self.perm, j0:j1] = uhat           # columns j0:j1 already read
            del uhat

        u = out[:, 0] if was_vector else out
        if input_is_numpy:
            return u.detach().cpu().numpy()
        return u
