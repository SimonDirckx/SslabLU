import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.linalg as tla
import matAssembly.HBS.ULVsparse_torch as ULVsparse

# ---------------------------------------------------------------------------
# Memory policy (host-first)
# ---------------------------------------------------------------------------
# An HBSMAT keeps exactly one persistent copy of its factors: a host master,
# one flat (page-locked when possible) buffer per group.  The tensor lists
# Umats/Vmats/Dmats and Qlist/Wlist/Rlist/Uulist are views into that master
# and always live on the host.
#
#   * construct() streams the blocks of each level through the device in
#     chunks and writes every factor to the host master as soon as it is
#     produced.  When it returns, nothing of the block is on the device.
#
#   * matmat/rmatmat/solve upload the factors they need for the duration of
#     that one call and drop them on return ("transient staging").
#
#   * Their right-hand sides may live on the host.  They are then streamed
#     through the device in column chunks of bounded size (_STREAM_BYTES),
#     and with out=/alpha=/beta= the result is combined on the device,
#         out <- alpha * op(X) + beta * C        (C defaults to out),
#     chunk by chunk, before it goes back.  Columns are independent, so this
#     is exact.  A caller that keeps its m x s tensors on the host therefore
#     never has one on the device: the device holds the block and one chunk.
#     Host tensors stored column-major (see host_empty) make every chunk one
#     contiguous region, so each transfer is a single DMA.
#
#   * prefetch() keeps a device mirror until evict().  Use it when the same
#     block is applied many times in a row and the re-uploads would cost more
#     than the memory.  Nothing creates a mirror implicitly.
#
# So the rule a caller has to know is: a block occupies device memory only
# while one of its own operations is running, or between prefetch and evict.
# ---------------------------------------------------------------------------

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
_PIN_HOST     = [True]  # page-lock host masters and host work buffers;
                        # set False if page-locked memory is limited
_SOLVE_CHUNK  = [512]   # columns per ULV solve in HBSMAT.solve; 0 disables
_UV_BASIS     = ['svd_host']  # fast-path basis extraction:
                        #   'svd_host'  SVD of L = RB^T (ny x ny) on the host,
                        #               batch-parallel, one LAPACK thread each
                        #   'svd_dev'   same SVD on the device (torch.linalg.svd)
                        #   'jw'        previous Jordan-Wielandt eigh (2ny x 2ny)
                        #               on the device, kept for A/B comparison
_SVD_WORKERS  = [0]     # host threads for 'svd_host'; 0 means the CPUs this process may use
_SVD_POOL     = [None, 0]   # (executor, its worker count), created lazily
_SVD_DEV_DRIVER = ['gesvd']  # for 'svd_dev': 'gesvd' or 'gesvdj', never 'gesvda'
_UV_CHUNK_ELEMS = [1<<28]  # secondary cap on c*n*s per chunk (32-bit size
                        # limits in batched cuSOLVER/cuBLAS/MAGMA).  0 disables.
_CHUNK_BYTES  = [2 << 30]  # device working set per construction chunk.  The
                        # number of blocks per chunk is derived from this; it
                        # is the knob that sets the construction peak.  0 means
                        # one chunk per level.
_SAMPLE_SPILL_BYTES = [None]  # where construction keeps the next level's
                        # four sample tensors:
                        #   None  on the device when they fit in the device
                        #         budget (_DEVICE_BUDGET) next to what is
                        #         already allocated, else page-locked host
                        #   0     always host;  int  host above that many bytes
_STREAM_BYTES = [2 << 30]  # device working set of one column chunk in a
                        # streamed apply/solve, as estimated by _ws_per_col
                        # (2-3x above measured); sets the chunk width.  The
                        # explicit column caps _MATMAT_CHUNK/_SOLVE_CHUNK
                        # still apply on top.
_PINNED = [0]           # bytes this module currently has page-locked
                        # (reporting only).  A registration the driver
                        # refuses falls back to ordinary memory: slower
                        # transfers, same results.
_DEVICE_BUDGET = [None]  # bytes of device memory the code may plan to use;
                        # None = 85% of the device's total memory
_STAGE_ON_CPU = [False]  # testing only: treat a CPU compute_device as if it
                        # were an accelerator, so the staging/mirror code paths
                        # run (with real copies) on a machine without a GPU.

_HOST = torch.device('cpu')


def device_budget(dev):
    """Bytes of device memory planned decisions may use (see _DEVICE_BUDGET)."""
    if _DEVICE_BUDGET[0] is not None:
        return int(_DEVICE_BUDGET[0])
    dev = torch.device(dev)
    if dev.type == 'cuda':
        return int(0.85 * torch.cuda.get_device_properties(dev).total_memory)
    return 1 << 62


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


def memlock_limit():
    """The process's page-lockable bytes (soft RLIMIT_MEMLOCK), or None when
    unlimited or unknown."""
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        return None if soft == resource.RLIM_INFINITY else int(soft)
    except Exception:
        return None


def _may_pin(nbytes, kind):
    """Try to page-lock a new buffer of nbytes?  Callers that need work
    buffers page-locked reserve them first (HostPool.reserve), so block
    storage can simply take whatever the driver still allows."""
    return bool(_PIN_HOST[0]) and nbytes > 0


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
    every transfer still works, just synchronously.
    """
    __slots__ = ('t', 'registered', 'kind')

    def __init__(self, numel, dtype, kind='master'):
        self.t = torch.empty(numel, dtype=dtype)
        self.registered = False
        self.kind = kind
        nb = self.t.numel() * self.t.element_size()
        if torch.cuda.is_available() and _may_pin(nb, kind):
            try:
                rc = torch.cuda.cudart().cudaHostRegister(self.t.data_ptr(), nb, 0)
                self.registered = (rc is None) or (int(rc) == 0)
            except Exception:
                self.registered = False
            if self.registered:
                _PINNED[0] += nb

    def __del__(self):
        if getattr(self, 'registered', False):
            try:
                torch.cuda.cudart().cudaHostUnregister(self.t.data_ptr())
                _PINNED[0] -= self.t.numel() * self.t.element_size()
            except Exception:
                pass


class HostPool:
    """Reusable host buffers, page-locked once and handed out again.

    Registering memory with cudaHostRegister costs time (roughly a second per
    10 GB), so buffers are kept after use and given to the next request that
    fits.  get() returns a 1-D tensor of exactly the requested length (a view
    of a buffer at least that large); put() takes back what get() returned.
    clear() releases every buffer that is not currently handed out.

    pinned=False requests are plain pageable tensors and are not pooled.
    """

    def __init__(self):
        self._free = []           # [_HostBuffer], not handed out
        self._lent = {}           # data_ptr -> _HostBuffer

    def get(self, numel, dtype=torch.float64, pinned=True):
        numel = int(numel)
        if not pinned or numel == 0:
            return torch.empty(numel, dtype=dtype)
        best = None
        for hb in self._free:
            if (hb.t.dtype == dtype and hb.t.numel() >= numel and
                    (best is None or hb.t.numel() < best.t.numel())):
                best = hb
        if best is None:
            best = _HostBuffer(numel, dtype, kind='work')
        else:
            self._free.remove(best)
        self._lent[best.t.data_ptr()] = best
        return best.t[:numel]

    def put(self, t):
        if t is None:
            return
        hb = self._lent.pop(t.data_ptr(), None)
        if hb is not None:
            self._free.append(hb)

    def clear(self):
        self._free = []

    def reserve(self, sizes, dtype=torch.float64, pinned=True):
        """Allocate (and page-lock) buffers of the given element counts now
        and keep them free for later get() calls -- so they are locked
        before anything else can use up the lockable memory.  Returns the
        number of them that are page-locked."""
        got = [self.get(n, dtype=dtype, pinned=pinned) for n in sizes]
        locked = sum(1 for x in got
                     if self._lent.get(x.data_ptr()) is not None
                     and self._lent[x.data_ptr()].registered)
        for x in got:
            self.put(x)
        return locked

    @property
    def nbytes(self):
        """(bytes handed out, bytes held free)"""
        lent = sum(hb.t.numel() * hb.t.element_size() for hb in self._lent.values())
        free = sum(hb.t.numel() * hb.t.element_size() for hb in self._free)
        return lent, free


host_pool = HostPool()      # shared by HBSMAT construction and its callers


def host_empty(m, ncol, dtype=torch.float64, pinned=True):
    """(m, ncol) host tensor stored column-major, from host_pool.

    Column-major is what streaming wants: the column chunk X[:, j0:j1] is
    one contiguous region, so it moves with a single DMA in either
    direction.  Return it with host_pool.put(X) when done."""
    flat = host_pool.get(m * ncol, dtype=dtype, pinned=pinned)
    return flat.view(ncol, m).t()


def _is_colmajor(X):
    return (X.dim() == 2 and X.stride(0) == 1 and
            (X.size(1) == 1 or X.stride(1) >= X.size(0)))


def _to_dev(X, dev):
    """Copy a (chunk of a) tensor to dev, keeping its layout: a column-major
    host chunk stays column-major, so the copy is one contiguous DMA."""
    if _dev_eq(X.device, dev):
        return X
    return X.to(dev, non_blocking=True)


def _from_dev(dst, src):
    """dst[...] = src across devices, matching dst's layout on src's device
    first so the transfer is one DMA.  Returns True if the copy may still be
    in flight (the caller synchronizes before host code reads dst)."""
    if dst.device == src.device:
        dst.copy_(src)
        return False
    if _is_colmajor(dst) and dst.size(1) > 1 and not _is_colmajor(src):
        src = src.t().contiguous().t()
    dst.copy_(src, non_blocking=True)
    return dst.device.type == 'cpu' and src.device.type == 'cuda'


_STAGE = [None]             # grow-only page-locked staging buffer


def _stage(numel, dtype):
    """A page-locked host scratch area of at least numel elements.  Only for
    blocking copies: the next call may hand out the same memory."""
    hb = _STAGE[0]
    if hb is None or hb.t.numel() < numel or hb.t.dtype != dtype:
        _STAGE[0] = None
        hb = _STAGE[0] = _HostBuffer(int(numel), dtype, kind='work')
    return hb.t[:numel]


def _rows_to_dev(X, rows, a, b, dev, dt):
    """Rows of a 2-D sample on dev as a contiguous (#rows, s) tensor.

    rows: host int64 index tensor, or None for the contiguous range a:b.
    A host X goes through the page-locked stage, gathered along its
    contiguous direction (so a column-major X is read column by column),
    then moves with one blocking DMA."""
    s = X.shape[1]
    if X.dtype != dt:
        X = X.to(dt)
    if X.device.type != 'cpu':
        R = X[a:b] if rows is None else X.index_select(0, rows.to(X.device))
        return R.to(dev).contiguous()
    nr = (b - a) if rows is None else rows.numel()
    staged = dev.type == 'cuda'
    if _is_colmajor(X) and s > 1:
        Xt = X.t()                                     # (s, m) contiguous
        if staged:
            buf = _stage(nr * s, dt).view(s, nr)
            if rows is None:
                buf.copy_(Xt[:, a:b])
            else:
                torch.index_select(Xt, 1, rows, out=buf)
            return buf.to(dev).t().contiguous()
        R = Xt[:, a:b] if rows is None else Xt.index_select(1, rows)
        return R.t().contiguous()
    if staged:
        buf = _stage(nr * s, dt).view(nr, s)
        if rows is None:
            buf.copy_(X[a:b])
        else:
            torch.index_select(X, 0, rows, out=buf)
        return buf.to(dev)
    R = X[a:b] if rows is None else X.index_select(0, rows)
    return R.contiguous()


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
# host masters
# ---------------------------------------------------------------------------

def _numel(shape):
    return math.prod(shape)


def _views(flat, layout):
    """{name: [view, ...]} carving `flat` according to `layout`, which maps
    each list name to a list of shapes.  Iteration order of `layout` fixes
    the offsets, so the same layout always gives the same carving."""
    out, o = {}, 0
    for name, shapes in layout.items():
        lst = []
        for shp in shapes:
            n = _numel(shp)
            lst.append(flat[o:o + n].view(shp))
            o += n
        out[name] = lst
    return out


def _new_master(layout, dtype):
    """Allocate a host master for `layout`.  Returns (record, lists)."""
    total = sum(_numel(s) for shapes in layout.values() for s in shapes)
    buf = _HostBuffer(total, dtype)
    rec = {'bufs': {dtype: buf}, 'layout': layout, 'dtype': dtype}
    return rec, _views(buf.t, layout)


def _level_shapes(Nb, nl, L, rk, fac):
    """Block counts and sizes the construction will produce.

    Returns ([(Nb_i, n_i, k_i) for list index i = 0..L-2], (Nb_r, n_r)):
    index 0 is the leaf, the pair is the root.  Mirrors the rank rule of the
    sweep: rkm_i = min(rk, nl * fac**i), k_i = min(rkm_i, n_i), and the next
    level has n_{i+1} = fac * k_i rows per block."""
    levels, n = [], nl
    for i in range(L - 1):
        k = min(rk, nl * fac ** i, n)
        levels.append((Nb, n, k))
        Nb //= fac
        n = fac * k
    return levels, (Nb, n)


def _core_layout(levels, root):
    lay = {'Umats': [], 'Vmats': [], 'Dmats': []}
    for Nb, n, k in levels:
        lay['Umats'].append((Nb, n, k))
        lay['Vmats'].append((Nb, n, k))
        lay['Dmats'].append((Nb, n, n))
    lay['Dmats'].append((root[0], root[1], root[1]))
    return lay


def _ulv_layout(levels, root):
    """Wlist and Uulist have no root entry: ULVsparse.solve never reads one."""
    lay = {'Qlist': [], 'Wlist': [], 'Rlist': [], 'Uulist': []}
    for Nb, n, k in levels:
        lay['Qlist'].append((Nb, n, n))
        lay['Wlist'].append((Nb, n, n - k))
        lay['Rlist'].append((Nb, n - k, n))
        lay['Uulist'].append((Nb, n - k, k))
    lay['Qlist'].append((root[0], root[1], root[1]))
    lay['Rlist'].append((root[0], root[1], root[1]))
    return lay


def _put(dst, src, pinned):
    """dst[...] = src.  Asynchronous only when dst is page-locked host memory
    or device memory; a D2H copy into pageable memory is synchronous.  The
    source may be freed right after: the copy is ordered on the current
    stream, and so is any reuse of the source's memory."""
    dst.copy_(src, non_blocking=pinned or dst.device.type != 'cpu')


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

    `device` and `fast` are accepted for call-site compatibility and ignored."""
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
    nw = _SVD_WORKERS[0] or (
        len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
        else (os.cpu_count() or 1))
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
    reorthogonalization in _uv_side restore unit columns."""
    ny = Rb.shape[-1]
    H = Rb.new_zeros((2 * ny, 2 * ny))
    H[:ny, ny:] = Rb.mT
    H[ny:, :ny] = Rb
    return tla.eigh(H).eigenvectors[:, -k:].flip(-1)[:ny, :]


def _left_sv_host(RB, k):
    """Top-k left singular vectors of L = RB^T, per block, on the host.

    RB: (Nb, ny, ny) upper triangular with Bp Bp^T = RB^T RB, so the left
    singular vectors of Bp are exactly those of L.  Returns (Nb, ny, k) on
    RB's device, columns ordered by decreasing sigma.  One LAPACK thread per
    worker; a chunk that fails to converge is redone block by block, with the
    Jordan-Wielandt eigh as the last resort."""
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


def _cholqr2(X, need_q=True, overwrite=False):
    """CholeskyQR of X^T, repeated.  X: (Nb, m, s) with s >= m.

    Returns (Q, R) with R (Nb, m, m) upper triangular and Q (Nb, m, s) having
    orthonormal ROWS, so that  X = R^T Q  and  X X^T = R^T R.

    Two passes give ||Q Q^T - I|| = O(u) and R accurate to O(u ||X||) as long
    as cond(X) <= u^{-1/2}.  A failed Cholesky means cond(X) is past that
    bound; the Gram is then shifted by 11(ms + m(m+1)) u ||X||_F^2 and a
    third pass is run, which extends the bound to cond(X) <= 1/u.

    need_q=False skips the final triangular solve when only R is wanted.

    overwrite=True lets the FIRST pass solve into X itself.  Only for an X the
    caller owns and will not read again (the projected samples Bp); without
    it the first pass allocates a second (Nb, m, s) tensor.  Later passes
    always reuse this function's own buffer.

    Costs one D2H sync per pass, from the Cholesky info check.
    """
    Nb, m, s = X.shape
    assert s >= m, f"_cholqr2: need s >= m, got s={s}, m={m}"
    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    npass = max(1, _UV_ORTH_PASSES[0])
    Q, R, p = X, None, 0
    while p < npass:
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
            in_place = (p > 1 or overwrite) and not Q.requires_grad
            Q = torch.linalg.solve_triangular(
                Ri.mT, Q, upper=False, left=True, out=Q if in_place else None)
        R = Ri if R is None else torch.bmm(Ri, R)
    return Q, R


def construct_D(U, V, M_om, M_psi):
    """D = (I - U U*) Y Om^+  +  U [ (I - V V*) Z Psi^+ ]* U*

    with M_om = Y Om^+ and M_psi = Z Psi^+ supplied by _uv_side."""
    P  = M_om - torch.bmm(U, torch.bmm(U.mT, M_om))     # (Nb, ny, n)
    Gk = torch.bmm(M_psi, U)                            # (Nb, ny, k)
    Gk = Gk - torch.bmm(V, torch.bmm(V.mT, Gk))
    return P + torch.bmm(U, Gk.mT)


def _uv_side(A, B, k, fast):
    """One half of a level's basis computation.

    Returns (U, M): U the leading k left singular vectors of B (I - A^+ A),
    M = B A^+.  Neither A nor B is written.

    fast=True   CholeskyQR2 on A^T for the projector, then the singular
                vectors of the projected samples without squaring them:

                    A^T = Q1 R          CholeskyQR2, ||Q1^T Q1 - I|| = O(u)
                    M   = (B Q1) R^{-T}                         = B A^+
                    Bp  = B - (B Q1) Q1^T
                    Bp  = RB^T QB       CholeskyQR2, R factor only
                    U   = top-k left singular vectors of RB^T

                The second CholeskyQR pass on A^T is not decorative: with Q1
                orthonormal only to eps, Bp keeps a leak of size eps ||B||
                in row(A).  _UV_ORTH_PASSES[0] = 1 measures what it is worth.

    fast=False  Householder QR of W = [A^T | B^T] = Q R, then
                M = R_ab^T R_aa^{-T},  U = svd(R[n:, n:]^T).U  on the host.
                Kept as the reference the fast path is validated against.
    """
    Nb, ny, s = B.shape
    n = A.shape[1]
    dev = B.device
    _UV_NCALL[0] += 1

    if fast:
        _uv_sync(dev); _t = time.time()
        QA, RA = _cholqr2(A)                      # A = RA^T QA
        _uv_sync(dev); _UV_TQR[0] += time.time() - _t

        _uv_sync(dev); _t = time.time()
        C  = torch.bmm(B, QA.mT)                  # (Nb, ny, n) = B Q1
        M  = torch.linalg.solve_triangular(
            RA.mT, C, upper=False, left=False).contiguous()
        Bp = torch.baddbmm(B, C, QA, beta=1.0, alpha=-1.0)   # B - C Q1^T
        del C, QA, RA
        _uv_sync(dev); _UV_TSETUP[0] += time.time() - _t

        _uv_sync(dev); _t = time.time()
        # Bp is private to this call: let the first pass reuse its storage.
        _, RB = _cholqr2(Bp, need_q=False, overwrite=True)
        del Bp
        if _UV_BASIS[0] == 'svd_host':
            UU = _left_sv_host(RB, k)
        elif _UV_BASIS[0] == 'svd_dev':
            UU = _left_sv_dev(RB, k)
        else:
            # Jordan-Wielandt form H = [[0, RB^T], [RB, 0]]; workspace is
            # budgeted in bytes since it grows with both batch and size.
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
        # Renormalize and reorthogonalize: a near no-op for the SVD paths,
        # required for 'jw'.  ULV builds the complement of V and needs
        # V^T V = I.
        UU = UU / UU.norm(dim=-2, keepdim=True).clamp_min(
            torch.finfo(UU.dtype).tiny)
        Gk = torch.bmm(UU.mT, UU)
        Lk, info = torch.linalg.cholesky_ex(Gk)
        if not bool(info.any()):
            UU = torch.linalg.solve_triangular(
                Lk.mT, UU, upper=True, left=False,
                out=UU if not UU.requires_grad else None)
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
        R = tla.qr(W, mode='r').R
        del W
        M = torch.linalg.solve_triangular(
            R[:, :n, :n].mT, R[:, :n, n:].mT, upper=False,
            left=False).contiguous()           # (Nb, ny, n) = B A^+
        _uv_sync(dev); _UV_TQR[0] += time.time() - _t

        _uv_sync(dev); _t = time.time()
        L = R[:, n:, n:].mT
        UU = tla.svd(L.to('cpu'), full_matrices=False).U[..., :k]
        UU = UU.to(dev).contiguous()
        del R, L
        _uv_sync(dev); _UV_TBASIS[0] += time.time() - _t

    return UU, M


def compute_UV_pair(Om, Y, Psi, Z, rk, device=None, fast=False):
    """((U, Y Om^+), (V, Z Psi^+)) for one batch of blocks.  Kept for
    callers outside this module; HBSMAT calls _uv_side directly."""
    Nb, ny, s = Y.shape
    n = Om.shape[1]
    k = min(rk, ny)
    assert Psi.shape == Om.shape and Z.shape == Y.shape, \
        "compute_UV_pair needs matching shapes"
    assert s >= n + k, \
        "undersampled: not enough columns left after projecting off Om"
    return _uv_side(Om, Y, k, fast), _uv_side(Psi, Z, k, fast)


def compute_UV_pair_chunked(Om, Y, Psi, Z, rk, device=None, fast=False):
    """compute_UV_pair split along the block dimension (exact: blocks are
    independent).  Kept for callers outside this module."""
    Nb, ny, s = Y.shape
    n = Om.shape[1]
    cap = _UV_CHUNK_ELEMS[0]
    c = Nb if not cap else max(1, min(Nb, cap // max(1, max(n, ny) * s)))
    if c >= Nb:
        return compute_UV_pair(Om, Y, Psi, Z, rk, device, fast=fast)
    parts = [compute_UV_pair(Om[j:j+c], Y[j:j+c], Psi[j:j+c], Z[j:j+c],
                             rk, device, fast=fast) for j in range(0, Nb, c)]
    cat = lambda xs: torch.cat(xs, dim=0)
    return ((cat([p[0][0] for p in parts]), cat([p[0][1] for p in parts])),
            (cat([p[1][0] for p in parts]), cat([p[1][1] for p in parts])))


# ---------------------------------------------------------------------------
# ULV, one chunk of blocks at a time
# ---------------------------------------------------------------------------

def _ulv_rhat(D, Uhat_c, R22_c):
    """R'_ell for a chunk: D at the leaf, else Uhat-blockdiag @ D + R22-blockdiag.
    Uhat_c / R22_c are the fac*c children of the chunk's c blocks."""
    if Uhat_c is None:
        return D
    Rhat = ULVsparse.sparse_block_mult_tens(Uhat_c, D)
    # Rhat is fresh from the bmm above and read nowhere else
    return ULVsparse.block_diag_add_tens(Rhat, R22_c, inplace=True)


def _ulv_chunk(D, U, V, Uhat_c, R22_c):
    """ULV step for a chunk of non-root blocks.

    Returns (Q, W1, Ru, Uu, Ud, R22) with Ud and R22 the carries the next
    level needs.  Same arithmetic as ULVsparse.compute_QRW_sparse followed by
    the Uhat update, except that

      * [W1 | V] is never concatenated: the two products fill one buffer;
      * R22 and Ru are compact copies, so the (c, n, n) R is freed here
        instead of being kept alive by views carried to the next level."""
    c, n, k = V.shape
    Rhat = _ulv_rhat(D, Uhat_c, R22_c)
    Uh = U if Uhat_c is None else ULVsparse.sparse_block_mult_tens(Uhat_c, U)

    W1 = ULVsparse.block_qr_tens(V)                 # (c, n, n-k)
    DW = torch.empty((c, n, n), dtype=D.dtype, device=D.device)
    DW[:, :, :n-k] = torch.bmm(Rhat, W1)
    DW[:, :, n-k:] = torch.bmm(Rhat, V)
    del Rhat
    Q, R = tla.qr(DW)
    del DW
    Ru  = R[:, :n-k, :].contiguous()
    R22 = R[:, n-k:, n-k:].contiguous()
    del R
    Uu = torch.bmm(Q[:, :, :n-k].mT, Uh)
    Ud = torch.bmm(Q[:, :, n-k:].mT, Uh)
    return Q, W1, Ru, Uu, Ud, R22


def _ulv_root(D, Uhat_c, R22_c):
    """(Q, R) of the root blocks; batched reduced QR of R'_root."""
    return tla.qr(_ulv_rhat(D, Uhat_c, R22_c))


class HBSMAT:
    """HBS matrix: compression, apply, and ULV solve.

    @init:      linear operator A, tree on DOFs (symmetric), target rank k
    @constructs HBS approximation to the source-target map
    @implements matvec/matmat (normal and transpose) and solve

    Memory policy: see the top of this module.  In short, the factor lists
    always hold host tensors (views of the host master); the device only ever
    holds a block's factors while one of its operations runs, or between
    prefetch() and evict().

    Level ordering: list index 0 is the leaf level and Dmats[-1] the root.
    Wlist and Uulist have no root entry.
    """

    # every attribute holding a list of tensors, by group
    _GROUPS = {
        'core': ('Umats', 'Vmats', 'Dmats'),
        'ulv' : ('Qlist', 'Wlist', 'Rlist', 'Uulist'),
    }

    def __init__(self, A=None, device=None, tree=None, quad=False):
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
        self._perm = None

        self._host   = {'core': None, 'ulv': None}   # host master records
        self._dev    = {}     # group -> {name: [device tensors], 'perm': ...}
                              # mutated in place, never rebound, so .T views
                              # share it
        self.compute_device = (torch.device(device) if device is not None
                               else _HOST)
        self.home    = _HOST
        self.strict  = False
        self.nFill = self.nSpill = 0
        self.bytesH2D = self.bytesD2H = 0

        self.A = A
        if A is not None:
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
        """Adopt externally built factors.  They are copied into a host
        master; the arguments may live anywhere."""
        layout = {'Umats': [tuple(t.shape) for t in Umats],
                  'Vmats': [tuple(t.shape) for t in Vmats],
                  'Dmats': [tuple(t.shape) for t in Dmats]}
        self._dev.clear()
        self._adopt_group('core', layout, {'Umats': Umats, 'Vmats': Vmats,
                                           'Dmats': Dmats})
        self._host['ulv'] = None
        for nm in self._GROUPS['ulv']:
            setattr(self, nm, [])
        n = sum(D.shape[0] * D.shape[1] for D in Dmats[:1])
        self.perm  = torch.arange(n)
        self.fac   = fac
        self.Nbvec = list(Nbvec)
        self.Nb    = Nbvec[0]
        self.L     = len(Dmats)
        self.shape = torch.tensor([n, n], dtype=torch.int64)
        self.dtype = Dmats[0].dtype

    def _adopt_group(self, g, layout, tensors):
        dtype = next(t.dtype for lst in tensors.values() for t in lst)
        rec, lists = _new_master(layout, dtype)
        for nm, lst in tensors.items():
            for dst, src in zip(lists[nm], lst):
                dst.copy_(src)
        self._host[g] = rec
        for nm, lst in lists.items():
            setattr(self, nm, lst)

    @property
    def nbytes(self):
        return (sum(U.nbytes for U in self.Umats)
                + sum(V.nbytes for V in self.Vmats)
                + sum(D.nbytes for D in self.Dmats))

    @property
    def device(self):
        return str(self.compute_device)

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
        """Leaf permutation, always a host int64 tensor."""
        return self._perm

    @perm.setter
    def perm(self, p):
        if p is None:
            self._perm = None
        elif torch.is_tensor(p):
            self._perm = p.detach().to(device=_HOST, dtype=torch.int64)
        else:
            self._perm = torch.as_tensor(np.asarray(p), dtype=torch.int64)

    # ------------------------------------------------------------------
    # device placement
    # ------------------------------------------------------------------

    @property
    def compute_device(self):
        """Where operations run.  Assigning it directly (as the red-black
        solver does) is supported: a mirror on the previous device is dropped,
        so an operation can never pick up factors from the wrong device."""
        return self._compute_device

    @compute_device.setter
    def compute_device(self, d):
        d = torch.device(d)
        old = self.__dict__.get('_compute_device')
        if old is not None and not _dev_eq(old, d):
            mirror = self.__dict__.get('_dev')
            if mirror:
                mirror.clear()
        self._compute_device = d

    def _staging(self):
        """True when compute_device is not the host, i.e. operations must
        upload factors first."""
        return self.compute_device.type != 'cpu' or _STAGE_ON_CPU[0]

    # -- names kept from the previous residency scheme, so external code
    #    that touches them keeps working.  None of them is needed any more.
    def _require_resident(self, what=None, need_ulv=False):
        """No-op: every operation stages what it needs."""
        return None

    def _reset_residency(self, groups=None):
        """No-op: there is no device copy to invalidate."""
        return None

    def _as_local(self, X):
        dev = self.compute_device
        if torch.is_tensor(X):
            return X.to(device=dev, dtype=self.dtype_t, non_blocking=True)
        return torch.from_numpy(np.asarray(X)).to(device=dev, dtype=self.dtype_t)

    @property
    def _resident(self):
        """{group: device or None}, for reporting.  A group with no master
        (nothing built) reports None."""
        out = {}
        for g in self._GROUPS:
            if self._host.get(g) is None:
                out[g] = None
            elif not self._staging() or g in self._dev:
                out[g] = self.compute_device
            else:
                out[g] = None
        return out

    @_resident.setter
    def _resident(self, value):
        """Ignored: residency is derived, not stored."""
        pass

    def _group_resident(self, g):
        """True if group g needs no upload before use."""
        if not self._staging() or self._host.get(g) is None:
            return True
        return g in self._dev

    def _upload(self, g):
        """Device copy of group g's master, as {name: [tensors]}; one
        transfer per group.  Core also carries the permutation."""
        rec = self._host[g]
        flat = rec['bufs'][rec['dtype']].t.to(self.compute_device,
                                              non_blocking=True, copy=True)
        lists = _views(flat, rec['layout'])
        nb = flat.nbytes
        if g == 'core':
            lists['perm'] = self.perm.to(self.compute_device, copy=True)
            nb += self.perm.nbytes
        self.nFill += 1
        self.bytesH2D += nb
        return lists

    def _factors(self, groups, what):
        """{name: [tensors on compute_device]} plus 'perm', for one operation.

        Uses the prefetched mirror where there is one; otherwise uploads a
        transient copy that the caller drops when the operation returns.
        Under strict=True a missing mirror is an error instead."""
        out = {}
        if not self._staging():
            for g in groups:
                for nm in self._GROUPS[g]:
                    out[nm] = getattr(self, nm)
            out['perm'] = self.perm
            return out
        for g in groups:
            if self._host.get(g) is None:
                for nm in self._GROUPS[g]:
                    out[nm] = getattr(self, nm)        # empty
                continue
            lists = self._dev.get(g)
            if lists is None:
                if self.strict:
                    raise RuntimeError(
                        f"{what}() on an HBS block whose {g} factors are not "
                        f"prefetched to {self.compute_device}.  Under "
                        "strict=True this is an error instead of a transient "
                        "upload.")
                lists = self._upload(g)
            out.update(lists)
        if 'perm' not in out:
            out['perm'] = self.perm.to(self.compute_device)
        return out

    def prefetch(self, groups=('core',), non_blocking=True):
        """Keep a device mirror of the named groups until evict().
        No-op for groups already mirrored, empty, or when computing on the
        host."""
        if not self._staging():
            return self
        for g in groups:
            if self._host.get(g) is not None and g not in self._dev:
                self._dev[g] = self._upload(g)
        return self

    def evict(self, groups=('core', 'ulv')):
        """Drop the device mirror of the named groups.  Never copies: the
        host master is always current."""
        for g in groups:
            if self._dev.pop(g, None) is not None:
                self.nSpill += 1
        return self

    def release_ulv(self):
        """Drop the ULV mirror, keep the apply mirror."""
        return self.evict(groups=('ulv',))

    def to(self, device, non_blocking=False):
        """Change where operations run.  Moves no data: the factors are
        already on the host, and a mirror is only made by prefetch()."""
        self.evict()
        self.compute_device = torch.device(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, index=None):
        return self.to('cuda' if index is None else f'cuda:{index}')

    def device_nbytes(self, include_ulv=True):
        """Bytes this block currently holds on the device (its mirror)."""
        groups = ('core', 'ulv') if include_ulv else ('core',)
        tot = 0
        for g in groups:
            lists = self._dev.get(g)
            if lists is None:
                continue
            for v in lists.values():
                for t in (v if isinstance(v, list) else [v]):
                    if torch.is_tensor(t):
                        tot += t.nbytes
        return tot

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def construct(self, rk, Om=None, Psi=None, Y=None, Z=None,
                  compute_ULV=False, fast=False, io_perm=None):
        if Om is None:
            if self.A is None:
                raise ValueError("Samples and LinOP cannot both be None")
            # hard requirement max(fac*rk, nl) + rk at the worst level, plus
            # oversampling
            n_max = max(self.nl, 2*rk)
            p = 10
            s = max(n_max + rk + p , (int)(np.ceil(1.5 * n_max)) )
            Om  = np.random.standard_normal(size=(self.A.shape[1], s))
            Psi = np.random.standard_normal(size=(self.A.shape[0], s))
            Y   = self.A @ Om
            Z   = self.A.T @ Psi
        self.constructHBS(rk, Om, Psi, Y, Z, fast=fast, with_ulv=compute_ULV,
                          io_perm=io_perm)

    def _chunk_blocks(self, Nb, n, s, ulv):
        """Blocks per chunk at a level with n rows per block.

        Per-block device working set, in elements: the four sample slices,
        Q_A and Bp (4ns + 2ns), one ns for the upload transpose, one more ns
        of slack for the next-level outputs and temporaries, plus the n x n
        factors (D, M, R'/Q/R when ulv).  Deliberately rough; _CHUNK_BYTES is
        the knob."""
        per = 8 * (8 * n * s + (8 if ulv else 3) * n * n)
        c = Nb
        if _CHUNK_BYTES[0]:
            c = min(c, _CHUNK_BYTES[0] // per)
        if _UV_CHUNK_ELEMS[0]:
            c = min(c, _UV_CHUNK_ELEMS[0] // max(1, n * s))
        return max(1, c)

    def _sample_buffers(self, shape, ws):
        """Four tensors for the next level's samples, and whether they are on
        the host.  Device when they fit in the device budget next to what is
        allocated plus this level's chunk working set `ws`; page-locked host
        (host_pool) otherwise.  See _SAMPLE_SPILL_BYTES."""
        dev = self.compute_device
        total = 4 * _numel(shape) * torch.finfo(self.dtype_t).bits // 8
        lim = _SAMPLE_SPILL_BYTES[0]
        if not self._staging():
            on_host = False
        elif lim is None:
            used = torch.cuda.memory_allocated(dev) if dev.type == 'cuda' else 0
            on_host = used + total + ws > device_budget(dev)
        else:
            on_host = total > lim
        if on_host:
            pin = dev.type == 'cuda'
            return [host_pool.get(_numel(shape), self.dtype_t, pinned=pin).view(shape)
                    for _ in range(4)], True
        return [torch.empty(shape, dtype=self.dtype_t, device=dev)
                for _ in range(4)], False

    def constructHBS(self, rk, Om0, Psi0, Y0, Z0, fast=False, with_ulv=False,
                     io_perm=None):
        """Compress from samples Y = A Om, Z = A^T Psi; optionally ULV too.

        Samples may be numpy arrays or tensors on any device, in any layout;
        host samples stored column-major (host_empty) are read fastest.  A
        sample may arrive boxed -- a one-element list -- which hands over the
        caller's only reference: it is then freed as soon as the leaf level
        is done.  Samples are never written.

        io_perm: the row order the samples are given in (row r holds original
        row io_perm[r]); None means original order.  When it equals this
        block's tree permutation, the leaf read is a contiguous slice instead
        of a gather.

        Each level is processed in chunks of blocks.  A chunk's four sample
        slices are gathered (at the leaf, through perm) where the samples
        live and uploaded; everything the chunk produces is written straight
        to the host master, and its share of the next level's samples to the
        buffers from _sample_buffers.  The ULV step runs on the same chunk
        while its D, U, V are still on the device.  Only the ULV carries
        (Uhat, R22: (Nb, k, k) per level) persist across chunks.

        The next level's samples are formed from small products:

            Y' = U^T (Y - D Om)   = U^T Y - (U^T D) Om
            Z' = V^T (Z - D^T Psi) = V^T Z - (D V)^T Psi

        so no (c, n, s) residual is ever formed, and Y, Z can be dropped as
        soon as U^T Y, V^T Z exist.
        """
        cur = []
        for X in (Om0, Psi0, Y0, Z0):
            if isinstance(X, list):
                X = X.pop()
            if not torch.is_tensor(X):
                X = torch.from_numpy(np.asarray(X))
            cur.append(X)
        Om0 = Psi0 = Y0 = Z0 = X = None

        dev = self.compute_device
        dt  = self.dtype_t
        fac, nl, L = self.fac, self.nl, self.L
        s = cur[0].shape[1]
        levels, root = _level_shapes(self.Nb, nl, L, rk, fac)
        for (Nb_i, n, k) in levels:
            assert s >= n + k, \
                f"undersampled: s={s} < n+k={n + k} at a level with n={n}"
        self.Nbvec    = [lv[0] for lv in levels] + [root[0]]
        self.nSamples = s
        self.NNvec    = np.zeros(1, dtype=np.int64)

        # ---- host masters, allocated once, filled as factors appear -------
        self.evict()
        self._dev.clear()
        rec, lists = _new_master(_core_layout(levels, root), dt)
        self._host['core'] = rec
        for nm, lst in lists.items():
            setattr(self, nm, lst)
        core_pin = rec['bufs'][dt].registered
        if with_ulv:
            rec, lists = _new_master(_ulv_layout(levels, root), dt)
            self._host['ulv'] = rec
            for nm, lst in lists.items():
                setattr(self, nm, lst)
            ulv_pin = rec['bufs'][dt].registered
        else:
            self._host['ulv'] = None
            for nm in self._GROUPS['ulv']:
                setattr(self, nm, [])
            ulv_pin = False
        nD2H = 0

        g_leaf = self._g_host(io_perm)     # None: leaf rows are a slice
        own = []        # host buffers this call took from host_pool for `cur`

        def fetch(X, a, b, n, leaf):
            """Blocks a:b of a level's samples, (b-a, n, s), on dev."""
            if leaf:
                rows = None if g_leaf is None else g_leaf[a * n:b * n]
                R = _rows_to_dev(X, rows, a * n, b * n, dev, dt)
                return R.reshape(b - a, n, s)
            return X[a:b].to(device=dev, dtype=dt, non_blocking=True)

        tic_compress = time.time()
        Uhat = R22 = None       # ULV carries of the level just finished
        leaf = True
        for i, (Nb_i, n, k) in enumerate(levels):
            c = self._chunk_blocks(Nb_i, n, s, with_ulv)
            ws = c * 8 * (8 * n * s + (8 if with_ulv else 3) * n * n)
            nxt, nxt_host = self._sample_buffers((Nb_i, k, s), ws)
            pin_nxt = nxt_host and dev.type == 'cuda'
            if with_ulv:
                Uhat_n = torch.empty((Nb_i, k, k), dtype=dt, device=dev)
                R22_n  = torch.empty((Nb_i, k, k), dtype=dt, device=dev)

            for a in range(0, Nb_i, c):
                b = min(a + c, Nb_i)
                tic = time.time()
                Om, Psi, Y, Z = (fetch(X, a, b, n, leaf) for X in cur)
                self.setupTime += time.time() - tic
                self.tSample   += time.time() - tic

                tic = time.time()
                U, M_om = _uv_side(Om, Y, k, fast)
                UtY = torch.bmm(U.mT, Y)
                del Y
                V, M_psi = _uv_side(Psi, Z, k, fast)
                VtZ = torch.bmm(V.mT, Z)
                del Z
                self.nullTime += time.time() - tic

                tic = time.time()
                D = construct_D(U, V, M_om, M_psi)
                del M_om, M_psi
                self.DTime += time.time() - tic

                # next-level samples, (c, k, s) each
                _put(nxt[2][a:b], torch.baddbmm(
                    UtY, torch.bmm(U.mT, D), Om, alpha=-1.0), pin_nxt)
                del UtY
                _put(nxt[3][a:b], torch.baddbmm(
                    VtZ, torch.bmm(D, V).mT, Psi, alpha=-1.0), pin_nxt)
                del VtZ
                _put(nxt[0][a:b], torch.bmm(V.mT, Om), pin_nxt)
                _put(nxt[1][a:b], torch.bmm(U.mT, Psi), pin_nxt)
                del Om, Psi

                _put(self.Umats[i][a:b], U, core_pin)
                _put(self.Vmats[i][a:b], V, core_pin)
                _put(self.Dmats[i][a:b], D, core_pin)
                nD2H += U.nbytes + V.nbytes + D.nbytes

                if with_ulv:
                    tic = time.time()
                    ch = slice(fac * a, fac * b)
                    Q, W1, Ru, Uu, Ud, R22c = _ulv_chunk(
                        D, U, V,
                        None if Uhat is None else Uhat[ch],
                        None if R22 is None else R22[ch])
                    for nm, t in (('Qlist', Q), ('Wlist', W1),
                                  ('Rlist', Ru), ('Uulist', Uu)):
                        _put(getattr(self, nm)[i][a:b], t, ulv_pin)
                        nD2H += t.nbytes
                    Uhat_n[a:b] = Ud
                    R22_n[a:b]  = R22c
                    del Q, W1, Ru, Uu, Ud, R22c
                    self.tULV += time.time() - tic
                del U, V, D

            if with_ulv:
                self.NNvec = np.append(self.NNvec, self.NNvec[-1] + (n - k) * Nb_i)
                Uhat, R22 = Uhat_n, R22_n
                Uhat_n = R22_n = None
            # The next level's blocks are fac consecutive blocks of this one:
            # (Nb_i, k, s) is (Nb_i/fac, fac*k, s) in the same memory.  This
            # rebinding frees the current level's samples (the caller's, at
            # the leaf, if they were boxed).
            cur = [X.view(Nb_i // fac, fac * k, s) for X in nxt]
            # Return the previous level's pool buffers.  Only ones this call
            # took: a caller's samples may come from the same pool.  Safe
            # without a sync: their pending reads and any later DMA into them
            # are ordered on the same stream, and no host code touches them.
            for X in own:
                host_pool.put(X)
            own = list(nxt) if nxt_host else []
            del nxt
            leaf = False

        # ---- root: one block (or a few), so no chunking -------------------
        Nb_r, n_r = root
        tic = time.time()
        Om = fetch(cur[0], 0, Nb_r, n_r, leaf)
        Y  = fetch(cur[2], 0, Nb_r, n_r, leaf)
        cur = None
        self.setupTime += time.time() - tic
        tic = time.time()
        D = block_solve_r(Y, Om)
        del Om, Y
        self.blockSolveTime += time.time() - tic
        _put(self.Dmats[-1], D, core_pin)
        nD2H += D.nbytes
        if with_ulv:
            tic = time.time()
            Q, R = _ulv_root(D, Uhat, R22)
            _put(self.Qlist[-1], Q, ulv_pin)
            _put(self.Rlist[-1], R, ulv_pin)
            nD2H += Q.nbytes + R.nbytes
            self.NNvec = np.append(self.NNvec, self.NNvec[-1] + Nb_r * R.shape[1])
            del Q, R
            self.tULV += time.time() - tic
        del D, Uhat, R22

        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)       # masters complete before use
        for X in own:                         # after the sync: no DMA in flight
            host_pool.put(X)
        if self._staging():
            self.bytesD2H += nD2H
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

    def matmat(self, v, chunk=None, out=None, alpha=1.0, beta=0.0, C=None,
               io_perm=None):
        return self._apply(v, transpose=False, chunk=chunk, out=out,
                           alpha=alpha, beta=beta, C=C, io_perm=io_perm)

    def rmatmat(self, v, chunk=None, out=None, alpha=1.0, beta=0.0, C=None,
                io_perm=None):
        return self._apply(v, transpose=True, chunk=chunk, out=out,
                           alpha=alpha, beta=beta, C=C, io_perm=io_perm)

    # -- row order of inputs/outputs ------------------------------------
    def _g_host(self, io_perm):
        """Host index g with g[r] = the row of an io_perm-ordered tensor that
        holds original row perm[r], i.e. inv(io_perm)[perm]; None when that is
        the identity (io_perm == perm), so no gather is needed at all.
        io_perm=None means original order, g = perm."""
        cache = self.__dict__.setdefault('_gcache', {})
        key = id(io_perm)
        ent = cache.get(key)
        if ent is not None and ent[0] is io_perm and ent[1] is self._perm:
            return ent[2]
        p = self.perm
        if io_perm is None:
            g = p
        else:
            io = torch.as_tensor(io_perm, dtype=torch.int64).cpu()
            inv = torch.empty_like(io)
            inv[io] = torch.arange(io.numel(), dtype=torch.int64)
            g = inv[p]
        if torch.equal(g, torch.arange(g.numel(), dtype=torch.int64)):
            g = None
        cache[key] = (io_perm, self._perm, g)
        cache.pop(('dev', key), None)
        return g

    def _gidx(self, io_perm, dev):
        """_g_host on dev (cached)."""
        g = self._g_host(io_perm)
        if g is None:
            return None
        cache = self._gcache
        ent = cache.get(('dev', id(io_perm)))
        if ent is not None and ent[0] is g and _dev_eq(ent[1].device, dev):
            return ent[1]
        gd = g.to(dev)
        cache[('dev', id(io_perm))] = (g, gd)
        return gd

    # -- column chunks ----------------------------------------------------
    def _ws_per_col(self, kind):
        """Device bytes one right-hand-side column costs in an apply or a ULV
        solve: the per-level vectors of the sweeps.  R is the total row count
        over all levels; m the matrix size.  Conservative (calibrated on CPU,
        see the module tests)."""
        R = sum(int(D.shape[0]) * int(D.shape[1]) for D in self.Dmats)
        m = int(self.shape[0])
        if kind == 'apply':
            return 8 * (2 * R + 4 * m)
        return 8 * (3 * R + 6 * m)

    def _cols(self, kind, ncol, chunk):
        if chunk is not None:
            c = chunk or ncol
        else:
            cap = _MATMAT_CHUNK[0] if kind == 'apply' else _SOLVE_CHUNK[0]
            c = ncol
            if cap:
                c = min(c, cap)
            if _STREAM_BYTES[0]:
                c = min(c, max(1, _STREAM_BYTES[0] // self._ws_per_col(kind)))
        return max(1, min(c, max(ncol, 1)))

    @staticmethod
    def _as_2d(X):
        if X is None:
            return None
        if not torch.is_tensor(X):
            X = torch.from_numpy(np.asarray(X))
        return X[:, None] if X.ndim == 1 else X

    def _stream(self, X, target, C, alpha, beta, c, body):
        """target[:, j] = alpha * body(X[:, j] on dev) + beta * C[:, j], one
        column chunk j at a time (C may be target itself).  X, target and C
        may each live on the host or the device; host chunks move with one
        DMA each when the tensor is column-major.  Synchronizes once at the
        end if a device-to-host copy may still be in flight."""
        dev, dt = self.compute_device, self.dtype_t
        ncol = X.shape[1]
        pending = False
        for j0 in range(0, ncol, c):
            j1 = min(j0 + c, ncol)
            xc = _to_dev(X[:, j0:j1], dev)
            if xc.dtype != dt:
                xc = xc.to(dt)
            y = body(xc)
            del xc
            if alpha != 1.0:
                y.mul_(alpha)
            if beta != 0.0:
                cc = _to_dev(C[:, j0:j1], dev)
                y.add_(cc.to(dt) if cc.dtype != dt else cc, alpha=beta)
                del cc
            pending |= _from_dev(target[:, j0:j1], y)
            del y
        if pending:
            torch.cuda.current_stream(dev).synchronize()

    def _apply(self, v, transpose, chunk=None, out=None, alpha=1.0, beta=0.0,
               C=None, io_perm=None):
        """A v (transpose=False) or A^T v (transpose=True).

        Without out:
            numpy in  -> numpy out (host)
            tensor in -> tensor on compute_device
            1-D in    -> 1-D out
        With out (a tensor anywhere, or a writeable float64 numpy array):
            out <- alpha * op(v) + beta * C     (C defaults to out)
        computed chunk by chunk on the device and written back; returns out.

        v is streamed in column chunks (_cols), so neither v nor the result
        ever exists in full on the device unless it lives there already.
        io_perm: row order of v, out and C (see constructHBS)."""
        if v.ndim not in (1, 2):
            raise ValueError(f"expected 1-D or 2-D input, got ndim={v.ndim}")
        numpy_input = isinstance(v, np.ndarray)
        was_vector  = (v.ndim == 1)
        X = self._as_2d(v)

        n_in  = int(self.shape[0] if transpose else self.shape[1])
        n_out = int(self.shape[1] if transpose else self.shape[0])
        what = 'rmatmat' if transpose else 'matmat'
        if X.shape[0] != n_in:
            raise ValueError(f"{what}: expected {n_in} rows, got {X.shape[0]}")

        dev, dt = self.compute_device, self.dtype_t
        ncol = X.shape[1]
        if out is None:
            if beta != 0.0 and C is None:
                raise ValueError(f"{what}: beta != 0 needs out or C")
            target = torch.empty((n_out, ncol), dtype=dt,
                                 device=_HOST if numpy_input else dev)
        else:
            target = self._as_2d(out)
            if tuple(target.shape) != (n_out, ncol):
                raise ValueError(f"{what}: out has shape {tuple(target.shape)}, "
                                 f"expected {(n_out, ncol)}")
        Cm = (target if C is None else self._as_2d(C)) if beta != 0.0 else None

        F = self._factors(('core',), what)
        g = self._gidx(io_perm, dev)
        self._stream(X, target, Cm, alpha, beta,
                     self._cols('apply', ncol, chunk),
                     lambda xc: self._apply_2d(xc, transpose, F, g))
        del F

        if out is not None:
            return out
        res = target[:, 0] if was_vector else target
        return res.numpy() if numpy_input else res

    def _apply_2d(self, V, transpose, F, g):
        """Core HBS apply.  V: 2-D tensor on compute_device in io order.
        F: factor lists on the same device (from _factors).  g: device gather
        index into V's rows (_gidx), or None when V is already in tree order."""
        dmode = 'T' if transpose else 'N'
        down  = F['Umats'] if transpose else F['Vmats']
        up    = F['Vmats'] if transpose else F['Umats']
        Dm    = F['Dmats']

        VV = [V.contiguous() if g is None else V[g]]
        for M in down:
            VV.append(block_matvec(M, VV[-1], mode='T'))
        u = block_matvec(Dm[-1], VV[-1], mode=dmode)
        for lvl in range(len(up) - 1, -1, -1):
            u = block_matvec(up[lvl], u) \
              + block_matvec(Dm[lvl], VV[lvl], mode=dmode)
        if g is None:
            return u
        out = torch.empty_like(u)
        out[g] = u
        return out

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def compute_ULV(self):
        """Factorize an already-compressed block (built with
        compute_ULV=False, or adopted through set_mats).

        Streams D, U, V from the host master level by level, in the same
        chunks as construction, and writes the factors to a new host master.
        constructHBS(with_ulv=True) is cheaper, since there the level factors
        are already on the device."""
        tic = time.time()
        dev, dt, fac = self.compute_device, self.dtype_t, self.fac
        levels = [tuple(U.shape) for U in self.Umats]
        root   = tuple(self.Dmats[-1].shape[:2])
        rec, lists = _new_master(_ulv_layout(levels, root), dt)
        pin = rec['bufs'][dt].registered
        self._dev.pop('ulv', None)
        s = getattr(self, 'nSamples', 1)
        up = lambda t: t.to(device=dev, dtype=dt)

        NNvec = [0]
        Uhat = R22 = None
        for i, (Nb_i, n, k) in enumerate(levels):
            c = self._chunk_blocks(Nb_i, n, max(s, n), True)
            Uhat_n = torch.empty((Nb_i, k, k), dtype=dt, device=dev)
            R22_n  = torch.empty((Nb_i, k, k), dtype=dt, device=dev)
            for a in range(0, Nb_i, c):
                b = min(a + c, Nb_i)
                ch = slice(fac * a, fac * b)
                Q, W1, Ru, Uu, Ud, R22c = _ulv_chunk(
                    up(self.Dmats[i][a:b]), up(self.Umats[i][a:b]),
                    up(self.Vmats[i][a:b]),
                    None if Uhat is None else Uhat[ch],
                    None if R22 is None else R22[ch])
                for nm, t in (('Qlist', Q), ('Wlist', W1),
                              ('Rlist', Ru), ('Uulist', Uu)):
                    _put(lists[nm][i][a:b], t, pin)
                Uhat_n[a:b] = Ud
                R22_n[a:b]  = R22c
                del Q, W1, Ru, Uu, Ud, R22c
            NNvec.append(NNvec[-1] + (n - k) * Nb_i)
            Uhat, R22 = Uhat_n, R22_n
        Q, R = _ulv_root(up(self.Dmats[-1]), Uhat, R22)
        _put(lists['Qlist'][-1], Q, pin)
        _put(lists['Rlist'][-1], R, pin)
        NNvec.append(NNvec[-1] + root[0] * R.shape[1])
        del Q, R, Uhat, R22
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)

        self._host['ulv'] = rec
        for nm, lst in lists.items():
            setattr(self, nm, lst)
        self.NNvec = np.asarray(NNvec, dtype=np.int64)
        self.tULV = time.time() - tic

    def solve(self, b, mode='N', chunk=None, overwrite_b=False, out=None,
              io_perm=None):
        """Apply the inverse (mode='N') or inverse transpose (mode='T') of the
        HBS operator via its ULV factorization.

        Permutation convention: the factors represent A_p = P A P^T with
        (P v)[i] = v[perm[i]], so A^{-1} = P^T A_p^{-1} P and likewise for the
        transpose; only the sweep inside ULVsparse.solve differs by mode.

        b is streamed in column chunks (_cols: _STREAM_BYTES, _SOLVE_CHUNK),
        so a host b never exists in full on the device.  Results:

            out given    -> written into out (anywhere), returns out
            overwrite_b  -> written into b itself (a torch tensor of the
                            factor dtype, or a writeable float64 numpy array)
            otherwise    numpy in -> numpy out;  tensor in -> tensor on
                         compute_device

        Overwriting is safe: each chunk is copied to the device before its
        solution is written back.  io_perm: row order of b and the result.
        """
        if not self.Qlist:
            raise RuntimeError(
                "solve() requires the ULV factorization, but this HBSMAT has "
                "an empty Qlist.  Build it with construct(..., compute_ULV=True) "
                "or call compute_ULV() first."
            )
        if mode not in ('N', 'T'):
            raise NotImplementedError(f"mode '{mode}' not recognized. Use 'N' or 'T'.")

        input_is_numpy = isinstance(b, np.ndarray)
        if not (input_is_numpy or torch.is_tensor(b)):
            raise TypeError("b must be either a numpy.ndarray or a torch.Tensor")
        if b.ndim not in (1, 2):
            raise ValueError(f"b must have ndim 1 or 2, got ndim={b.ndim}")

        dev, dt = self.compute_device, self.dtype_t
        B = self._as_2d(b)
        was_vector = (b.ndim == 1)
        nrow, ncol = B.shape
        if nrow != self.perm.shape[0]:
            raise ValueError(f"solve: expected {self.perm.shape[0]} rows, got {nrow}")

        if out is not None:
            target, ret = self._as_2d(out), out
        elif overwrite_b and B.dtype == dt and (
                not input_is_numpy or b.flags.writeable):
            target, ret = B, b
        else:
            target = torch.empty((nrow, ncol), dtype=dt,
                                 device=_HOST if input_is_numpy else dev)
            res = target[:, 0] if was_vector else target
            ret = res.numpy() if input_is_numpy else res

        F = self._factors(('core', 'ulv'), 'solve')
        g = self._gidx(io_perm, dev)

        def body(xc):
            # u = P^T ( ULV^{-1} ( P b ) ) on one chunk; the gather is a copy
            bp = xc.contiguous() if g is None else xc[g]
            u = ULVsparse.solve(
                F['Umats'], F['Dmats'], F['Qlist'], F['Wlist'], F['Uulist'],
                F['Rlist'], self.NNvec, bp, device=dev, mode=mode,
                Vmats=F['Vmats'])
            del bp
            if g is None:
                return u
            r = torch.empty_like(u)
            r[g] = u
            return r

        self._stream(B, target, None, 1.0, 0.0,
                     self._cols('solve', ncol, chunk), body)
        del F
        return ret
