import numpy as np
import scipy.linalg as splinalg
import time
import matAssembly.HBS.ULVsparse_torch as ULVsparse
import torch.linalg as tla
import torch
import matAssembly.HBS.HBSnew as HBSnew
#sparse block matrix operations

_UV_TQR    = [0.0]      # the QR of W = [Om^T | Y^T]
_UV_TBASIS = [0.0]      # eigh/svd extraction of U from L
_UV_TSETUP = [0.0]      # allocation and the two copies into W
_UV_NCALL  = [0]
_UV_SYNC   = [False]
_UV_MODE = ['ne']
_MATMAT_CHUNK = [256]
_EIGH_CHUNK = [128]
_PIN_HOST = [True]      # page-lock host masters; set False if host RAM is tight
_SOLVE_CHUNK = [512]


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

def to_block_tensor(M, n, b):
    """(n*b, s) -> (n, b, s) block tensor (analogue of convert_to_torch_tens)."""
    s = M.shape[1]
    return M.reshape(n, b, s)

def _rsolve(P, B, fast=False, rtol=None, QR=None):
    """
    Batched min-residual solution of  X @ B = P   (X = P B^+).
    B: (Nb, n, s) with s >= n.  P: (Nb, ny, s).  Returns (Nb, ny, n).
    Solves the transposed system B^T X^T = P^T, which is tall/overdetermined.
    fast=True : QR + triangular solve (assumes full column rank of B^T).
    fast=False: SVD applied factor-by-factor with an explicit relative cutoff.
    QR        : optional precomputed (Q, R) of B.mT, see note below.
    """
    assert B.shape[-1] >= B.shape[-2], "undersampled: s < n, this is a different problem"

    #if fast:
    #    if QR is None:
    #        Q, R = tla.qr(B.mT, mode='reduced')      # Q:(Nb,s,n)  R:(Nb,n,n)
    #    else:
    #        Q, R = QR
    #    PQ = torch.bmm(P, Q)                          # (Nb, ny, n)
    #    # X R^T = PQ ;  R^T is lower triangular, solve from the right
    #    return torch.linalg.solve_triangular(R.mT, PQ, upper=False, left=False)

    U, S, Vh = tla.svd(B, full_matrices=False)        # U:(Nb,n,n) S:(Nb,n) Vh:(Nb,n,s)
    if rtol is None:
        rtol = max(B.shape[-1], B.shape[-2]) * torch.finfo(B.dtype).eps
    Sinv = torch.where(S > rtol * S[..., :1], 1.0 / S, torch.zeros_like(S))
    # X = P V S^+ U^T, applied right-to-left; the (s,n) pinv is never built
    return torch.bmm(torch.bmm(P, Vh.mT) * Sinv.unsqueeze(-2), U.mT)

def block_solve_r(A,B,device,fast=False):
    # batched: solves X[i] = A[i] @ pinv(B[i]) over the block dim
    return _rsolve(A, B, fast=fast)

def block_mult(A,B,device,mode='N'):
    # A: (Nb, n, k), B: (Nb, k, m)
    if mode=='N':
        return torch.bmm(A, B)
    elif mode=='T':
        return torch.bmm(A.mT, B)
    else:
        raise ValueError("mode not recognized")

def block_matvec(A,B,device,mode='N'):
    # A: (Nb, n, k), B: (Nb*nB, col) flat
    Nb  = A.shape[0]
    k   = B.shape[1]
    nB  = B.shape[0] // Nb
    Bm  = B.reshape(Nb, nB, k)
    if mode=='N':
        return torch.bmm(A, Bm).reshape(Nb * A.shape[1], k)
    elif mode=='T':
        return torch.bmm(A.mT, Bm).reshape(Nb * A.shape[2], k)
    else:
        raise ValueError("mode not recognized")
def block_mult_and_reduce(A,B,fac,device,mode='N'):
    # A: (Nb, n, rk), B: (Nb, rk, s) or (Nb, n, s)
    # After bmm: (Nb, *, s); reshape to group fac blocks together
    Nb = A.shape[0]
    if mode=='N':
        C = torch.bmm(A, B)                             # (Nb, n, s)
        return C.reshape(Nb//fac, fac*A.shape[1], B.shape[2])
    elif mode=='T':
        C = torch.bmm(A.mT, B)                         # (Nb, rk, s)
        return C.reshape(Nb//fac, fac*A.shape[2], B.shape[2])
    else:
        raise ValueError("mode not recognized")


def _small_pinv_factors(R, s=None, rtol=None):
    """B = R^T Q^T  =>  pinv(R^T) = Vh^T diag(Sinv) Uc^T, all n x n."""
    n = R.shape[-1]
    Uc, S, Vhc = tla.svd(R.mT, full_matrices=False)
    if rtol is None:
        rtol = max(s if s is not None else n, n) * torch.finfo(R.dtype).eps
    Sinv = torch.where(S > rtol*S[...,:1], S.reciprocal(), torch.zeros_like(S))
    return Uc, Sinv, Vhc


def _rsolve_qr(P, QR, s=None, fast=False):
    """P B^+ where B = R^T Q^T. Replaces _rsolve(P, B)."""
    Q, R = QR
    PQ = torch.bmm(P, Q)                                  # (Nb, ny, n)
    if fast:
        return torch.linalg.solve_triangular(R.mT, PQ, upper=False, left=False)
    Uc, Sinv, Vhc = _small_pinv_factors(R, s=s)
    return torch.bmm(torch.bmm(PQ, Vhc.mT) * Sinv.unsqueeze(-2), Uc.mT)


def _pinv_apply_left(QR, U, s=None, fast=False):
    """B^+ U = Q pinv(R^T) U, applied to k columns."""
    Q, R = QR
    if fast:
        T = torch.linalg.solve_triangular(R.mT, U, upper=False, left=True)
    else:
        Uc, Sinv, Vhc = _small_pinv_factors(R, s=s)
        T = torch.bmm(Vhc.mT, Sinv.unsqueeze(-1) * torch.bmm(Uc.mT, U))
    return torch.bmm(Q, T)                                # (Nb, s, k)
def _tri_rsolve_T(R, P, rcond):
    """P (R^T)^+ , R upper triangular n x n.  Triangular solve where R is
    safely conditioned; SVD pseudo-inverse on the (rare) entries that are not.

    Om is Gaussian with s >= n + k, so R_oo is well conditioned with
    overwhelming probability -- the SVD branch is a guard, not the norm."""
    d   = torch.diagonal(R, dim1=-2, dim2=-1).abs()
    bad = (d.amin(-1) <= rcond * d.amax(-1).clamp_min(torch.finfo(R.dtype).tiny))

    X = torch.linalg.solve_triangular(R.mT, P, upper=False, left=False)

    if bool(bad.any()):                            # one sync, only when needed
        idx = bad.nonzero(as_tuple=True)[0]
        Uc, S, Vhc = tla.svd(R[idx].mT, full_matrices=False)
        Sinv = torch.where(S > rcond * S[..., :1], S.reciprocal(),
                           torch.zeros_like(S))
        X[idx] = torch.bmm(torch.bmm(P[idx], Vhc.mT) * Sinv.unsqueeze(-2),
                           Uc.mT)
    return X
def _tri_lsolve_T(R, U, rcond=1e-12):
    d   = torch.diagonal(R, dim1=-2, dim2=-1).abs()
    tiny = torch.finfo(R.dtype).tiny
    bad = d.amin(-1) <= rcond * d.amax(-1).clamp_min(tiny)

    T = torch.linalg.solve_triangular(R.mT, U, upper=False, left=True)

    if bool(bad.any()):                   # one D2H sync, only when it fires
        idx = bad.nonzero(as_tuple=True)[0]
        Uc, S, Vhc = tla.svd(R[idx].mT, full_matrices=False)
        Sinv = torch.where(S > rcond * S[..., :1], S.reciprocal(),
                           torch.zeros_like(S))
        # (R^T)^+ U = V S^+ U_c^T U
        T[idx] = torch.bmm(Vhc.mT, Sinv.unsqueeze(-1) * torch.bmm(Uc.mT, U[idx]))
    return T
def _spd_factor(G, check=True):
    """Factor G = Om Om^T.  Om is (n, s) Gaussian with s >= n + k, so G is SPD
    with cond ~ ((1+sqrt(n/s))/(1-sqrt(n/s)))^2 -- about 100 at s/n = 1.5 and
    14 at s/n = 3.  Cholesky is the right tool; the LU fallback is a guard.

    Returns an opaque handle for _spd_solve.  `check` costs one D2H sync per
    call; drop it once the path is validated."""
    L, info = torch.linalg.cholesky_ex(G)
    if check and bool(info.any()):
        d = torch.diagonal(G, dim1=-2, dim2=-1).mean(-1)
        eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
        L, info = torch.linalg.cholesky_ex(G + (1e-13 * d)[:, None, None] * eye)
        if bool(info.any()):
            return ('lu', torch.linalg.lu_factor(G))
    return ('chol', L)
def _spd_solve(fac, B):
    kind, F = fac
    if kind == 'chol':
        return torch.cholesky_solve(B, F)
    LU, piv = F
    return torch.linalg.lu_solve(LU, piv, B)
def _construct_D_ne(U, V, M_om, M_psi):
    """D = (I - UU*) Y Om^+  +  U [ (I - VV*) Z Psi^+ ]* U*

    M_om = Y Om^+ and M_psi = Z Psi^+ were already formed by the
    normal-equations path, so this is four GEMMs and nothing else -- no
    triangular solves, no SVD fallback, no width-s intermediate.

    Equivalent to construct_D's QR branch term for term:
      R_oy^T R_oo^{-T} = Y Om^T (R_oo^T R_oo)^{-1} = Y Om^T G^{-1} = M_om
      R_pz^T R_pp^{-T} = Z Psi^T G_psi^{-1}                        = M_psi
    """
    P = M_om - torch.bmm(U, torch.bmm(U.mT, M_om))     # (Nb, ny, n)
    Gk = torch.bmm(M_psi, U)                           # (Nb, ny, k)
    Gk = Gk - torch.bmm(V, torch.bmm(V.mT, Gk))
    return P + torch.bmm(U, Gk.mT)
def construct_D(U, V, om_R, psi_R, fast=True, rcond=1e-12):
    """D = (I-UU*) Y Om^+  +  U [ (I-VV*) Z Psi^+ ]^* U ... (see derivation)

    Uses  Y Q_om = R_oy^T  and  Z Q_psi = R_pz^T, so neither Q nor any
    width-s intermediate is ever formed.  All work is n x n and n x k.
    """
    if len(om_R)==1:
        return _construct_D_ne(U,V,om_R[0],psi_R[0])
    R_oo, R_oy = om_R
    R_pp, R_pz = psi_R

    # ---- term 1:  (I - UU*) R_oy^T R_oo^{-T} --------------------------
    P = R_oy.mT                                   # (Nb, ny, n) == Y Q_om
    P = P - torch.bmm(U, torch.bmm(U.mT, P))      # project, width n not s
    term1 = _tri_rsolve_T(R_oo, P, rcond)         # X R_oo^T = P

    # ---- term 2:  U [ (I - VV*) R_pz^T R_pp^{-T} U ]^* ----------------
    T  = _tri_lsolve_T(R_pp, U, rcond)            # R_pp^T T = U   -> (Nb,n,k)
    G  = torch.bmm(R_pz.mT, T)                    # (Nb, ny, k)
    G  = G - torch.bmm(V, torch.bmm(V.mT, G))
    term2 = torch.bmm(U, G.mT)

    return term1 + term2
def _qr_R_only(W, mode='house', jitter=1e-12):
    p = W.shape[-1]
    A,tau = torch.geqrf(W)

    return torch.triu(A[...,:p,:])#tla.qr(W, mode='r').R
def _eigh_topk(S, k, chunk=None):
    """Top-k eigenvectors of a batch of SPD matrices, chunked over the batch.

    cusolver's batched syevd workspace scales with batch size; at
    (512, 512, 512) float64 it asks for 2.19 GB in one allocation on top of
    the eigenvector matrix.  The eigh is ~7% of the sweep, so chunking it is
    close to free."""
    c = _EIGH_CHUNK[0] if chunk is None else chunk
    Nb = S.shape[0]
    if not c or Nb <= c:
        return tla.eigh(S).eigenvectors[..., -k:].flip(-1)
    out = torch.empty((Nb, S.shape[-1], k), dtype=S.dtype, device=S.device)
    for j in range(0, Nb, c):
        out[j:j+c] = tla.eigh(S[j:j+c]).eigenvectors[..., -k:].flip(-1)
    return out
def _compute_UV_pair_ne(Om, Y, Psi, Z, k, fast=False):
    """Normal-equations form of compute_UV_pair.  The large QR never exists.

        G  = Om Om^T                (Nb, n, n)   SPD, cond ~ 100
        M  = (G^{-1} Om Y^T)^T      (Nb, ny, n)  = Y Om^+
        Bp = Y - M Om               (Nb, ny, s)  = Y (I - Om^+ Om)
        S  = Bp Bp^T                (Nb, ny, ny) = L L^T of the QR path
        U  = top-k eigenvectors of S

    S is identical to the QR path's L L^T in exact arithmetic.  Bp is formed
    explicitly rather than as Y Y^T - (Om Y^T)^T G^{-1}(Om Y^T): both are the
    same matrix, but the subtraction form carries two powers of ||Y||/||Bp||
    (the diagonal-dominance ratio) in its error, while Bp = Y - M Om carries
    one -- the same as the Householder QR it replaces.

    The two sides run sequentially rather than stacked into a 2Nb batch.
    Stacking was measured at exactly zero speedup on the QR path (null_qr
    130.01 s either way, since a per-matrix loop just loops twice as long) and
    it doubles peak memory, which is now what binds.
    """
    U_om,  M_om  = _uv_ne_side(Om,  Y, k)
    U_psi, M_psi = _uv_ne_side(Psi, Z, k)
    return (U_om, (M_om,)), (U_psi, (M_psi,))


def _uv_ne_side(A, B, k):
    """One side of the pair; see _compute_UV_pair_ne for the algebra."""
    Nb, ny, s = B.shape

    _uv_sync(B.device); _t = time.time()
    G = torch.bmm(A, A.mT)
    G = 0.5 * (G + G.mT)
    C = torch.bmm(A, B.mT)                  # (Nb, n, ny) = Om Y^T
    fac = _spd_factor(G)
    del G
    X = _spd_solve(fac, C)                  # G^{-1} Om Y^T
    del C, fac
    M = X.mT.contiguous()                   # (Nb, ny, n) = Y Om^+
    del X
    _uv_sync(B.device); _UV_TQR[0] += time.time() - _t

    _uv_sync(B.device); _t = time.time()
    Bp = torch.baddbmm(B, M, A, beta=1.0, alpha=-1.0)    # Y - M Om
    S = torch.bmm(Bp, Bp.mT)
    del Bp
    S = 0.5 * (S + S.mT)
    UU = _eigh_topk(S, k)
    del S
    _uv_sync(B.device); _UV_TBASIS[0] += time.time() - _t
    return UU, M
def compute_UV(Om, Y, rk, device, fast=False):
    """Returns (U, R_oo, R_oy) where W = [Om^T | Y^T] = Q R,
       R_oo = R[:, :n, :n]  (upper triangular, n x n)
       R_oy = R[:, :n, n:]  (n x ny)   -- note  Y Q_om = R_oy^T  exactly.
       Q is never formed: nothing downstream needs it."""
    Nb, ny, s = Y.shape
    n = Om.shape[1]
    k = min(rk, ny)
    print(f"  lvl-shape Nb={Nb:5d} s={s:5d} n={n:4d} ny={ny:5d} k={k:4d}")
    assert s >= n + k
    _UV_NCALL[0] += 1
    _uv_sync(Y.device); _t = time.time()
    W = torch.empty((Nb, s, n + ny), dtype=Y.dtype, device=Y.device)
    W[:, :, :n].copy_(Om.mT)
    W[:, :, n:].copy_(Y.mT)
    _uv_sync(Y.device); _UV_TSETUP[0] += time.time() - _t
    _uv_sync(Y.device); _t = time.time()
    prev = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library('magma')
    R = _qr_R_only(W, mode=_UV_QR_MODE[0])
    torch.backends.cuda.preferred_linalg_library(prev)
    _uv_sync(Y.device); _UV_TQR[0] += time.time() - _t
    R_oo = R[:, :n, :n]
    R_oy = R[:, :n, n:]
    L    = R[:, n:, n:].mT                    # (Nb, ny, r-n)
    _uv_sync(Y.device); _t = time.time()
    if fast:
        G = torch.bmm(L, L.mT); G = 0.5 * (G + G.mT)
        U = tla.eigh(G).eigenvectors[..., -k:].flip(-1)
    else:
        U = tla.svd(L.contiguous(), full_matrices=False).U[..., :k]
    _uv_sync(Y.device); _UV_TBASIS[0] += time.time() - _t
    return U, (R_oo.contiguous(), R_oy.contiguous())
def compute_UV_pair(Om, Y, Psi, Z, rk, device, fast=False,mode=None):
    """Both halves of a level's basis computation in one batched call.

    compute_UV(Om, Y, ...) and compute_UV(Psi, Z, ...) operate on identically
    shaped inputs -- Om and Psi are both (Nb, n, s), Y and Z both (Nb, ny, s)
    -- so the two factorizations can be stacked along the batch dimension and
    issued as one.  That doubles Nb for the QR and the basis extraction,
    which together are ~90% of compression time and sit on a scaling curve
    that is still improving at these batch sizes.

    Returns ((U, om_R), (V, psi_R)), matching two compute_UV calls exactly.
    The Om/Y result occupies batch entries [:Nb], Psi/Z entries [Nb:].
    """
    Nb, ny, s = Y.shape
    n = Om.shape[1]
    k = min(rk, ny)

    assert Psi.shape == Om.shape and Z.shape == Y.shape, \
        "compute_UV_pair needs matching shapes; call compute_UV twice instead"
    assert s >= n + k, \
        "undersampled: not enough columns left after projecting off Om"

    _UV_NCALL[0] += 2          # counts as two logical compute_UV calls

    if (mode or _UV_MODE[0]) == 'ne':
        return _compute_UV_pair_ne(Om, Y, Psi, Z, k, fast=fast)

    # ---- setup: one arena, four copies -----------------------------------
    _uv_sync(Y.device); _t = time.time()
    W = torch.empty((2 * Nb, s, n + ny), dtype=Y.dtype, device=Y.device)
    W[:Nb, :, :n].copy_(Om.mT)
    W[:Nb, :, n:].copy_(Y.mT)
    W[Nb:, :, :n].copy_(Psi.mT)
    W[Nb:, :, n:].copy_(Z.mT)
    _uv_sync(Y.device); _UV_TSETUP[0] += time.time() - _t
    # ---- one QR over the doubled batch -----------------------------------
    _uv_sync(Y.device); _t = time.time()
    R = tla.qr(W, mode='r').R              # (2Nb, r, n+ny), r = min(s, n+ny)
    _uv_sync(Y.device); _UV_TQR[0] += time.time() - _t

    # W is dead here and is the largest tensor in the routine; dropping it
    # before the basis extraction keeps peak usage close to the unmerged
    # version rather than holding W and G simultaneously.
    del W

    L = R[:, n:, n:].mT                    # (2Nb, ny, r-n)

    # ---- basis extraction, also over the doubled batch --------------------
    _uv_sync(Y.device); _t = time.time()
    if fast:
        G = torch.bmm(L, L.mT)
        G = 0.5 * (G + G.mT)
        UU = tla.eigh(G).eigenvectors[..., -k:].flip(-1)
        del G
    else:
        UU = tla.svd(L.contiguous(), full_matrices=False).U[..., :k]
    _uv_sync(Y.device); _UV_TBASIS[0] += time.time() - _t

    # .contiguous() on every slice: these are views into the (2Nb, r, n+ny)
    # R, and holding any one of them alive would keep the whole thing
    # allocated for as long as the returned tuples live.
    om_R  = (R[:Nb, :n, :n].contiguous(), R[:Nb, :n, n:].contiguous())
    psi_R = (R[Nb:, :n, :n].contiguous(), R[Nb:, :n, n:].contiguous())
    U_out = UU[:Nb].contiguous()
    V_out = UU[Nb:].contiguous()

    return (U_out, om_R), (V_out, psi_R)


class HBSMAT:
    """

    HBS mat in new framework

    @init:
            linear operator A
            tree on DOFS (symmetric)
            target rank k

    @constructs: 
            HBS approximation to the source-target map
    @implements:
            matvec (normal/transpose)

    Device policy
    -------------
    Every tensor the object stores lives on self.device.  There is no per-level
    exception: Dmats is handled exactly like Umats and Vmats, so any consumer
    that takes the whole list (ULVsparse.solve, ULVsparse.compute_ULV) sees a
    list on one device.  Use to()/cpu() to relocate the whole object; that is
    the only supported way to trade VRAM for host memory.

    Note on level ordering: the construction loop runs L-1 down to 0, so
    Dmats[0] is the leaf level and Dmats[-1] is the ROOT.  Earlier versions
    parked Dmats[-1] in pinned host memory under the name "leaf D"; that block
    is in fact the smallest in the list (one block of side ~2*rk), so the
    saving was negligible while the mixed-device list silently broke solve().

    """

    # every attribute holding a list of tensors; to() walks these.
    # NNvec is numpy and Nbvec is a list of ints, so both stay put.
    _GROUPS = {
        'core': ('Umats', 'Vmats', 'Dmats'),
        'ulv' : ('Qlist', 'Wlist', 'Rlist', 'Uulist'),
    }
    _tensor_lists = ('Umats', 'Vmats', 'Dmats',
                     'Qlist', 'Wlist', 'Rlist', 'Uulist')

    def __init__(self,A=None,device=None,tree=None,quad=False):
        # perm lives in a one-element box so that .T views (shallow __dict__
        # copies) see the same object when residency rebinds it.
        self._permbox = [None]
        self.Umats  =   []
        self.Vmats  =   []
        self.Dmats  =   []
        self.Qlist  =   []
        self.Rlist  =   []
        self.Wlist  =   []
        self.Uulist =   []
        torch.set_default_dtype(torch.float64)
        self.dtype = np.float64
        self.dtype_t = torch.float64

        self.mode   =   'N'
        self._tree  =   None

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
            self.A      =   A
            self.shape  =   self.A.shape
            self.dtype  = A.dtype

        if tree is not None:
            self.tree   =   tree
            self.perm   =   tree.perm_leaf
            self.Nb = tree.nleaves
            self.nl = len(self.perm)//self.Nb
            self.L = tree.nlevels
            self.shape = (len(self.perm),len(self.perm))
            
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
        self.tSample = 0
        self.tConstruct = 0
        self.Nbvec = []
        self.quad = quad
        self.tULV = 0
        self.tCompress = 0
        if quad:
            self.fac = 4
        else:
            self.fac = 2
    def set_Nbvec(self,Nbvec):
        self.Nbvec = Nbvec    
        self.Nb = Nbvec[0]
        self.nl = self.A.shape[0]//self.Nb
        self.L = len(Nbvec)
        self.perm   =   torch.arange(self.A.shape[0],dtype=torch.int64)
    def set_mats(self,Umats,Dmats,Vmats,Nbvec,fac=4):
        self.Umats = Umats
        self.Dmats = Dmats
        self.Vmats = Vmats
        self._host['core'] = None
        self._resident['core'] = Dmats[0].device if torch.is_tensor(Dmats[0]) else None
        self.perm = torch.arange(Dmats[0].shape[0])
        self.fac = fac
        self.Nb = Nbvec[0]
        self.shape = torch.tensor([Dmats[0].shape[0],Dmats[0].shape[0]],dtype = torch.int64)
        self.dtype = Dmats[0][0].dtype
    @property
    def nbytes(self):
        ctr = 0
        ctr+=sum([U.nbytes for U in self.Umats])
        ctr+=sum([V.nbytes for V in self.Vmats])
        ctr+=sum([D.nbytes for D in self.Dmats])
        return ctr
    @property
    def device(self):
        return str(self.compute_device)
    def _as_local(self,X):
        dev = self.compute_device
        if torch.is_tensor(X):
            return X.to(device=dev,dtype=self.dtype_t,non_blocking=True)
        return torch.from_numpy(X).to(device=dev,dtype=self.dtype_t)
    def construct(self,rk,Om=None,Psi=None,Y=None,Z=None,compute_ULV=False,fast=False):
        if Om is None:
            if self.A is None:
                raise ValueError("Samples and LinOP cannot both be None")
            else:
                # hard requirement max(fac*rk, nl) + rk at the worst level,
                # plus oversampling p = rk there (every other level has more)
                s = self.fac*max(rk, self.nl) + rk + 10

                Om = np.random.standard_normal(size = (self.A.shape[1],s))
                Psi= np.random.standard_normal(size = (self.A.shape[0],s))
                Y = self.A@Om
                Z = self.A.T@Psi

        if compute_ULV:
            self.constructHBS_ULV(rk,Om,Psi,Y,Z,fast=fast)
        else: 
            self.constructHBS(rk,Om,Psi,Y,Z,fast=fast)

    def constructHBS(self,rk,Om0,Psi0,Y0,Z0,fast=False):
        s = Om0.shape[1]
        Nb = self.Nb
        self.Nbvec = [Nb]
        nl = self.nl
        self.nSamples = s
        self._reset_residency()
        tic = time.time()
        if torch.is_tensor(self.perm):
            self.perm = self.perm.to(self.compute_device)
        else:
            self.perm = torch.as_tensor(self.perm, dtype=torch.int64,
                                        device=self.compute_device)        
        Ompr  = self._as_local(Om0 )[self.perm, :]
        Psipr = self._as_local(Psi0)[self.perm, :]
        Ypr   = self._as_local(Y0  )[self.perm, :]
        Zpr   = self._as_local(Z0  )[self.perm, :]

        Y = ULVsparse.convert_to_torch_tens(Ypr,self.Nb,device=self.device)
        Z = ULVsparse.convert_to_torch_tens(Zpr,self.Nb,device=self.device)
        Om  = ULVsparse.convert_to_torch_tens(Ompr, self.Nb, device=self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psipr,self.Nb, device=self.device)


        #Om  = to_block_tensor(Ompr,  Nb, nl)
        #Psi = to_block_tensor(Psipr, Nb, nl)
        #Y   = to_block_tensor(Ypr,   Nb, nl)
        #Z   = to_block_tensor(Zpr,   Nb, nl)
        
        
        nl = self.nl
        self.setupTime+=time.time()-tic
        self.tSample+=time.time()-tic
        tic_compress = time.time()
        for lvl in range(self.L-1,-1,-1):
            
            if lvl == self.L-1:
                Om_ell      = Om
                Psi_ell     = Psi
                Y_ell       = Y
                Z_ell       = Z
                rkm = min(rk,nl)
            else:
                Y_ell       -=block_mult(D_ell,Om_ell,self.device)
                Y_ell       = block_mult_and_reduce(U_ell,Y_ell,self.fac,self.device,mode='T')

                Z_ell       -= block_mult(D_ell,Psi_ell,self.device,mode='T')
                Z_ell       = block_mult_and_reduce(V_ell,Z_ell,self.fac,self.device,mode='T')
                
                Om_ell      = block_mult_and_reduce(V_ell,Om_ell,self.fac,self.device,mode='T')
                Psi_ell     = block_mult_and_reduce(U_ell,Psi_ell,self.fac,self.device,mode='T')
                
                Nb = Nb//self.fac
                rkm = min(rk,nl*((self.fac)**(self.L-1-lvl)))
            
            if lvl>0:
                
                tic = time.time()
                
                (U_ell,om_R),(V_ell,psi_R) = compute_UV_pair(Om_ell,Y_ell,Psi_ell,Z_ell,rkm,self.device,fast=fast)
                
                self.nullTime+=time.time()-tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,om_R,psi_R,fast=fast)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device,fast=fast)
                self.blockSolveTime+=time.time()-tic
                # stored on self.device like every other level; use to()/cpu()
                # to relocate the whole object rather than one block of it
                self.Dmats+=[D_ell]
            self.Nbvec+=[Nb]
        self.tCompress = time.time()-tic_compress
    def constructHBS_ULV(self,rk,Om0,Psi0,Y0,Z0,fast=True):
        s = Om0.shape[1]
        Nb = self.Nb
        self.Nbvec = [Nb]
        nl = self.nl
        self.nSamples = s
        self._reset_residency()
        tic = time.time()
        if torch.is_tensor(self.perm):
            self.perm = self.perm.to(self.compute_device)
        else:
            self.perm = torch.as_tensor(self.perm, dtype=torch.int64,
                                        device=self.compute_device)
        Ompr  = self._as_local(Om0 )[self.perm, :]
        Psipr = self._as_local(Psi0)[self.perm, :]
        Ypr   = self._as_local(Y0  )[self.perm, :]
        Zpr   = self._as_local(Z0  )[self.perm, :]

        Y = ULVsparse.convert_to_torch_tens(Ypr,self.Nb,device=self.device)
        Z = ULVsparse.convert_to_torch_tens(Zpr,self.Nb,device=self.device)
        Om  = ULVsparse.convert_to_torch_tens(Ompr, self.Nb, device=self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psipr,self.Nb, device=self.device)
        
        Nb = self.Nb
        nl = self.nl
        self.setupTime+=time.time()-tic
        self.tSample+=time.time()-tic
        self.NNvec = np.zeros(shape=(0,),dtype=np.int64)
        self.NNvec = np.append(self.NNvec,0)
        tic_compress = time.time()
        for lvl in range(self.L-1,-1,-1):
            
            if lvl == self.L-1:
                Om_ell      = Om
                Psi_ell     = Psi
                Y_ell       = Y
                Z_ell       = Z
                rkm = min(rk,nl)
            else:
                
                Y_ell       -=block_mult(D_ell,Om_ell,self.device)
                Y_ell       = block_mult_and_reduce(U_ell,Y_ell,self.fac,self.device,mode='T')

                Z_ell       -= block_mult(D_ell,Psi_ell,self.device,mode='T')
                Z_ell       = block_mult_and_reduce(V_ell,Z_ell,self.fac,self.device,mode='T')
                
                Om_ell      = block_mult_and_reduce(V_ell,Om_ell,self.fac,self.device,mode='T')
                Psi_ell     = block_mult_and_reduce(U_ell,Psi_ell,self.fac,self.device,mode='T')

                
                Nb = Nb//self.fac
                rkm = min(rk,nl*(self.fac**(self.L-1-lvl)))
            #print("lvl//Nb = ",lvl,"//",Nb)
            self.Nbvec+=[Nb]
            if lvl>0:
                tic = time.time()
                (U_ell, om_R), (V_ell, psi_R) = compute_UV_pair(
                    Om_ell, Y_ell, Psi_ell, Z_ell, rkm, self.device, fast=fast)
                self.nullTime += time.time() - tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,om_R,psi_R,fast=fast)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device,fast=fast)
                self.blockSolveTime+=time.time()-tic
                # stored on self.device like every other level; use to()/cpu()
                # to relocate the whole object rather than one block of it
                self.Dmats+=[D_ell]
                U_ell = torch.eye(D_ell.shape[1], dtype=D_ell.dtype,device=self.device)[None, :, :]
            
            tic = time.time()
            if lvl==self.L-1:
                    Rhat = D_ell
            else:
                Rhat = ULVsparse.sparse_block_mult_tens(Uhat,D_ell,device=self.device)
                Rhat = ULVsparse.block_diag_add_tens(Rhat,R_22,device=self.device)
            
            Q,W,Ru,R_22,NN = ULVsparse.compute_QRW_sparse(Rhat,V_ell,self.Nbvec[-1],device=self.device)
            self.Qlist+=[Q]
            if W is not None:
                # W = [W1 | V_ell] and the V_ell half is Vmats[-1] verbatim.
                # Store only the complement; ULVsparse.solve reads V from Vmats.
                W = W[:, :, :W.shape[2] - V_ell.shape[2]].contiguous()
            self.Wlist+=[W]
            self.Rlist+=[Ru]
            self.NNvec=np.append(self.NNvec,self.NNvec[-1]+NN)

            if lvl == self.L-1:
                Uhat = U_ell
            else:
                Uhat = ULVsparse.sparse_block_mult_tens(Uhat,U_ell,device=self.device)
            
            Uu = ULVsparse.sparse_block_mult_tens(Q[:,:,:-rkm],Uhat,device=self.device,mode='T')
            Ud = ULVsparse.sparse_block_mult_tens(Q[:,:,-rkm:],Uhat,device=self.device,mode='T')
            self.Uulist+=[Uu]
            Uhat=Ud
            self.tULV +=time.time()-tic
        if self.compute_device.type == 'cuda':
            torch.cuda.synchronize()
        self.tCompress = time.time()-tic_compress

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

    def matvec(self,v):
        return self._apply(v, transpose=False)
    def rmatvec(self,v):
        return self._apply(v, transpose=True)

    def matmat(self,v,chunk=None):
        return self._apply(v, transpose=False, chunk=chunk)

    def rmatmat(self,v,chunk=None):
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

        V = self._as_local(v)                      # tensor, compute_device, dtype_t
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
        (restriction) sweep and in the mode used on D, so they share one body;
        the previous copy-pasted pair is how rmatmat ended up calling matmat.
        """
        dmode = 'T' if transpose else 'N'
        down  = self.Umats if transpose else self.Vmats
        up    = self.Vmats if transpose else self.Umats

        VV = [V[self.perm, :]]
        for M in down:
            VV.append(block_matvec(M, VV[-1], self.device, mode='T'))
        u = block_matvec(self.Dmats[-1], VV[-1], self.device, mode=dmode)
        for lvl in range(len(up) - 1, -1, -1):
            u = block_matvec(up[lvl], u, self.device) \
              + block_matvec(self.Dmats[lvl], VV[lvl], self.device, mode=dmode)
        out = torch.zeros_like(u)
        out[self.perm, :] = u
        return out

    def __matmul__(self, v):
        if self.mode == 'N':
            return self.matmat(v)
        elif self.mode == 'T':
            return self.rmatmat(v)
        raise ValueError("mode not recognized")

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
    # Device placement
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
        # A group with no tensors (e.g. ulv on a compute_ULV=False block)
        # is vacuously resident -- there is nothing to stage.
        if not any(len(getattr(self, n)) for n in self._GROUPS[g]):
            return True
        r = self._resident[g]
        return r is not None and _dev_eq(r, self.compute_device)

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

        compute_device is NOT touched.  An evicted block that is applied
        again must be re-staged, never silently demoted to host arithmetic;
        that is the whole point of this method existing instead of to('cpu').
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
        # tree-supplied numpy perm becomes a tensor on the target rather
        # than staying host-side and forcing an implicit H2D on every
        # fancy-index in matmat/constructHBS.
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

    def compute_ULV(self):
        self._require_resident('compute_ULV')
        tic=time.time()
        self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec = ULVsparse.compute_ULV(self.Umats,self.Dmats,self.Vmats,self.Nbvec)
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
        modes; only the triangular sweep in ULVsparse.solve differs.  (An
        earlier version scattered the rhs and gathered the result for 'T',
        which is wrong for any non-involutive permutation.)

        Memory
        ------
        The ULV sweeps hold roughly 7 rhs-sized tensors at their peak (the
        permuted rhs, the peeled-off parts per level, and the leaf-level
        W1*y, V*x and their sum).  For the 2s-column fused solves of the
        red-black factorization that is the device-memory peak, so columns
        are processed in chunks of `chunk` (default _SOLVE_CHUNK[0]; 0
        disables chunking).  The transient then scales with the chunk, not
        with b.shape[1].  Columns are independent, so the result is identical
        up to BLAS kernel selection.

        overwrite_b=True writes the solution into b itself (when b is a torch
        tensor already on compute_device with the factor dtype, or a numpy
        array aliased by a CPU compute device) and returns it, so no output
        buffer is allocated.  Safe because each chunk's columns are gathered
        (copied) before that chunk's solution is scattered back.  If b had
        to be converted anyway, the private copy is always reused as output.
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

        # ------------------------------------------------------------
        # Input handling
        # ------------------------------------------------------------
        input_is_numpy = isinstance(b, np.ndarray)
        input_is_torch = torch.is_tensor(b)
        if not (input_is_numpy or input_is_torch):
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

        # Output buffer: reuse b_torch when allowed (caller opted in) or when it
        # is our own copy anyway; otherwise allocate.
        if overwrite_b or not aliased:
            out = b_torch
        else:
            out = torch.empty((nrow, ncol), dtype=dtype_t, device=b_torch.device)

        c = _SOLVE_CHUNK[0] if chunk is None else chunk
        if not c or c > ncol:
            c = max(ncol, 1)

        # ------------------------------------------------------------
        # Solve:  u = P^T ( ULV^{-1} ( P b ) ),  one column chunk at a time
        # ------------------------------------------------------------
        for j0 in range(0, ncol, c):
            j1 = min(j0 + c, ncol)
            bperm = b_torch[self.perm, j0:j1]      # advanced-index gather: a copy
            uhat = ULVsparse.solve(
                self.Umats,
                self.Dmats,
                self.Qlist,
                self.Wlist,
                self.Uulist,
                self.Rlist,
                self.NNvec,
                bperm,
                device=self.compute_device,
                mode=mode,
                Vmats=self.Vmats,
            )
            del bperm                              # free before the scatter
            out[self.perm, j0:j1] = uhat           # columns j0:j1 already read
            del uhat

        # ------------------------------------------------------------
        # Restore shape and array type
        # ------------------------------------------------------------
        u = out[:, 0] if was_vector else out
        if input_is_numpy:
            return u.detach().cpu().numpy()
        return u
