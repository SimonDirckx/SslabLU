"""
HBStorch_strong.py
==================

Strong-admissibility HBS compression via *randomized sketching only*.

This is a standalone companion to HBStorch.py.  It does NOT import or modify
that module; it reuses none of its state.  It implements the pieces we worked
out together:

  * row/column cluster bases U, V in dense *tensor* form  (Nb, b, rk)
  * the near-field error term E in *block-sparse tensor* form
        E_ij = A_ij - U_i U_i^*  A_ij  V_j V_j^*           (near-field pairs only)
    recovered purely from the sketches Y = A Omega, Z = A^* Psi
    (no far-field coupling B_ij is ever needed).

The two design ideas that avoid the k_max memory blow-up:

  1.  Null-space via a *shared per-block thin QR* + an *assembled b x b Gram*.
      The s-wide near-field stack  W_tau = [U_blk(j) : j in NF(tau)]  is never
      materialised; it only ever appears inside  W^*W  (assembled from cached
      b x b cross-Grams) and inside  Y_tau W  (collapsed to b x m on the fly).
      The only s-carrying object is the single, un-replicated table {U_blk}.

  2.  The far-field annihilates under the complement projectors, so each of the
      two terms of E is recovered from one sketch by a batched right-solve
      against the near-field Gaussian stack -- again with the s axis contracted
      away through the same assembled Gram.

------------------------------------------------------------------------------
ADMISSIBILITY PRECONDITION (important for your own tests)
------------------------------------------------------------------------------
The recovery identity is exact *iff* the operator is genuinely admissible at
the chosen rank, i.e. for every block-row i the far-field tiles share a common
rank-rk LEFT factor, and for every block-col j they share a common rank-rk
RIGHT factor:

        A_ij = U_i^true  S_ij  (V_j^true)^*     for all far (i, j).

Equivalently  (I - U_i U_i^*) A_ij = 0  and  A_ij (I - V_j V_j^*) = 0  for far
pairs.  A matrix with independent low-rank tiles does NOT satisfy this and the
row basis cannot span the far field -- it is not an HBS-admissible operator.
Build your validation operator accordingly (see `make_admissible_operator`).

------------------------------------------------------------------------------
ADJACENCY API (matches slabTree.level_adj_list)
------------------------------------------------------------------------------
`adj_sets` for a level is a list of length n (= #blocks at the level); entry r
is an iterable of the *level-local ranks* adjacent to block r.  This is exactly
`tree.level_adj_list[lvl]`.  Self-inclusion is NOT present (verified), so the
near-field of tau is built here as  {tau} U adj_sets[tau].

Block ordering: rank r corresponds to the r-th block in the (Nb, b, s) sketch
tensors, consistent with slabTree's sorted-global ordering and
ULVsparse.convert_to_torch_tens.
"""

from __future__ import annotations
import torch
import numpy as np
import time
from collections import defaultdict
from contextlib import contextmanager


def _cpu_rss_bytes():
    """Resident host memory of this process, in bytes (best-effort)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        try:
            import resource, sys
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is KB on Linux, bytes on macOS
            return ru * (1024 if sys.platform != 'darwin' else 1)
        except Exception:
            return 0


class ConstructProfiler:
    """
    Per-phase profiler for HBSMAT.construct.

    Two granularities:
      * top-level PHASES (input_load, basis_UV@lvl, recover_E@lvl, reduction@lvl,
        offload@lvl) -- recorded chronologically, each with an isolated GPU peak
        (reset_peak_memory_stats at the phase boundary).
      * DETAIL subroutines inside a phase (the QR / SVD / Gram pieces of the
        basis computation) -- these fire once per chunk per level, so they are
        AGGREGATED by name (summed time, summed GPU net delta, max resident,
        call count).  Detail blocks do NOT reset the peak counter, so they don't
        corrupt the enclosing phase's peak measurement.

    Each phase is kind='compute' or kind='transfer' (data movement).  Wall clock
    with a CUDA sync at boundaries.  Nothing prints automatically; call
    HBSMAT.print_profile().
    """

    def __init__(self, device):
        self.device = device
        self.on_cuda = str(device).startswith('cuda')
        self.records = []            # chronological top-level phases
        self.detail = {}             # name -> aggregated subroutine stats
        self.total_compute = 0.0
        self.total_transfer = 0.0
        self.enabled = True
        self.detail_enabled = True

    @contextmanager
    def phase(self, name, kind='compute'):
        """Top-level phase: isolated GPU peak via reset, recorded chronologically."""
        if not self.enabled:
            yield
            return
        if self.on_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            gpu_before = torch.cuda.memory_allocated(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.on_cuda:
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            if self.on_cuda:
                gpu_peak = torch.cuda.max_memory_allocated(self.device)
                gpu_after = torch.cuda.memory_allocated(self.device)
            else:
                gpu_peak = gpu_after = gpu_before = 0
            self.records.append(dict(
                name=name, kind=kind, time=dt, gpu_peak=gpu_peak,
                gpu_delta=(gpu_after - gpu_before) if self.on_cuda else 0,
                cpu_rss=_cpu_rss_bytes()))
            if kind == 'transfer':
                self.total_transfer += dt
            else:
                self.total_compute += dt

    @contextmanager
    def sub(self, name, parent):
        """
        Detail subroutine inside a phase.  Aggregated by name across all calls
        (chunks, U and V).  Does NOT reset the peak counter (so the enclosing
        phase's GPU peak stays valid); reports net GPU delta + resident-after.
        """
        if not (self.enabled and self.detail_enabled):
            yield
            return
        cur0 = torch.cuda.memory_allocated(self.device) if self.on_cuda else 0
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.on_cuda:
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            cur1 = torch.cuda.memory_allocated(self.device) if self.on_cuda else 0
            cpu = _cpu_rss_bytes()
            d = self.detail.get(name)
            if d is None:
                d = dict(name=name, parent=parent, calls=0, time=0.0,
                         gpu_delta=0, gpu_after=0, cpu_after=0)
                self.detail[name] = d
            d['calls']     += 1
            d['time']      += dt
            d['gpu_delta'] += (cur1 - cur0)
            d['gpu_after']  = max(d['gpu_after'], cur1)
            d['cpu_after']  = max(d['cpu_after'], cpu)

    def reset(self):
        self.records.clear()
        self.detail.clear()
        self.total_compute = 0.0
        self.total_transfer = 0.0


# =============================================================================
#  Near-field bookkeeping: turn ragged adjacency into grouped, padded gathers
# =============================================================================

class NearField:
    """
    Precomputed near-field structure for one level.

    Builds, from a level adjacency (list of sets of ranks, no self-inclusion):
      * self.nf[r]      : sorted list of near-field ranks of block r, INCLUDING r
      * grouping by near-field cardinality k = |nf(r)| so each group is a
        uniform batched problem
      * for each group: a (G, k) long index tensor of block ranks, with the
        target block's position within its own near-field recorded

    The grouping key is simply k (the number of near-field blocks); because the
    leaf/block size b is uniform, equal k means equal stacked-row count m = k*b,
    which is all the batched linear algebra needs.
    """

    def __init__(self, adj_sets, device=None):
        self.device = device
        n = len(adj_sets)
        self.n = n
        # full near-field incl. self, as sorted lists
        self.nf = [sorted(set(adj_sets[r]) | {r}) for r in range(n)]
        # self position within own near-field (index of r in nf[r])
        self.self_pos = [self.nf[r].index(r) for r in range(n)]

        # group block ranks by near-field cardinality k
        from collections import defaultdict
        groups = defaultdict(list)            # k -> list of block ranks r
        for r in range(n):
            groups[len(self.nf[r])].append(r)

        self.groups = []                      # list of dicts, one per group
        for k, ranks in sorted(groups.items()):
            ranks_t = torch.tensor(ranks, dtype=torch.long, device=device)
            # (G, k) matrix of near-field ranks
            nf_idx = torch.tensor([self.nf[r] for r in ranks],
                                   dtype=torch.long, device=device)   # (G, k)
            self_pos = torch.tensor([self.self_pos[r] for r in ranks],
                                    dtype=torch.long, device=device)  # (G,)
            self.groups.append(dict(k=k, ranks=ranks_t, nf_idx=nf_idx,
                                    self_pos=self_pos, G=len(ranks)))


# =============================================================================
#  Shared per-block thin QR  (the only s-carrying, un-replicated table)
# =============================================================================

def _block_thinQR(S_blk):
    """
    S_blk : (Nb, b, s)   one Gaussian sketch block per cluster (e.g. Omega).
    Returns Uq : (Nb, s, b) with orthonormal columns spanning each block's row
    space  (i.e. column space of S_blk[i]^T).  Stored once; shared by every
    near-field that contains the block.  No replication.
    """
    # reduced QR of S_blk^T : (Nb, s, b) -> Q (Nb, s, b)
    return torch.linalg.qr(S_blk.transpose(-2, -1), mode='reduced').Q


def _cross_gram(Uq, nf_idx):
    """
    Assemble, for every cluster in a group, the m x m Gram  M = W^* W  where
    W = [Uq[j] : j in near-field], WITHOUT forming W (no s-wide stack).

    Uq      : (Nb, s, b)
    nf_idx  : (G, k) long   near-field block ranks per cluster in the group
    returns : M (G, k*b, k*b)
    """
    G, k = nf_idx.shape
    b = Uq.shape[-1]
    # gather the k near-field U-blocks for each cluster: (G, k, s, b)
    Wb = Uq[nf_idx]                                  # advanced index -> (G, k, s, b)
    # M[g] has (a,c) tile = Uq[ja]^* Uq[jc]  (b x b)
    # compute all tiles via batched matmul over (G, k, k)
    # reshape to do one big bmm: (G, k, b, s) x (G, 1, s, k*b)? Simpler: einsum.
    # Wb: (G,k,s,b)  ->  tiles[g,a,c] = Wb[g,a].T @ Wb[g,c]  (b,b)
    M = torch.einsum('gasp,gcsq->gacpq', Wb, Wb)     # (G, k, k, b, b)
    M = M.permute(0, 1, 3, 2, 4).reshape(G, k * b, k * b).contiguous()
    return M, Wb


def _gram_range(Yi, Wb, M, rk, eps=1e-12):
    """
    Far-field range of  Y_i (I - W M^{-1} W^*)  from its b x b Gram, via eigh.

    Yi : (G, b, s)        target block-row of the sketch (Y for U, Z for V)
    Wb : (G, k, s, b)     near-field U-blocks (from _cross_gram)
    M  : (G, k*b, k*b)    assembled W^*W
    Returns basis (G, b, rk) : leading eigenvectors of the projected Gram.

    G_tau = Yi Yi^* - (Yi W) M^{-1} (Yi W)^* ,   all contractions land in b-dim.
    """
    G, k, s, b = Wb.shape
    # Yi W : (G, b, k*b).  Yi (G,b,s) times each Wb[:,a] (G,s,b) -> (G,b,b), concat.
    # batched: einsum over k
    YW = torch.einsum('gbs,gksp->gbkp', Yi, Wb).reshape(G, b, k * b)   # (G, b, k*b)
    # solve M X = YW^*  ->  X = M^{-1} (YW)^*  : (G, k*b, b)
    I = torch.eye(k*b, device=M.device, dtype=M.dtype)
    L = torch.linalg.cholesky(M + eps * I)
    Minv_YWt = torch.cholesky_solve(
    YW.transpose(-2, -1),
    L
    )
    #Minv_YWt = torch.linalg.lstsq(M, YW.transpose(-2, -1),rcond = 1e-12).solution
    Ggram = torch.bmm(Yi, Yi.transpose(-2, -1)) - torch.bmm(YW, Minv_YWt)  # (G,b,b)
    # symmetrise for numerical safety
    Ggram = 0.5 * (Ggram + Ggram.transpose(-2, -1))
    w, V = torch.linalg.eigh(Ggram)               # ascending
    # take rk largest -> last rk columns, reorder to descending
    Vtop = V[..., -rk:].flip(-1)                  # (G, b, rk)
    return Vtop


def _nf_rowspace(Snf, rcond=1e-12):
    """
    [fast=False]  Orthonormal basis Q of the near-field ROW space via a
    rcond-truncated SVD of the stack -- conditioning kappa(Snf), not
    kappa(Snf)^2, and robust to rank deficiency (small directions dropped).

    Snf : (G, m, s)   near-field stack (m = k*b rows).
    returns Q : (G, s, m)  (columns beyond the numerical rank are zeroed).
    """
    _, S, Vh = torch.linalg.svd(Snf, full_matrices=False)   # Vh: (G, m, s)
    keep = (S > rcond * S[..., :1]).to(Snf.dtype).unsqueeze(-1)
    return (Vh * keep).transpose(-2, -1)                    # (G, s, m)


def _range_via_projection(Yi, Q, rk):
    """
    [fast=False]  Far-field range of Y_i (I - Q Q^*) via SVD of the projected
    sketch -- no Gram, no PSD difference, no cancellation.
    Yi:(G,b,s) ; Q:(G,s,m). Returns (G,b,rk) leading left singular vectors.
    """
    Yperp = Yi - torch.bmm(torch.bmm(Yi, Q), Q.transpose(-2, -1))
    return torch.linalg.svd(Yperp, full_matrices=False).U[..., :rk]


def _pinv_rhs_via_qr(rhs, Snf, rcond=1e-12):
    """
    [fast=False]  Batched right-solve X = rhs @ pinv(S_nf) via a rcond-truncated
    SVD of the near-field stack -- conditioning kappa(S_nf), robust to rank
    deficiency.  S_nf = U S V^* ; pinv = V S^+ U^* ; X = rhs V S^+ U^*.
    rhs:(G,b,s) ; Snf:(G,m,s).  returns X:(G,b,m).
    """
    U, S, Vh = torch.linalg.svd(Snf, full_matrices=False)   # U:(G,m,r) Vh:(G,r,s)
    sinv = torch.where(S > rcond * S[..., :1], 1.0/S, torch.zeros_like(S))
    rV  = torch.bmm(rhs, Vh.transpose(-2, -1))              # (G, b, r)
    return torch.bmm(rV * sinv.unsqueeze(-2), U.transpose(-2, -1))   # (G, b, m)


# =============================================================================
#  Public: bases U / V  in tensor form  (Nb, b, rk)
# =============================================================================

def compute_basis_strong(S_blk, Sketch_blk, nearfield: NearField, rk,
                         device=None, group_chunk=None, fast=True,
                         prof=None, tag=''):
    """
    Compute one cluster-basis tensor (rows: pass Omega & Y ; cols: pass Psi & Z).

    fast : True  -> Gram path (per-block QR + W^*W + eigh).  Low memory; squares
                    conditioning, so coarse far-field can be lost to cancellation
                    when weak.  Default (matches the existing behaviour).
           False -> stable path (truncated-SVD near-field projector + projected
                    SVD).  Forms the (chunk, m, s) stack (use group_chunk), but
                    conditioning is kappa(Snf), not kappa(Snf)^2.
    group_chunk: max clusters per batched call (bounds the s-carrying transient).
    prof, tag  : optional ConstructProfiler and a label suffix; if given, the
                 internal subroutines are timed/measured as aggregated 'detail'
                 entries under parent f'basis_UV{tag}'.
    returns    : basis (Nb, b, rk)
    """
    from contextlib import nullcontext
    parent = f'basis_UV{tag}'
    def sub(name):
        return prof.sub(f'{name}{tag}', parent) if prof is not None else nullcontext()

    Nb, b, s = S_blk.shape
    out = S_blk.new_zeros(Nb, b, rk)
    if fast:
        with sub('  block_thinQR'):
            Uq = _block_thinQR(S_blk)
    else:
        Uq = None
    for grp in nearfield.groups:
        k       = grp['k']
        ranks   = grp['ranks']
        nf_idx  = grp['nf_idx']
        rk_g    = min(rk, b)
        G       = ranks.shape[0]
        cs      = G if group_chunk is None else group_chunk
        for c0 in range(0, G, cs):
            r_c   = ranks[c0:c0+cs]
            nf_c  = nf_idx[c0:c0+cs]
            Yi    = Sketch_blk[r_c]                # (Gc, b, s)
            if fast:
                with sub('  cross_gram (W*W)'):
                    M, Wb = _cross_gram(Uq, nf_c)
                with sub('  gram_range (eigh)'):
                    basis = _gram_range(Yi, Wb, M, rk_g)
                del M, Wb
            else:
                with sub('  gather_Snf'):
                    Snf = S_blk[nf_c].reshape(nf_c.shape[0], k * b, s)
                with sub('  nf_rowspace (SVD)'):
                    Q = _nf_rowspace(Snf)
                with sub('  range_proj (SVD)'):
                    basis = _range_via_projection(Yi, Q, rk_g)
                del Snf, Q
            if rk_g < rk:
                basis = torch.nn.functional.pad(basis, (0, rk - rk_g))
            out[r_c] = basis
            del Yi, basis
    return out


# =============================================================================
#  Public: block-sparse error term E
# =============================================================================

class BlockSparseE:
    """
    Near-field error term in block-sparse (BSR-of-blocks) form.

    val : (nnz, b, b)   dense tiles
    row : (nnz,) long   block-row rank
    col : (nnz,) long   block-col rank
    n   : number of blocks at this level (for the BSR shape)
    b   : block size

    Tiles are stored for every ordered near-field pair (i, j) with j in NF(i),
    including the diagonal (i, i).
    """
    def __init__(self, val, row, col, n, b):
        self.val = val
        # keep coordinate tensors colocated with the values (GPU-safety)
        self.row = row.to(val.device)
        self.col = col.to(val.device)
        self.n, self.b = n, b
        self.mode='N'
        self.nnz_chunk = None     # bound the (nnz,b,s) matvec transient; None = whole

    def to(self, device):
        """Move tiles and coordinates to `device` (cheap if already there).
        Used to offload completed-level E to host during construction and to
        stream it back to the compute device in matvec."""
        v = object.__new__(self.__class__)
        v.__dict__ = self.__dict__.copy()
        v.val = self.val.to(device)
        v.row = self.row.to(device)
        v.col = self.col.to(device)
        return v

    @property
    def device(self):
        return self.val.device

    @property
    def nbytes(self):
        return (self.val.element_size()*self.val.nelement()
                + self.row.element_size()*self.row.nelement()
                + self.col.element_size()*self.col.nelement())

    @property
    def nnz(self):
        return self.val.shape[0]

    def to_dense(self):
        """Dense (n*b, n*b) reconstruction of the stored near-field tiles."""
        b = self.b
        M = self.val.new_zeros(self.n * b, self.n * b)
        for t in range(self.nnz):
            i = int(self.row[t]); j = int(self.col[t])
            M[i*b:(i+1)*b, j*b:(j+1)*b] = self.val[t]
        return M

    def to_sparse_bsr(self):
        """Convert to a torch sparse BSR tensor of shape (n*b, n*b)."""
        # sort by (row, col) into CSR-of-blocks layout
        order = torch.argsort(self.row * self.n + self.col)
        row_s = self.row[order]; col_s = self.col[order]; val_s = self.val[order]
        crow = torch.zeros(self.n + 1, dtype=torch.long, device=val_s.device)
        counts = torch.bincount(row_s, minlength=self.n)
        crow[1:] = torch.cumsum(counts, 0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.sparse_bsr_tensor(crow, col_s, val_s,
                                           size=(self.n * self.b, self.n * self.b))
    def __matmul__(self, T):
        """
        Multiply the block-sparse matrix by a tensor T of shape (n,b,s).

        The (nnz, b, s) gather/product transient (= k_max x a full sketch at the
        leaf) is built and accumulated in batches of `self.nnz_chunk` nonzeros,
        so the reduction's peak is bounded independent of the total nnz.
        Set nnz_chunk=None (default) to do it in one batch.

        Returns
        -------
        out : (n,b,s)
        """
        assert T.ndim == 3
        assert T.shape[0] == self.n
        assert T.shape[1] == self.b
        if self.mode not in ('N', 'T'):
            raise ValueError("mode not recognized")
        nnz = self.nnz
        src = self.col if self.mode == 'N' else self.row   # gather index
        dst = self.row if self.mode == 'N' else self.col   # scatter index
        out = torch.zeros((self.n, self.b, T.shape[2]),
                          dtype=T.dtype, device=T.device)
        nc = getattr(self, 'nnz_chunk', None)
        step = nnz if nc is None else nc
        for p0 in range(0, nnz, step):
            p1 = min(p0 + step, nnz)
            blocks = T[src[p0:p1]]                           # (chunk, b, s)
            if self.mode == 'N':
                contrib = torch.matmul(self.val[p0:p1], blocks)
            else:
                contrib = torch.matmul(self.val[p0:p1].transpose(-2, -1), blocks)
            out.index_add_(0, dst[p0:p1], contrib)
            del blocks, contrib
        return out

    def matvec(self, T, transpose=False):
        """Block-sparse apply with an explicit transpose flag.
        T : (n, b, cols) -> (n, b, cols).  Equivalent to (self.T @ T) when
        transpose else (self @ T); lets callers pass a runtime flag instead of
        building a transposed view."""
        return (self.T @ T) if transpose else (self @ T)

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

def _pinv_rhs_via_gram(rhs, Sg, M):
    """
    Batched min-norm right-solve  X (S_nf) = rhs   i.e.  X = rhs @ pinv(S_nf),
    via the m x m Gram (S_nf S_nf^*); the s-wide (G,m,s) stack is never formed.

        X = rhs @ S_nf^* (S_nf S_nf^*)^{-1}

    rhs : (G, b, s)
    Sg  : (G, k, b, s)   near-field blocks, gathered once (also used for M).
    M   : (G, k*b, k*b)  = S_nf S_nf^*   (from _assemble_raw_gram)
    returns X : (G, b, k*b)
    """
    G, k, b, s = Sg.shape
    # rhs @ S_nf^* , block a = rhs @ Sg[:,a]^* ; s summed inside einsum
    rhs_St = torch.einsum('gbs,gkcs->gbkc', rhs, Sg).reshape(G, b, k * b)
    X = (torch.linalg.lstsq(M, rhs_St.transpose(-2, -1),rcond = 1e-12).solution).transpose(-2, -1)
    return X                                          # (G, b, k*b)


def _pairwise_gram_table(S_blk):
    """
    Cache of all b x b cross-Grams  G[i,j] = S_blk[i] S_blk[j]^*  needed later,
    computed once.  Returns a (Nb, Nb, b, b) tensor.  This carries NO s-axis
    (the s dimension is contracted out) so it is cheap: Nb^2 * b^2, and in
    practice only the near-field-co-occurring (i,j) are ever read.

    For large Nb prefer `_assemble_raw_gram` which computes only the needed
    tiles per group; this dense table is a convenience for small/medium levels.
    """
    # G[i,j] = S_i S_j^* ; do it as one contraction (Nb,b,s)x(Nb,b,s)->(Nb,Nb,b,b)
    return torch.einsum('ibs,jcs->ijbc', S_blk, S_blk)


def _assemble_raw_gram(S_blk, nf_idx):
    """
    Assemble  M[g] = S_NF(g) S_NF(g)^*   (m x m, m=k*b) for every cluster in a
    group, from b x b raw cross-Gram tiles, WITHOUT forming the (G,m,s) stack.

    S_blk  : (Nb, b, s)
    nf_idx : (G, k) long
    returns M : (G, k*b, k*b)

    Tile (a,c) of M[g] is  S[nf_idx[g,a]] S[nf_idx[g,c]]^* , a b x b matrix.
    We gather the k near-field blocks (G,k,b,s) and contract pairwise; the s
    axis is summed inside the einsum, so the only s-carrying object is the
    gathered near-field blocks themselves -- the same (G,k,b,s) we already need
    for the right-hand side, not an additional m x s replicated stack.
    """
    G, k = nf_idx.shape
    b = S_blk.shape[1]
    Sg = S_blk[nf_idx]                                # (G, k, b, s)
    M = torch.einsum('gabs,gcds->gacbd', Sg, Sg)      # (G,k,k,b,b)
    M = M.permute(0, 1, 3, 2, 4).reshape(G, k * b, k * b).contiguous()
    return M, Sg


def recover_E_strong(U_ell, V_ell, Y_blk, Z_blk, Om_blk, Psi_blk,
                     nearfield: NearField, device=None, group_chunk=None,
                     tile_device=None, fast=True):
    """
    Recover the block-sparse near-field error term E from sketches only.

    For every near-field pair (i, j), j in NF(i):
        E_ij = (I - U_i U_i^*) A_ij  +  U_i U_i^* A_ij (I - V_j V_j^*)

    fast        : True  -> Gram normal-equations solve (squares conditioning).
                  False -> truncated-SVD right-solve (stable).
    group_chunk : max clusters per batched solve; bounds the (chunk,k,b,s)
                  transient.  None = whole group.
    tile_device : where recovered (b,b) tiles accumulate (default: sketch device).

    U_ell, V_ell : (Nb, b, rk) ; Y_blk, Z_blk, Om_blk, Psi_blk : (Nb, b, s)
    returns BlockSparseE
    """
    Nb, b, s = Y_blk.shape
    tdev = tile_device if tile_device is not None else Y_blk.device
    tile = {}   # (i,j) -> (b,b) running E tile, kept on tdev

    def chunks(G):
        cs = G if group_chunk is None else group_chunk
        for c0 in range(0, G, cs):
            yield c0, min(c0 + cs, G)

    def right_solve(rhs, S_blk_, nf_c, k):
        """X = rhs @ pinv(S_NF) for the chunk, via Gram (fast) or trunc-SVD."""
        Gc = nf_c.shape[0]
        if fast:
            M, Sg = _assemble_raw_gram(S_blk_, nf_c)         # (Gc,kb,kb),(Gc,k,b,s)
            return _pinv_rhs_via_gram(rhs, Sg, M)            # (Gc, b, k*b)
        Snf = S_blk_[nf_c].reshape(Gc, k * b, s)             # (Gc, m, s)
        return _pinv_rhs_via_qr(rhs, Snf)                    # (Gc, b, k*b)

    # --- term1: per cluster i, solve against its OWN near-field Omega stack ---
    for grp in nearfield.groups:
        k = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']; G = grp['G']
        for c0, c1 in chunks(G):
            r_c  = ranks[c0:c1]; nf_c = nf_idx[c0:c1]; Gc = r_c.shape[0]
            Yi   = Y_blk[r_c]; Ui = U_ell[r_c]
            Yperp = Yi - torch.bmm(Ui, torch.bmm(Ui.transpose(-2, -1), Yi))
            X1   = right_solve(Yperp, Om_blk, nf_c, k).reshape(Gc, b, k, b)
            X1c  = X1.to(tdev)
            for g in range(Gc):
                i = int(r_c[g])
                for a in range(k):
                    tile[(i, int(nf_c[g, a]))] = X1c[g, :, a, :].clone()
            del Yi, Ui, Yperp, X1, X1c

    # --- term2: per cluster j, solve against its OWN near-field Psi stack ------
    for grp in nearfield.groups:
        k = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']; G = grp['G']
        for c0, c1 in chunks(G):
            r_c  = ranks[c0:c1]; nf_c = nf_idx[c0:c1]; Gc = r_c.shape[0]
            Zj   = Z_blk[r_c]; Vj = V_ell[r_c]
            Zperp = Zj - torch.bmm(Vj, torch.bmm(Vj.transpose(-2, -1), Zj))
            X2   = right_solve(Zperp, Psi_blk, nf_c, k).reshape(Gc, b, k, b)
            for g in range(Gc):
                j = int(r_c[g])
                for a in range(k):
                    i = int(nf_c[g, a])
                    Aij_perp = X2[g, :, a, :].transpose(-2, -1)        # (b,b)
                    Ui = U_ell[i]
                    t2 = (Ui @ (Ui.transpose(-2, -1) @ Aij_perp)).to(tdev)
                    tile[(i, j)] = tile[(i, j)] + t2 if (i, j) in tile else t2
            del Zj, Vj, Zperp, X2

    # --- pack into block-sparse tensor (on tdev) -----------------------------
    pairs = sorted(tile.keys())
    val = torch.stack([tile[p] for p in pairs], 0)
    row = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=tdev)
    col = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=tdev)
    return BlockSparseE(val, row, col, nearfield.n, b)


# =============================================================================
#  Convenience: build an admissible test operator and a level driver
# =============================================================================

def make_admissible_operator(adj_sets, b, rk, device=None, seed=0):
    """
    Construct a dense (n*b, n*b) operator that is genuinely HBS-admissible at
    rank rk for the given (single-level) adjacency: dense near-field tiles, and
    far-field tiles  A_ij = U_i^true S_ij (V_j^true)^*  with shared rk bases.
    Returns A, and also the near-field set for reference.
    """
    g = torch.Generator(device=device or 'cpu'); g.manual_seed(seed)
    n = len(adj_sets)
    nf = [sorted(set(adj_sets[r]) | {r}) for r in range(n)]
    Ut = [torch.linalg.qr(torch.randn(b, rk, generator=g, device=device)).Q
          for _ in range(n)]
    Vt = [torch.linalg.qr(torch.randn(b, rk, generator=g, device=device)).Q
          for _ in range(n)]
    A = torch.zeros(n * b, n * b, device=device)
    for i in range(n):
        for j in range(n):
            if j in nf[i]:
                A[i*b:(i+1)*b, j*b:(j+1)*b] = torch.randn(b, b, generator=g, device=device)
            else:
                S = torch.randn(rk, rk, generator=g, device=device)
                A[i*b:(i+1)*b, j*b:(j+1)*b] = Ut[i] @ S @ Vt[j].transpose(-2, -1)
    return A, nf


def to_block_tensor(M, n, b):
    """(n*b, s) -> (n, b, s) block tensor (analogue of convert_to_torch_tens)."""
    s = M.shape[1]
    return M.reshape(n, b, s)


def compress_level_strong(Om_blk,Psi_blk,Y_blk,Z_blk, adj_sets, rk,device=None,
                          group_chunk=None, tile_device=None, fast=True):
    """
    One-level strong-admissibility compression.  Returns (U, V, E, nfo).

    fast        : Gram path (True, default) vs stable trunc-SVD path (False).
    group_chunk : bounds the (chunk,k,*,s) transient on the compute device.
    tile_device : where E's (b,b) tiles accumulate (default = data device).
    NearField indices are pinned to the SKETCH device so gathers don't mismatch.
    """
    n = len(adj_sets)
    nfo = NearField(adj_sets, device=Om_blk.device)
    U = compute_basis_strong(Om_blk,  Y_blk, nfo, rk, device=device,
                             group_chunk=group_chunk, fast=fast)
    V = compute_basis_strong(Psi_blk, Z_blk, nfo, rk, device=device,
                             group_chunk=group_chunk, fast=fast)
    E = recover_E_strong(U, V, Y_blk, Z_blk, Om_blk, Psi_blk, nfo, device=device,
                         group_chunk=group_chunk, tile_device=tile_device, fast=fast)
    return U, V, E, nfo

def has_farfield(adj_sets):
    """True iff at least one block still has a non-empty far-field at this level.
    Far-field of block r is everything outside NF(r) = {r} U adj(r); if every
    block's near-field already covers all n blocks, there is nothing left to
    compress and we are at the base case."""
    n = len(adj_sets)
    return any(len(set(adj_sets[r]) | {r}) < n for r in range(n))
def compute_final_E(Y_blk, Om_blk, device=None):
    """Base case (no far-field left): the whole reduced operator is near-field.
    Recover the dense reduced operator B from  B Om = Y  (B = Y pinv(Om)) and
    pack it as an all-pairs BlockSparseE.  Exact when s >= n*b."""
    n, b, s = Y_blk.shape
    Yf = Y_blk.reshape(n * b, s)
    Of = Om_blk.reshape(n * b, s)
    B  = Yf @ torch.linalg.pinv(Of)                      # (n*b, n*b)
    idx = [(i, j) for i in range(n) for j in range(n)]
    val = torch.stack([B[i*b:(i+1)*b, j*b:(j+1)*b] for i, j in idx], 0)
    row = torch.tensor([i for i, _ in idx], dtype=torch.long, device=device)
    col = torch.tensor([j for _, j in idx], dtype=torch.long, device=device)
    return BlockSparseE(val, row, col, n, b)

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

    """

    def __init__(self,A=None,device=None,tree=None,quad=False):
        self.Umats  =   []
        self.Vmats  =   []
        self.Emats  =   []
        
        self.mode   =   'N'
        self._tree  =   None
        torch.set_default_dtype(torch.float64)
        self.dtype = torch.float64
        if A is not None:
            self.A      =   A
            self.shape  =   self.A.shape
            self.dtype  = A.dtype
        self.device = device
        # where completed-level U/V/E are parked between construct and matvec.
        # 'cpu' keeps device memory free during construction (the leaf level is
        # the binding allocation); set to self.device to keep everything resident.
        self.store_device = 'cpu'
        # clusters processed per batched solve in the basis / E recovery; bounds
        # the (chunk,k,*,s) transient on the compute device.  None = whole level.
        self.group_chunk = None

        if tree is not None:
            self.tree   =   tree
            self.perm   =   tree.perm_leaf
            self.Nb = tree.nleaves
            self.shape = (len(self.perm),len(self.perm))
            self.nl = self.shape[0]//self.Nb
            self.L = tree.nlevels
            
            
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
        self.tSample = 0
        self.tConstruct = 0
        self.Nbvec = []
        self.quad = quad
        self.profiler = None     # populated by construct(); see print_profile()
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
    
    def set_mats(self,Umats,Emats,Vmats,Nbvec,fac=4):
        self.Umats = Umats
        self.Emats = Emats
        self.Vmats = Vmats
        self.perm = torch.arange(Emats[0].shape[0])
        self.fac = fac
        self.Nb = Nbvec[0]
        self.shape = torch.tensor([Emats[0].shape[0],Emats[0].shape[0]],dtype = torch.int64)
        self.dtype = Emats[0][0].dtype
    @property
    def nbytes(self):
        ctr = 0
        ctr+=sum([U.nbytes for U in self.Umats])
        ctr+=sum([V.nbytes for V in self.Vmats])
        ctr+=sum([D.nbytes for D in self.Dmats])
        return ctr
    
    def construct(self,rk,Om,Psi,Y,Z,group_chunk=None,store_device=None,
                  empty_cache=True,fast=True,profile=True,profile_detail=True):
        #assume numpy input here:
        if group_chunk is not None: self.group_chunk = group_chunk
        if store_device is not None: self.store_device = store_device
        self.profiler = ConstructProfiler(self.device)
        self.profiler.enabled = profile
        self.profiler.detail_enabled = profile_detail
        prof = self.profiler
        t_all = time.perf_counter()
        Nb = self.Nb
        nl = Om.shape[0]//Nb
        # single host->device move per array, in the target dtype, then permute
        # with a device-resident index (avoids a CPU-index-into-GPU sync).
        with prof.phase('input_load (H2D)', kind='transfer'):
            perm = torch.as_tensor(self.perm, dtype=torch.long, device=self.device)
            dt = self.dtype
            Ompr  = torch.from_numpy(Om ).to(device=self.device, dtype=dt)[perm, :]
            Psipr = torch.from_numpy(Psi).to(device=self.device, dtype=dt)[perm, :]
            Ypr   = torch.from_numpy(Y  ).to(device=self.device, dtype=dt)[perm, :]
            Zpr   = torch.from_numpy(Z  ).to(device=self.device, dtype=dt)[perm, :]
            Om_blk  = to_block_tensor(Ompr,  Nb, nl)
            Psi_blk = to_block_tensor(Psipr, Nb, nl)
            Y_blk   = to_block_tensor(Ypr,   Nb, nl)
            Z_blk   = to_block_tensor(Zpr,   Nb, nl)
            del Ompr, Psipr, Ypr, Zpr
        self.constructHBS(rk,Om_blk,Psi_blk,Y_blk,Z_blk,empty_cache=empty_cache,fast=fast)
        self.tConstruct = time.perf_counter() - t_all

    def constructHBS(self,rk,Om_blk,Psi_blk,Y_blk,Z_blk,empty_cache=True,fast=True):
        Nb = self.Nb
        nl = self.nl
        s = Om_blk.shape[2]
        rkm = rkm = min(rk,nl)
        Nbvec = [Nb]
        sdev = self.store_device
        prof = self.profiler
        gc = self.group_chunk
        on_cuda = (str(self.device).startswith('cuda'))
        def _free():
            if empty_cache and on_cuda:
                torch.cuda.empty_cache()
        for lvl in range(self.L-1, -1, -1):
            adj_level = self.tree.level_adj_list[lvl]

            if has_farfield(adj_level):
                # NearField indices on the sketch device
                nfo = NearField(adj_level, device=Om_blk.device)
                # --- basis U, V (compute), with subroutine detail ---
                with prof.phase(f'basis_UV@lvl{lvl}', kind='compute'):
                    U = compute_basis_strong(Om_blk, Y_blk, nfo, rkm,
                                             device=self.device, group_chunk=gc, fast=fast,
                                             prof=prof, tag=f'@lvl{lvl}')
                    V = compute_basis_strong(Psi_blk, Z_blk, nfo, rkm,
                                             device=self.device, group_chunk=gc, fast=fast,
                                             prof=prof, tag=f'@lvl{lvl}')
                # --- error term E (compute), tiles kept on compute device ---
                with prof.phase(f'recover_E@lvl{lvl}', kind='compute'):
                    E = recover_E_strong(U, V, Y_blk, Z_blk, Om_blk, Psi_blk, nfo,
                                         device=self.device, group_chunk=gc,
                                         tile_device=self.device, fast=fast)
                # --- reduction to next coarser level (compute) ---
                with prof.phase(f'reduction@lvl{lvl}', kind='compute'):
                    Nb_prev = Nb
                    Nb = Nb // self.fac
                    rkm_used = U.shape[-1]
                    # bound the BlockSparseE matvec's (nnz,b,s) transient to about
                    # group_chunk block-rows' worth of tiles (avg degree nnz/Nb_prev).
                    if gc is not None:
                        avg_deg = max(1, E.nnz // max(1, Nb_prev))
                        E.nnz_chunk = max(1, gc * avg_deg)
                    # process Y (uses E, Om_blk, U) then free what each step kills.
                    Y_blk = Y_blk - E @ Om_blk
                    Y_blk = torch.bmm(U.transpose(-2, -1), Y_blk).reshape(Nb, self.fac*rkm_used, s)
                    Z_blk = Z_blk - E.T @ Psi_blk
                    Z_blk = torch.bmm(V.transpose(-2, -1), Z_blk).reshape(Nb, self.fac*rkm_used, s)
                    Om_new  = torch.bmm(V.transpose(-2, -1), Om_blk).reshape(Nb, self.fac*rkm_used, s)
                    del Om_blk; Om_blk = Om_new; del Om_new
                    Psi_new = torch.bmm(U.transpose(-2, -1), Psi_blk).reshape(Nb, self.fac*rkm_used, s)
                    del Psi_blk; Psi_blk = Psi_new; del Psi_new
                # --- offload completed-level operators to host (transfer) ---
                with prof.phase(f'offload@lvl{lvl} (D2H)', kind='transfer'):
                    self.Umats += [U.to(sdev)]
                    self.Vmats += [V.to(sdev)]
                    self.Emats += [E.to(sdev)]
                self.Nbvec += [Nb]
                del U, V, E
                _free()
                rkm = rk
            else:
                with prof.phase(f'final_E@lvl{lvl}', kind='compute'):
                    E = compute_final_E(Y_blk, Om_blk, device=self.device)
                with prof.phase(f'offload@lvl{lvl} (D2H)', kind='transfer'):
                    self.Emats += [E.to(sdev)]
                del E
                _free()
                break

    def print_profile(self, sort_by=None):
        """
        Print per-phase construction stats gathered by the profiler.  Call this
        manually after construct(); nothing prints on its own.

        Columns: phase | kind | time(s) | GPU peak | GPU delta | CPU RSS.
        `sort_by` in {None,'time','gpu_peak'}; None keeps chronological order.
        """
        prof = getattr(self, 'profiler', None)
        if prof is None or not prof.records:
            print("[HBSMAT] no profile available — run construct(..., profile=True) first.")
            return

        def fmt_bytes(n):
            if n is None: return '   -   '
            for unit in ('B','KB','MB','GB','TB'):
                if abs(n) < 1024 or unit == 'TB':
                    return f"{n:7.1f}{unit}"
                n /= 1024

        recs = list(prof.records)
        if sort_by == 'time':
            recs = sorted(recs, key=lambda r: -r['time'])
        elif sort_by == 'gpu_peak':
            recs = sorted(recs, key=lambda r: -r['gpu_peak'])

        on_cuda = prof.on_cuda
        W = 26
        print("=" * 92)
        print(f"HBSMAT.construct profile   (device={self.device}, "
              f"store={self.store_device}, group_chunk={self.group_chunk})")
        print("-" * 92)
        print(f"{'phase':<{W}}{'kind':<10}{'time(s)':>10}"
              f"{'GPU peak':>13}{'GPU delta':>13}{'CPU RSS':>13}")
        print("-" * 92)
        for r in recs:
            print(f"{r['name']:<{W}}{r['kind']:<10}{r['time']:>10.4f}"
                  f"{fmt_bytes(r['gpu_peak']) if on_cuda else '   -   ':>13}"
                  f"{fmt_bytes(r['gpu_delta']) if on_cuda else '   -   ':>13}"
                  f"{fmt_bytes(r['cpu_rss']):>13}")
        print("-" * 92)
        gpu_peak_overall = max((r['gpu_peak'] for r in recs), default=0)
        cpu_peak_overall = max((r['cpu_rss'] for r in recs), default=0)
        print(f"{'TOTAL compute':<{W}}{'':<10}{prof.total_compute:>10.4f}")
        print(f"{'TOTAL transfer':<{W}}{'':<10}{prof.total_transfer:>10.4f}")
        print(f"{'TOTAL wall (construct)':<{W}}{'':<10}{getattr(self,'tConstruct',0.0):>10.4f}")
        if on_cuda:
            print(f"{'peak GPU (any phase)':<{W}}{'':<10}{'':>10}{fmt_bytes(gpu_peak_overall):>13}")
        print(f"{'peak CPU RSS':<{W}}{'':<10}{'':>10}{'':>13}{'':>13}{fmt_bytes(cpu_peak_overall):>13}")
        print("=" * 92)

        # ---- subroutine detail (aggregated across chunks; U and V combined) ---
        if prof.detail:
            WD = 30
            print("\nBASIS SUBROUTINE DETAIL  (aggregated over chunks; U+V combined)")
            print("-" * 92)
            print(f"{'subroutine':<{WD}}{'calls':>7}{'time(s)':>10}"
                  f"{'GPU net':>13}{'GPU after':>13}{'CPU after':>13}")
            print("-" * 92)
            # group by parent phase, preserve first-seen order
            parents = []
            for d in prof.detail.values():
                if d['parent'] not in parents:
                    parents.append(d['parent'])
            for p in parents:
                print(f"[{p}]")
                items = [d for d in prof.detail.values() if d['parent'] == p]
                if sort_by == 'time':
                    items = sorted(items, key=lambda d: -d['time'])
                for d in items:
                    print(f"{d['name']:<{WD}}{d['calls']:>7}{d['time']:>10.4f}"
                          f"{fmt_bytes(d['gpu_delta']) if on_cuda else '   -   ':>13}"
                          f"{fmt_bytes(d['gpu_after']) if on_cuda else '   -   ':>13}"
                          f"{fmt_bytes(d['cpu_after']):>13}")
            print("-" * 92)
            print("GPU net = net allocated change over the subroutine (summed over calls);")
            print("GPU after / CPU after = max resident observed at the subroutine's exit.")
            print("=" * 92)
    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

    def matvec(self,v):
        return self.matmat(v)
    def rmatvec(self,v):
        return self.rmatmat(v)

    # ------------------------------------------------------------------
    #  Shared block-form sweep (mirrors weak matmat: down -> base -> up).
    #
    #  Strong-admissibility telescoping (validated):  A = U B V^* + E,
    #  with B recursively the next-level operator and E the near-field
    #  residual.  Everything is kept in BLOCK form (Nb, b, cols) because
    #  Emats are BlockSparseE and Umats/Vmats are dense (Nb, b, rk).
    #
    #     down:  coarse_in  = V^*(or U^*) in ,  regrouped by fac
    #     base:  apply dense reduced operator  Emats[-1]  (all-pairs)
    #     up:    out = U(or V) coarse_out  +  E(or E^*) in_at_this_level
    #
    #  Invariant from construct:  len(Emats) == len(Umats) + 1, the last
    #  Emats entry being the dense base.  mode='T' swaps U<->V and applies
    #  the transposed tiles.
    # ------------------------------------------------------------------
    def _sweep(self, v, mode='N'):
        nlev = len(self.Umats)
        assert len(self.Emats) == nlev + 1, (
            "matvec needs a dense base: expected len(Emats)==len(Umats)+1, got "
            f"{len(self.Emats)} vs {nlev}. The construct loop must terminate via "
            "the has_farfield base case (compute_final_E), not by running out of "
            "levels.")

        numpy_input = isinstance(v, np.ndarray)
        if numpy_input:
            v = torch.from_numpy(v)
        cdev = self.device                       # compute device
        ref_dtype = self.Emats[-1].val.dtype
        v = v.to(device=cdev, dtype=ref_dtype)   # single move (device + dtype)
        perm = torch.as_tensor(self.perm, dtype=torch.long, device=cdev)
        col_vec = (v.ndim == 1)
        vperm = v[perm, None] if col_vec else v[perm, :]
        cols = vperm.shape[1]

        downmats = self.Vmats if mode == 'N' else self.Umats
        upmats   = self.Umats if mode == 'N' else self.Vmats
        transpose = (mode == 'T')

        # ---- down-sweep: project + regroup fac children into a parent block.
        # Mats are streamed from store_device to the compute device per level and
        # released immediately, so matvec peak stays bounded just like construct.
        VV = [vperm.reshape(self.Nb, self.nl, cols)]
        for lvl in range(nlev):
            A = downmats[lvl].to(cdev)                         # (Nb_lvl, b_lvl, rk)
            proj = torch.bmm(A.transpose(-2, -1), VV[lvl])     # (Nb_lvl, rk, cols)
            Nb_lvl, rk_lvl = A.shape[0], A.shape[2]
            VV.append(proj.reshape(Nb_lvl // self.fac, self.fac * rk_lvl, cols))
            del A

        # ---- dense base ----
        uperm = self.Emats[-1].to(cdev).matvec(VV[-1], transpose=transpose)

        # ---- up-sweep: split parent back to children, U(coarse) + E(this) ----
        for lvl in range(nlev - 1, -1, -1):
            U = upmats[lvl].to(cdev)                           # (Nb_lvl, b_lvl, rk)
            Nb_lvl, rk_lvl = U.shape[0], U.shape[2]
            uperm = uperm.reshape(Nb_lvl, rk_lvl, cols)
            far  = torch.bmm(U, uperm)                          # (Nb_lvl, b_lvl, cols)
            near = self.Emats[lvl].to(cdev).matvec(VV[lvl], transpose=transpose)
            uperm = far + near
            del U, far, near

        # ---- flatten, un-permute, restore input shape ----
        uperm = uperm.reshape(-1, cols)
        u = torch.zeros_like(uperm)
        u[perm, :] = uperm
        if col_vec:
            u = u.flatten()
        if numpy_input:
            u = u.cpu().numpy()
        return u

    def matmat(self, v):
        return self._sweep(v, mode='N')

    def rmatmat(self, v):
        return self._sweep(v, mode='T')

    def __matmul__(self, v):
        return self._sweep(v, mode=self.mode)

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, t):
        self._tree = t