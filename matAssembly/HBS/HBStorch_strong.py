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
#  Stable near-field algebra  (truncated SVD; conditioning kappa, not kappa^2)
# =============================================================================
#
#  The Gram framework (per-block thin QR + assembled W^*W + eigh of a PSD
#  difference + normal-equations right-solve) was removed: it squares the
#  conditioning and loses the weak coarse far-field to cancellation.  Every
#  near-field operation now goes through ONE truncated SVD of the near-field
#  Gaussian stack  S_nf (G, m, s),  m = k*b:
#
#      S_nf = U_s diag(S) Vh        (full_matrices=False ; U_s:(G,m,r), Vh:(G,r,s))
#
#  from which we read off, with no extra factorisation,
#      * the orthonormal near-field ROW-space basis  Q = (Vh truncated)^T   and
#      * the pseudo-inverse right-solve              X = rhs @ pinv(S_nf).
#
#  basis-U and E-term1 both act against the Omega near-field stack, and basis-V
#  and E-term2 both act against the Psi near-field stack, so the SVD is computed
#  once per (level, group, chunk) and reused via NFStackCache.
# =============================================================================


class NFStackCache:
    """Per-(group, chunk) cache of the near-field-stack factorization (a tagged
    tuple: ('qr', Q, R) or ('svd', U, S, Vh)).  Keyed by (group_index,
    chunk_start).  The basis pass fills it; recover_E_strong reads it (and
    recomputes on a miss, so the two stay usable independently)."""
    def __init__(self):
        self.store = {}
    def get(self, key):
        return self.store.get(key)
    def put(self, key, fac):
        self.store[key] = fac


def _factor_nf(Snf, rcond=1e-12, stats=None):
    """Factor the near-field stack S_nf (G, m, s), m = k*b, for both the
    row-space projector and the pseudo-inverse right-solve.

    Fast path -- QR of S_nf^T:  S_nf^T = Q R,  Q:(G,s,m) orthonormal columns
    spanning row(S_nf), R:(G,m,m) upper-triangular.  Valid when m <= s and the
    stack is full row rank (generic for Gaussian sketches).  This computes no
    singular values (the projector only needs an orthonormal row-space basis,
    the right-solve only needs R) and uses geqrf/orgqr + trsm, which are far
    faster than batched gesvd -- especially on GPU.

    Safe fallback -- truncated SVD:  taken when m > s, or when min|diag(R)| is a
    tiny fraction of max|diag(R)| (near rank-deficient), so R can't be inverted
    safely.  Conditioning is kappa(S_nf) on both paths (never squared).

    Returns a tagged tuple consumed by _rowspace / _pinv_rhs.  `stats`, if given,
    counts how often each path is taken (for the construct profiler)."""
    G, m, s = Snf.shape
    if m <= s:
        Q, R = torch.linalg.qr(Snf.transpose(-2, -1), mode='reduced')   # (G,s,m),(G,m,m)
        d = torch.diagonal(R, dim1=-2, dim2=-1).abs()                    # (G, m)
        if bool((d.amin(dim=-1) > rcond * d.amax(dim=-1).clamp_min(0)).all()):
            if stats is not None: stats['qr'] = stats.get('qr', 0) + 1
            return ('qr', Q, R)
    U, S, Vh = torch.linalg.svd(Snf, full_matrices=False)
    if stats is not None: stats['svd'] = stats.get('svd', 0) + 1
    return ('svd', U, S, Vh)


def _rowspace(fac, rcond=1e-12):
    """Orthonormal near-field ROW-space basis Q:(G,s,m) from a factorization."""
    if fac[0] == 'qr':
        return fac[1]
    return _rowspace_from_svd(fac[2], fac[3], rcond)


def _rowspace_from_svd(S, Vh, rcond=1e-12):
    """Row-space basis from a (fallback) SVD; near-rank columns zeroed."""
    keep = (S > rcond * S[..., :1]).to(Vh.dtype).unsqueeze(-1)
    return (Vh * keep).transpose(-2, -1)


def _pinv_rhs(rhs, fac, rcond=1e-12):
    """Batched right-solve X = rhs @ pinv(S_nf) from a factorization.
    rhs:(G,b,s) -> X:(G,b,m)."""
    if fac[0] == 'qr':
        _, Q, R = fac
        # pinv(S_nf) = Q R^{-T} ; X = (rhs Q) R^{-T}  ->  X^T solves R X^T = (rhs Q)^T
        W = torch.bmm(rhs, Q)                                   # (G, b, m)
        XT = torch.linalg.solve_triangular(R, W.transpose(-2, -1), upper=True, left=True)
        return XT.transpose(-2, -1)
    _, U, S, Vh = fac
    sinv = torch.where(S > rcond * S[..., :1], 1.0 / S, torch.zeros_like(S))
    rV = torch.bmm(rhs, Vh.transpose(-2, -1))                   # (G, b, r)
    return torch.bmm(rV * sinv.unsqueeze(-2), U.transpose(-2, -1))


def _range_via_projection(Yi, Q, rk):
    """Far-field range of Y_i (I - Q Q^*): leading rk left singular vectors of
    the projected sketch Yperp (G,b,s), b <= s.

    Computed EXACTLY but without a fat-matrix SVD: QR of Yperp^T (s x b) gives
    Yperp = R^T Q_y^T with Q_y orthonormal columns, so the left singular vectors
    of Yperp equal those of the tiny (b x b) matrix R^T.  This is an exact SVD
    (not randomized) -- it just moves the only gesvd onto a b x b matrix, which
    avoids the cuSOLVER batched-gesvd convergence fallback that the (b x s) SVD
    triggers, and is far cheaper.
    Yi:(G,b,s) ; Q:(G,s,m). Returns (G,b,rk)."""
    Yperp = Yi - torch.bmm(torch.bmm(Yi, Q), Q.transpose(-2, -1))   # (G,b,s)
    _, R = torch.linalg.qr(Yperp.transpose(-2, -1), mode='reduced')  # R:(G,b,b)
    Ur = torch.linalg.svd(R.transpose(-2, -1), full_matrices=False).U  # (G,b,b)
    return Ur[..., :rk]


def _basis_from_factor(Yi, fac, rk_g):
    """Far-field basis (G,b,rk_g) for target sketch Yi from a near-field-stack
    factorization: project out the near-field row space, then range via SVD."""
    return _range_via_projection(Yi, _rowspace(fac), rk_g)


# =============================================================================
#  Public: bases U / V  in tensor form  (Nb, b, rk)
# =============================================================================

def compute_basis_strong(S_blk, Sketch_blk, nearfield: NearField, rk,
                         device=None, group_chunk=None, cache: 'NFStackCache' = None,
                         prof=None, tag='', stats=None):
    """
    Compute one cluster-basis tensor (rows: pass Omega & Y ; cols: pass Psi & Z).

    For each group/chunk we form the near-field stack S_nf (Gc, k*b, s), factor
    it once (QR fast path / SVD fallback, see _factor_nf), build the row-space
    projector, and read off the far-field range of the projected sketch.

    The (chunk, k*b, s) stack is the only s-carrying transient; bound it with
    group_chunk on large levels.

    cache : optional NFStackCache.  If given, the per-chunk factorization of S_nf
            is stored under (group_index, chunk_start) so recover_E_strong can
            reuse it (basis-U / E-term1 share Omega; basis-V / E-term2 share Psi).
    returns : basis (Nb, b, rk)
    """
    from contextlib import nullcontext
    parent = f'basis_UV{tag}'
    def sub(name):
        return prof.sub(f'{name}{tag}', parent) if prof is not None else nullcontext()

    Nb, b, s = S_blk.shape
    out = S_blk.new_zeros(Nb, b, rk)
    for gi, grp in enumerate(nearfield.groups):
        k = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']
        rk_g = min(rk, b); G = ranks.shape[0]
        cs = G if group_chunk is None else group_chunk
        for c0 in range(0, G, cs):
            r_c = ranks[c0:c0+cs]; nf_c = nf_idx[c0:c0+cs]
            Yi  = Sketch_blk[r_c]                  # (Gc, b, s)
            with sub('  gather_Snf'):
                Snf = S_blk[nf_c].reshape(nf_c.shape[0], k * b, s)
            with sub('  factor_nf (QR/SVD)'):
                fac = _factor_nf(Snf, stats=stats)
            if cache is not None:
                cache.put((gi, c0), fac)
            with sub('  range (SVD)'):
                basis = _basis_from_factor(Yi, fac, rk_g)
            if rk_g < rk:
                basis = torch.nn.functional.pad(basis, (0, rk - rk_g))
            out[r_c] = basis
            del Yi, Snf, fac, basis
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

    # default near-field apply strategy; 'bucket' won the bench_matvec.py study
    # (batched dense GEMM per degree group, no atomic scatter).  'flat' is the
    # old gather + index_add; 'bsr' uses torch.sparse_bsr_tensor @ dense.
    apply_strategy = 'bucket'

    def to(self, device):
        """Move tiles and coordinates to `device` (cheap if already there).
        Resets any prepared apply cache so it rebuilds on the new device (matvec
        runs after the final device placement, so it is built exactly once)."""
        v = object.__new__(self.__class__)
        v.__dict__ = self.__dict__.copy()
        v.val = self.val.to(device)
        v.row = self.row.to(device)
        v.col = self.col.to(device)
        v._invalidate_prepared()
        return v

    def _invalidate_prepared(self):
        for a in ('_prepared', '_info_N', '_info_T', '_bsr_N', '_bsr_T'):
            if hasattr(self, a):
                delattr(self, a)

    @property
    def device(self):
        return self.val.device

    @property
    def shape(self):
        return (self.n * self.b, self.n * self.b)

    @property
    def dtype(self):
        return self.val.dtype

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

    def to_sparse_bsr(self, row=None, col=None, val=None):
        """Convert to a torch sparse BSR tensor of shape (n*b, n*b)."""
        row = self.row if row is None else row
        col = self.col if col is None else col
        val = self.val if val is None else val
        order = torch.argsort(row * self.n + col)
        row_s = row[order]; col_s = col[order]; val_s = val[order].contiguous()
        crow = torch.zeros(self.n + 1, dtype=torch.long, device=val_s.device)
        counts = torch.bincount(row_s, minlength=self.n)
        crow[1:] = torch.cumsum(counts, 0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.sparse_bsr_tensor(crow.to(torch.int32), col_s.to(torch.int32),
                                           val_s, size=(self.n * self.b, self.n * self.b))

    # ------------------------------------------------------------------
    #  Prepared apply structures (built once; the sparsity pattern is fixed
    #  across every GMRES iteration).
    # ------------------------------------------------------------------
    @staticmethod
    def _coord_buckets(self_coord, other_coord, nnz):
        """Group tile indices by `self_coord` node; within a bucket every node
        shares degree k.  Returns list of (nodes(G,), nbr(G,k), tile(G,k))."""
        dev = self_coord.device
        cs = self_coord.cpu().numpy(); co = other_coord.cpu().numpy()
        by = defaultdict(list)
        for t in range(nnz):
            by[int(cs[t])].append((int(co[t]), t))
        degs = defaultdict(list)
        for node, lst in by.items():
            degs[len(lst)].append(node)
        out = []
        for k, nodes in sorted(degs.items()):
            nodes = sorted(nodes)
            nbr  = [[o for o, _ in by[node]] for node in nodes]
            tile = [[t for _, t in by[node]] for node in nodes]
            out.append((torch.tensor(nodes, dtype=torch.long, device=dev),
                        torch.tensor(nbr,  dtype=torch.long, device=dev),
                        torch.tensor(tile, dtype=torch.long, device=dev)))
        return out

    def _prepare(self):
        """Degree-bucket structures.  Forward (N) packs tiles into bucket order
        in place (no extra tile memory, no per-call gather); transpose (T) keeps
        bucket indices and gathers/transposes per bucket at apply time (rare)."""
        if getattr(self, '_prepared', False):
            return
        # --- N: group by row, then reorder val so each bucket is contiguous ---
        bk_N = self._coord_buckets(self.row, self.col, self.nnz)
        perm = torch.cat([tile.reshape(-1) for _, _, tile in bk_N]) \
            if bk_N else torch.arange(self.nnz, device=self.val.device)
        self.val = self.val[perm].contiguous()
        self.row = self.row[perm].contiguous()
        self.col = self.col[perm].contiguous()
        info_N, off = [], 0
        for nodes, nbr, tile in bk_N:
            G, k = nbr.shape
            info_N.append((nodes, nbr.reshape(-1), off, G, k)); off += G * k
        self._info_N = info_N
        # --- T: group by col (on the reordered coords), gather tiles at apply ---
        bk_T = self._coord_buckets(self.col, self.row, self.nnz)
        self._info_T = [(nodes, nbr.reshape(-1), tile.reshape(-1), nbr.shape[0], nbr.shape[1])
                        for nodes, nbr, tile in bk_T]
        self._prepared = True

    def _apply_bucket(self, T, transpose):
        n, b, cols = T.shape
        out = torch.empty((n, b, cols), dtype=T.dtype, device=T.device)
        if transpose:
            for nodes, nbr, tile, G, k in self._info_T:
                tiles = self.val[tile].transpose(-2, -1)          # (G*k,b,b)
                xs = T[nbr]                                        # (G*k,b,cols)
                prod = torch.bmm(tiles, xs).reshape(G, k, b, cols)
                out[nodes] = prod.sum(dim=1)
        else:
            for nodes, nbr, off, G, k in self._info_N:
                xs = T[nbr]                                        # (G*k,b,cols)
                prod = torch.bmm(self.val[off:off + G * k], xs).reshape(G, k, b, cols)
                out[nodes] = prod.sum(dim=1)
        return out

    def _apply_flat(self, T, transpose):
        """Old gather + index_add path (fallback / reference)."""
        nnz = self.nnz
        src = self.col if not transpose else self.row
        dst = self.row if not transpose else self.col
        out = torch.zeros((self.n, self.b, T.shape[2]), dtype=T.dtype, device=T.device)
        nc = getattr(self, 'nnz_chunk', None)
        step = nnz if nc is None else nc
        for p0 in range(0, nnz, step):
            p1 = min(p0 + step, nnz)
            blocks = T[src[p0:p1]]
            v = self.val[p0:p1] if not transpose else self.val[p0:p1].transpose(-2, -1)
            out.index_add_(0, dst[p0:p1], torch.matmul(v, blocks))
        return out

    def _apply_bsr(self, T, transpose):
        n, b, cols = T.shape
        if transpose and not hasattr(self, '_bsr_T'):
            self._bsr_T = self.to_sparse_bsr(self.col, self.row,
                                             self.val.transpose(-2, -1).contiguous())
        if (not transpose) and not hasattr(self, '_bsr_N'):
            self._bsr_N = self.to_sparse_bsr()
        A = self._bsr_T if transpose else self._bsr_N
        return (A @ T.reshape(n * b, cols)).reshape(n, b, cols)

    def matvec(self, T, transpose=False):
        """Block-sparse apply with an explicit transpose flag.
        T : (n, b, cols) -> (n, b, cols)."""
        assert T.ndim == 3 and T.shape[0] == self.n and T.shape[1] == self.b
        strat = self.apply_strategy
        if strat == 'flat':
            return self._apply_flat(T, transpose)
        if strat == 'bsr':
            return self._apply_bsr(T, transpose)
        self._prepare()
        return self._apply_bucket(T, transpose)

    def __matmul__(self, T):
        if self.mode not in ('N', 'T'):
            raise ValueError("mode not recognized")
        return self.matvec(T, transpose=(self.mode == 'T'))

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

def recover_E_strong(U_ell, V_ell, Y_blk, Z_blk, Om_blk, Psi_blk,
                     nearfield: NearField, device=None, group_chunk=None,
                     tile_device=None, cache_Om: 'NFStackCache' = None,
                     cache_Psi: 'NFStackCache' = None):
    """
    Recover the block-sparse near-field error term E from sketches only.

    For every near-field pair (i, j), j in NF(i):
        E_ij = (I - U_i U_i^*) A_ij  +  U_i U_i^* A_ij (I - V_j V_j^*)

    Stable truncated-SVD right-solve only (the Gram normal-equations path was
    removed).  X = rhs @ pinv(S_NF) is read off the SVD of the near-field stack.

    cache_Om / cache_Psi : optional NFStackCache filled by compute_basis_strong.
        term1 reuses the Omega-stack SVD (shared with basis-U); term2 reuses the
        Psi-stack SVD (shared with basis-V).  On a miss the SVD is recomputed.

    group_chunk : max clusters per batched solve; bounds the (chunk, k*b, s)
                  transient.  Must match the value used for compute_basis_strong
                  for the cache keys to line up.
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

    def right_solve(rhs, S_blk_, nf_c, k, cache, gi, c0):
        """X = rhs @ pinv(S_NF), reusing the cached factorization of S_NF (QR
        fast path / SVD fallback) when available; recompute on a miss."""
        Gc = nf_c.shape[0]
        fac = cache.get((gi, c0)) if cache is not None else None
        if fac is None:
            Snf = S_blk_[nf_c].reshape(Gc, k * b, s)
            fac = _factor_nf(Snf)
        return _pinv_rhs(rhs, fac)                           # (Gc, b, k*b)

    # --- term1: per cluster i, solve against its OWN near-field Omega stack ---
    for gi, grp in enumerate(nearfield.groups):
        k = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']; G = grp['G']
        for c0, c1 in chunks(G):
            r_c  = ranks[c0:c1]; nf_c = nf_idx[c0:c1]; Gc = r_c.shape[0]
            Yi   = Y_blk[r_c]; Ui = U_ell[r_c]
            Yperp = Yi - torch.bmm(Ui, torch.bmm(Ui.transpose(-2, -1), Yi))
            X1   = right_solve(Yperp, Om_blk, nf_c, k, cache_Om, gi, c0).reshape(Gc, b, k, b)
            X1c  = X1.to(tdev)
            for g in range(Gc):
                i = int(r_c[g])
                for a in range(k):
                    tile[(i, int(nf_c[g, a]))] = X1c[g, :, a, :].clone()
            del Yi, Ui, Yperp, X1, X1c

    # --- term2: per cluster j, solve against its OWN near-field Psi stack ------
    for gi, grp in enumerate(nearfield.groups):
        k = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']; G = grp['G']
        for c0, c1 in chunks(G):
            r_c  = ranks[c0:c1]; nf_c = nf_idx[c0:c1]; Gc = r_c.shape[0]
            Zj   = Z_blk[r_c]; Vj = V_ell[r_c]
            Zperp = Zj - torch.bmm(Vj, torch.bmm(Vj.transpose(-2, -1), Zj))
            X2   = right_solve(Zperp, Psi_blk, nf_c, k, cache_Psi, gi, c0).reshape(Gc, b, k, b)
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


def compute_bases_and_E_strong(Om_blk, Psi_blk, Y_blk, Z_blk, nearfield: NearField,
                               rk, device=None, group_chunk=None, tile_device=None,
                               prof=None, tag='', stats=None):
    """
    Fused single-factorization construction of the bases U, V AND the near-field
    error term E -- mathematically identical to
    compute_bases_strong(...) followed by recover_E_strong(...), but each
    near-field stack is factored exactly ONCE and reused for both its basis and
    its E term, instead of being factored for the bases and then recomputed in
    recover_E.  That halves the (dominant) factorization count.

    For every near-field pair (i, j), j in NF(i):
        E_ij = (I - U_i U_i^*) A_ij  +  U_i U_i^* A_ij (I - V_j V_j^*)
               '------------ term1 ------------'  '----------- term2 -----------'

      Pass 1 (Omega stack):  factor -> U[chunk]  and  term1 tiles
      Pass 2 (Psi   stack):  factor -> V[chunk]  and  term2 tiles

    term2 of column-cluster j needs U_i for every i in NF(j); those may live in
    other chunks, which is safe because ALL of U is produced in pass 1 before
    pass 2 starts.  Only the current chunk's factorization is ever resident, so
    peak memory is bounded by group_chunk exactly as in the two-pass design --
    this is a pure compute win (~2x fewer factorizations), memory-neutral.

    group_chunk : max clusters per batched factorization/solve (bounds the
                  (chunk, k*b, s) transient).
    tile_device : where recovered (b,b) tiles accumulate (default: data device).
    returns (U, V, E)
    """
    from contextlib import nullcontext
    parent = f'bases_E{tag}'
    def sub(name):
        return prof.sub(f'{name}{tag}', parent) if prof is not None else nullcontext()

    Nb, b, s = Om_blk.shape
    tdev = tile_device if tile_device is not None else Om_blk.device
    U = Om_blk.new_zeros(Nb, b, rk)
    V = Om_blk.new_zeros(Nb, b, rk)
    tile = {}                                   # (i,j) -> (b,b) running E tile

    def chunks(G):
        cs = G if group_chunk is None else group_chunk
        for c0 in range(0, G, cs):
            yield c0, min(c0 + cs, G)

    # ---- Pass 1: Omega factor -> basis U  +  term1 = (I - U_iU_i*) A_ij --------
    for gi, grp in enumerate(nearfield.groups):
        k = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']; G = grp['G']
        rk_g = min(rk, b)
        for c0, c1 in chunks(G):
            r_c = ranks[c0:c1]; nf_c = nf_idx[c0:c1]; Gc = r_c.shape[0]
            with sub('  gather_Snf'):
                Snf = Om_blk[nf_c].reshape(Gc, k * b, s)
            with sub('  factor_nf (QR/SVD)'):
                fac = _factor_nf(Snf, stats=stats)
            del Snf
            Yi = Y_blk[r_c]
            with sub('  range (QR+svd)'):
                Ublk = _basis_from_factor(Yi, fac, rk_g)
            if rk_g < rk:
                Ublk = torch.nn.functional.pad(Ublk, (0, rk - rk_g))
            U[r_c] = Ublk
            with sub('  term1 (pinv+assembly)'):
                Yperp = Yi - torch.bmm(Ublk, torch.bmm(Ublk.transpose(-2, -1), Yi))
                X1 = _pinv_rhs(Yperp, fac).reshape(Gc, b, k, b).to(tdev)
                for g in range(Gc):
                    i = int(r_c[g])
                    for a in range(k):
                        tile[(i, int(nf_c[g, a]))] = X1[g, :, a, :].clone()
            del fac, Yi, Ublk, Yperp, X1

    # ---- Pass 2: Psi factor -> basis V  +  term2 = U_iU_i* A_ij (I - V_jV_j*) --
    for gi, grp in enumerate(nearfield.groups):
        k = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']; G = grp['G']
        rk_g = min(rk, b)
        for c0, c1 in chunks(G):
            r_c = ranks[c0:c1]; nf_c = nf_idx[c0:c1]; Gc = r_c.shape[0]
            with sub('  gather_Snf'):
                Snf = Psi_blk[nf_c].reshape(Gc, k * b, s)
            with sub('  factor_nf (QR/SVD)'):
                fac = _factor_nf(Snf, stats=stats)
            del Snf
            Zj = Z_blk[r_c]
            with sub('  range (QR+svd)'):
                Vblk = _basis_from_factor(Zj, fac, rk_g)
            if rk_g < rk:
                Vblk = torch.nn.functional.pad(Vblk, (0, rk - rk_g))
            V[r_c] = Vblk
            with sub('  term2 (pinv+assembly)'):
                Zperp = Zj - torch.bmm(Vblk, torch.bmm(Vblk.transpose(-2, -1), Zj))
                X2 = _pinv_rhs(Zperp, fac).reshape(Gc, b, k, b)
                for g in range(Gc):
                    j = int(r_c[g])
                    for a in range(k):
                        i = int(nf_c[g, a])
                        Aij_perp = X2[g, :, a, :].transpose(-2, -1)        # (b,b)
                        Ui = U[i]
                        t2 = (Ui @ (Ui.transpose(-2, -1) @ Aij_perp)).to(tdev)
                        tile[(i, j)] = tile[(i, j)] + t2 if (i, j) in tile else t2
            del fac, Zj, Vblk, Zperp, X2

    # ---- assemble E (on tdev) ------------------------------------------------
    pairs = sorted(tile.keys())
    val = torch.stack([tile[p] for p in pairs], 0)
    row = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=tdev)
    col = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=tdev)
    return U, V, BlockSparseE(val, row, col, nearfield.n, b)


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
                          group_chunk=None, tile_device=None):
    """
    One-level strong-admissibility compression.  Returns (U, V, E, nfo).

    Stable truncated-SVD path only.  The Omega/Psi near-field-stack SVDs are
    computed during the basis step and reused in E recovery.
    group_chunk : bounds the (chunk,k*b,s) transient on the compute device.
    tile_device : where E's (b,b) tiles accumulate (default = data device).
    NearField indices are pinned to the SKETCH device so gathers don't mismatch.
    """
    n = len(adj_sets)
    nfo = NearField(adj_sets, device=Om_blk.device)
    U, V, E = compute_bases_and_E_strong(Om_blk, Psi_blk, Y_blk, Z_blk, nfo, rk,
                                         device=device, group_chunk=group_chunk,
                                         tile_device=tile_device)
    return U, V, E, nfo

def has_farfield(adj_sets):
    """True iff at least one block still has a non-empty far-field at this level.
    Far-field of block r is everything outside NF(r) = {r} U adj(r); if every
    block's near-field already covers all n blocks, there is nothing left to
    compress and we are at the base case."""
    n = len(adj_sets)
    return any(len(set(adj_sets[r]) | {r}) < n for r in range(n))
class DenseBase:
    """Dense coarse base operator B (n*b x n*b), applied as a single GEMM.

    The all-pairs base is dense, so applying it through the block-sparse
    gather/scatter machinery is wasteful (n^2 tiny tiles + scatter).  This
    wrapper keeps B dense and applies it as one (n*b)x(n*b) . (n*b)xcols matmul.
    Exposes the BlockSparseE-compatible surface used by HBSMAT._sweep
    (matvec / to / dtype / device / nbytes)."""
    def __init__(self, B, n, b):
        self.B = B; self.n = n; self.b = b
        self.mode = 'N'

    @property
    def dtype(self):
        return self.B.dtype

    @property
    def shape(self):
        return tuple(self.B.shape)

    @property
    def device(self):
        return self.B.device

    @property
    def nbytes(self):
        return self.B.element_size() * self.B.nelement()

    def to(self, device):
        v = object.__new__(self.__class__)
        v.__dict__ = self.__dict__.copy()
        v.B = self.B.to(device)
        return v

    def matvec(self, T, transpose=False):
        n, b, cols = T.shape
        x = T.reshape(n * b, cols)
        y = (self.B.transpose(-2, -1) @ x) if transpose else (self.B @ x)
        return y.reshape(n, b, cols)


def compute_final_E(Y_blk, Om_blk, device=None):
    """Base case (no far-field left): the whole reduced operator is near-field.
    Recover the dense reduced operator B from  B Om = Y  (B = Y pinv(Om)) and
    return it as a DenseBase (applied as one GEMM in matvec).  Exact when
    s >= n*b."""
    n, b, s = Y_blk.shape
    Yf = Y_blk.reshape(n * b, s)
    Of = Om_blk.reshape(n * b, s)
    B  = Yf @ torch.linalg.pinv(Of)                      # (n*b, n*b)
    return DenseBase(B, n, b)

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
        # after construction, migrate operators to the compute device once and
        # keep them resident (no per-matvec streaming).  Set False for low-VRAM
        # use, then matvec requires operators already on self.device.
        self.migrate_after_construct = True
        self._matvec_perm = None      # cached device-resident leaf permutation
        # counts of near-field factorizations taken on each path during construct
        self.factor_stats = {'qr': 0, 'svd': 0}

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
        self.dtype = Emats[0].dtype
    @property
    def nbytes(self):
        ctr = 0
        ctr+=sum([U.nbytes for U in self.Umats])
        ctr+=sum([V.nbytes for V in self.Vmats])
        ctr+=sum([E.nbytes for E in self.Emats])
        return ctr
    
    def construct(self,rk,Om,Psi,Y,Z,group_chunk=None,store_device=None,
                  empty_cache=True,fast=None,profile=True,profile_detail=True):
        # `fast` is accepted for backward compatibility but ignored: the Gram
        # "fast path" was removed (it squared the conditioning); the stable
        # truncated-SVD path is now the only path.
        #assume numpy input here:
        if group_chunk is not None: self.group_chunk = group_chunk
        if store_device is not None: self.store_device = store_device
        self.profiler = ConstructProfiler(self.device)
        self.profiler.enabled = profile
        self.profiler.detail_enabled = profile_detail
        self.factor_stats = {'qr': 0, 'svd': 0}
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
        self.constructHBS(rk,Om_blk,Psi_blk,Y_blk,Z_blk,empty_cache=empty_cache)
        self.tConstruct = time.perf_counter() - t_all

    def constructHBS(self,rk,Om_blk,Psi_blk,Y_blk,Z_blk,empty_cache=True):
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
                # --- bases U, V AND error term E in one fused pass: each
                #     near-field stack is factored exactly once (QR fast path)
                #     and reused for its basis and its E term.  Peak is bounded
                #     by group_chunk; ~2x fewer factorizations than basis-then-
                #     recover_E. ---
                with prof.phase(f'bases_E@lvl{lvl}', kind='compute'):
                    U, V, E = compute_bases_and_E_strong(
                        Om_blk, Psi_blk, Y_blk, Z_blk, nfo, rkm,
                        device=self.device, group_chunk=gc, tile_device=self.device,
                        prof=prof, tag=f'@lvl{lvl}', stats=self.factor_stats)
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
                    # Use the nnz-chunked flat apply here so the construction-time
                    # transient stays bounded (the bucket path is for the matvec).
                    Y_blk = Y_blk - E._apply_flat(Om_blk, False)
                    Y_blk = torch.bmm(U.transpose(-2, -1), Y_blk).reshape(Nb, self.fac*rkm_used, s)
                    Z_blk = Z_blk - E._apply_flat(Psi_blk, True)
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

        # Operators were parked on store_device (default 'cpu') to bound peak
        # device memory during construction.  Once construction is done that
        # pressure is gone, so migrate everything to the compute device ONCE and
        # leave it resident -- the matvec then streams nothing (the per-iteration
        # H2D of every U/V/E was the dominant strong-case matvec cost in GMRES).
        if self.migrate_after_construct and str(self.device).startswith('cuda'):
            self.to_device(self.device)

    def to_device(self, device=None):
        """Move all U/V/E operators to `device` (default self.device) once and
        keep them resident.  matvec asserts the operators live here, so no
        per-iteration host<->device transfer happens.  Returns self."""
        device = self.device if device is None else device
        self.Umats = [U.to(device) for U in self.Umats]
        self.Vmats = [V.to(device) for V in self.Vmats]
        self.Emats = [E.to(device) for E in self.Emats]
        self.store_device = device
        self._matvec_perm = None      # rebuilt on the new device at next matvec
        return self

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
        fs = getattr(self, 'factor_stats', None)
        if fs:
            tot = fs.get('qr', 0) + fs.get('svd', 0)
            print(f"{'near-field factor':<{W}}{'':<10}"
                  f"QR {fs.get('qr',0)} / SVD-fallback {fs.get('svd',0)}  (of {tot})")
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
        ref_dtype = self.Emats[-1].dtype         # DenseBase / BlockSparseE both expose .dtype
        v = v.to(device=cdev, dtype=ref_dtype)   # single move (device + dtype)

        # Operators are kept resident on the compute device after construction
        # (see to_device / migrate_after_construct), so the sweep streams nothing
        # and never re-wraps a BlockSparseE (which would drop its prepared apply
        # cache).  In low-VRAM mode (migration disabled) operators are streamed.
        resident = (str(self.store_device) == str(cdev))
        def dev(x):
            return x if resident else x.to(cdev)

        if self._matvec_perm is None or self._matvec_perm.device != torch.device(cdev):
            self._matvec_perm = torch.as_tensor(self.perm, dtype=torch.long, device=cdev)
        perm = self._matvec_perm
        col_vec = (v.ndim == 1)
        vperm = v[perm, None] if col_vec else v[perm, :]
        cols = vperm.shape[1]

        downmats = self.Vmats if mode == 'N' else self.Umats
        upmats   = self.Umats if mode == 'N' else self.Vmats
        transpose = (mode == 'T')

        # ---- down-sweep: project + regroup fac children into a parent block ----
        VV = [vperm.reshape(self.Nb, self.nl, cols)]
        for lvl in range(nlev):
            A = dev(downmats[lvl])                             # (Nb_lvl, b_lvl, rk)
            proj = torch.bmm(A.transpose(-2, -1), VV[lvl])     # (Nb_lvl, rk, cols)
            Nb_lvl, rk_lvl = A.shape[0], A.shape[2]
            VV.append(proj.reshape(Nb_lvl // self.fac, self.fac * rk_lvl, cols))

        # ---- dense base (single GEMM) ----
        uperm = dev(self.Emats[-1]).matvec(VV[-1], transpose=transpose)

        # ---- up-sweep: split parent back to children, U(coarse) + E(this) ----
        for lvl in range(nlev - 1, -1, -1):
            U = dev(upmats[lvl])                                # (Nb_lvl, b_lvl, rk)
            Nb_lvl, rk_lvl = U.shape[0], U.shape[2]
            uperm = uperm.reshape(Nb_lvl, rk_lvl, cols)
            far  = torch.bmm(U, uperm)                          # (Nb_lvl, b_lvl, cols)
            near = dev(self.Emats[lvl]).matvec(VV[lvl], transpose=transpose)
            uperm = far + near

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