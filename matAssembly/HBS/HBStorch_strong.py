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


def _gram_range(Yi, Wb, M, rk, eps=0.0):
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
    Minv_YWt = torch.linalg.lstsq(M, YW.transpose(-2, -1),rcond = 1e-12).solution
    Ggram = torch.bmm(Yi, Yi.transpose(-2, -1)) - torch.bmm(YW, Minv_YWt)  # (G,b,b)
    # symmetrise for numerical safety
    Ggram = 0.5 * (Ggram + Ggram.transpose(-2, -1))
    w, V = torch.linalg.eigh(Ggram)               # ascending
    # take rk largest -> last rk columns, reorder to descending
    Vtop = V[..., -rk:].flip(-1)                  # (G, b, rk)
    return Vtop


# =============================================================================
#  Public: bases U / V  in tensor form  (Nb, b, rk)
# =============================================================================

def compute_basis_strong(S_blk, Sketch_blk, nearfield: NearField, rk,
                         device=None, group_chunk=None):
    """
    Compute one cluster-basis tensor (rows: pass Omega & Y ; cols: pass Psi & Z).

    S_blk      : (Nb, b, s)   Gaussian blocks   (Omega for U, Psi for V)
    Sketch_blk : (Nb, b, s)   operator sketch   (Y     for U, Z   for V)
    nearfield  : NearField for this level
    rk         : target rank
    group_chunk: max clusters processed per batched call (bounds the s-carrying
                 transient Wb of shape (chunk,k,s,b)).  None = whole group.
    returns    : basis (Nb, b, rk), orthonormal columns per block
    """
    Nb, b, s = S_blk.shape
    Uq = _block_thinQR(S_blk)                     # (Nb, s, b)  shared table
    out = S_blk.new_zeros(Nb, b, rk)
    for grp in nearfield.groups:
        k       = grp['k']
        ranks   = grp['ranks']                    # (G,)
        nf_idx  = grp['nf_idx']                    # (G, k)
        rk_g    = min(rk, b)                        # never exceed block size
        G       = ranks.shape[0]
        cs      = G if group_chunk is None else group_chunk
        for c0 in range(0, G, cs):
            r_c   = ranks[c0:c0+cs]
            nf_c  = nf_idx[c0:c0+cs]
            M, Wb = _cross_gram(Uq, nf_c)          # (Gc, k*b, k*b), (Gc,k,s,b)
            Yi    = Sketch_blk[r_c]                # (Gc, b, s)
            basis = _gram_range(Yi, Wb, M, rk_g)   # (Gc, b, rk_g)
            if rk_g < rk:
                basis = torch.nn.functional.pad(basis, (0, rk - rk_g))
            out[r_c] = basis
            del M, Wb, Yi, basis
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
        self.val, self.row, self.col = val, row, col
        self.n, self.b = n, b
        self.mode='N'

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

        Returns
        -------
        out : (n,b,s)
        """
        assert T.ndim == 3
        assert T.shape[0] == self.n
        assert T.shape[1] == self.b
        if self.mode == 'N':
            blocks = T[self.col]                    # (nnz,b,s)
            contrib = torch.matmul(self.val, blocks)  # (nnz,b,s)

            out = torch.zeros(
                (self.n, self.b, T.shape[2]),
                dtype=T.dtype,
                device=T.device,
            )

            out.index_add_(0, self.row, contrib)
        elif self.mode=='T':
            blocks = T[self.row]                      # (nnz, b, s)

            # Multiply by transposed tiles
            contrib = torch.matmul(
                self.val.transpose(-2, -1),           # (nnz, b, b)
                blocks                                # (nnz, b, s)
            )                                         # (nnz, b, s)

            out = torch.zeros(
                (self.n, self.b, T.shape[2]),
                dtype=T.dtype,
                device=T.device,
            )

            # Accumulate into original column indices
            out.index_add_(0, self.col, contrib)
        else:
            raise ValueError("mode not recognized")


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
                     nearfield: NearField, device=None):
    """
    Recover the block-sparse near-field error term E from sketches only.

    For every near-field pair (i, j), j in NF(i):
        E_ij = (I - U_i U_i^*) A_ij  +  U_i U_i^* A_ij (I - V_j V_j^*)
             =  term1                +  term2
    term1 from Y:  [(I-U_iU_i^*) A_i,:]_NF(i) = (I-U_iU_i^*) Y_i  pinv(Om_NF(i))
    term2 from Z:  [(I-V_jV_j^*) A_:,j^*]_NF(j) = (I-V_jV_j^*) Z_j pinv(Psi_NF(j))
                   transpose tile, left-apply U_i U_i^*.

    U_ell, V_ell : (Nb, b, rk)
    Y_blk, Z_blk, Om_blk, Psi_blk : (Nb, b, s)
    returns BlockSparseE
    """
    Nb, b, s = Y_blk.shape

    # --- term1: per cluster i, solve against its OWN near-field Omega stack ---
    # accumulate tiles into a dict keyed by (i, j)
    tile = {}   # (i,j) -> (b,b) running E tile

    for grp in nearfield.groups:
        k      = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']
        G      = grp['G']
        # M_i = Om_NF Om_NF^* assembled from b x b tiles; Om_nf gathered once.
        M_i, Om_nf = _assemble_raw_gram(Om_blk, nf_idx)  # (G,k*b,k*b),(G,k,b,s)
        Yi     = Y_blk[ranks]                          # (G, b, s)
        Ui     = U_ell[ranks]                          # (G, b, rk)
        Yperp  = Yi - torch.bmm(Ui, torch.bmm(Ui.transpose(-2, -1), Yi))
        X1     = _pinv_rhs_via_gram(Yperp, Om_nf, M_i)   # (G, b, k*b)
        X1     = X1.reshape(G, b, k, b)                  # (G, b, k, b)
        for g in range(G):
            i = int(ranks[g])
            for a in range(k):
                j = int(nf_idx[g, a])
                tile[(i, j)] = X1[g, :, a, :].clone()    # term1 part

    # --- term2: per cluster j, solve against its OWN near-field Psi stack ------
    for grp in nearfield.groups:
        k      = grp['k']; ranks = grp['ranks']; nf_idx = grp['nf_idx']
        G      = grp['G']
        M_j, Psi_nf = _assemble_raw_gram(Psi_blk, nf_idx)  # (G,k*b,k*b),(G,k,b,s)
        Zj     = Z_blk[ranks]                           # (G, b, s)
        Vj     = V_ell[ranks]                           # (G, b, rk)
        Zperp  = Zj - torch.bmm(Vj, torch.bmm(Vj.transpose(-2, -1), Zj))
        X2     = _pinv_rhs_via_gram(Zperp, Psi_nf, M_j)  # (G, b, k*b)
        X2     = X2.reshape(G, b, k, b)                  # (G, b, k, b)
        for g in range(G):
            j = int(ranks[g])
            for a in range(k):
                i = int(nf_idx[g, a])
                # X2[g,:,a,:] = (I-VjVj^*) A_ij^*   ->  A_ij(I-VjVj^*) = its transpose
                Aij_perp = X2[g, :, a, :].transpose(-2, -1)   # (b,b)
                Ui = U_ell[i]
                t2 = Ui @ (Ui.transpose(-2, -1) @ Aij_perp)
                if (i, j) in tile:
                    tile[(i, j)] = tile[(i, j)] + t2
                else:
                    tile[(i, j)] = t2

    # --- pack into block-sparse tensor ---------------------------------------
    pairs = sorted(tile.keys())
    val = torch.stack([tile[p] for p in pairs], 0)       # (nnz, b, b)
    row = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    col = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
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


def compress_level_strong(Om_blk,Psi_blk,Y_blk,Z_blk, adj_sets, rk,device=None):
    """
    One-level strong-admissibility compression of a dense operator A (for
    testing / single-level use).  Returns (U, V, E) with U,V tensors (n,b,rk)
    and E a BlockSparseE.

    s : number of samples.  Default = m_max + 4*rk + 10 where m_max = k_max*b.
    """
    n = len(adj_sets)
    nfo = NearField(adj_sets, device=device)
    U = compute_basis_strong(Om_blk,  Y_blk, nfo, rk, device=device)
    V = compute_basis_strong(Psi_blk, Z_blk, nfo, rk, device=device)
    E = recover_E_strong(U, V, Y_blk, Z_blk, Om_blk, Psi_blk, nfo, device=device)
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
        self.device = device
        if A is not None:
            self.A      =   A
            self.shape  =   self.A.shape
            self.dtype  = A.dtype
            

        if tree is not None:
            self.tree   =   tree
            self.perm   =   tree.perm_leaf
            self.Nb = tree.nleaves
            self.nl = len(tree.get_box_inds(tree.get_leaves()[0]))
            self.shape = (self.nl*self.Nb,self.nl*self.Nb)
            self.L = tree.nlevels
            
            
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
        self.tSample = 0
        self.tConstruct = 0
        self.Nbvec = []
        self.quad = quad
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
    
    def construct(self,rk,Om,Psi,Y,Z):
        #assume numpy input here:
        Nb = self.Nb
        nl = Om.shape[0]//Nb
        Ompr = torch.from_numpy(Om).to(self.device)
        Psipr = torch.from_numpy(Psi).to(self.device)
        Ypr = torch.from_numpy(Y).to(self.device)
        Zpr = torch.from_numpy(Z).to(self.device)
        Om_blk = to_block_tensor(Ompr[self.perm,:], Nb, nl)
        Psi_blk = to_block_tensor(Psipr[self.perm,:], Nb, nl)
        Y_blk = to_block_tensor(Ypr[self.perm,:], Nb, nl)
        Z_blk = to_block_tensor(Zpr[self.perm,:], Nb, nl)
        self.constructHBS(rk,Om_blk,Psi_blk,Y_blk,Z_blk)

    def constructHBS(self,rk,Om_blk,Psi_blk,Y_blk,Z_blk):
        Nb = self.Nb
        nl = self.nl
        s = Om_blk.shape[2]
        rkm = rkm = min(rk,nl)
        Nbvec = [Nb]
        for lvl in range(self.L-1, -1, -1):
            adj_level = self.tree.level_adj_list[lvl]

            if has_farfield(adj_level):
                U, V, E, _ = compress_level_strong(Om_blk, Psi_blk, Y_blk, Z_blk,
                                                    adj_level, rkm, device=None)
                self.Umats += [U]
                self.Vmats += [V]
                self.Emats += [E]

                # --- reduction to the next coarser level (uses THIS level's U,V,E) ---
                Nb = Nb // self.fac
                rkm_used = U.shape[-1]                       # width actually produced

                Y_blk = Y_blk - E @ Om_blk
                Y_blk = torch.bmm(U.transpose(-2, -1), Y_blk)
                Y_blk = Y_blk.reshape(Nb, self.fac * rkm_used, s)

                Z_blk = Z_blk - E.T @ Psi_blk
                Z_blk = torch.bmm(V.transpose(-2, -1), Z_blk)
                Z_blk = Z_blk.reshape(Nb, self.fac * rkm_used, s)

                Om_blk  = torch.bmm(V.transpose(-2, -1), Om_blk).reshape(Nb, self.fac * rkm_used, s)
                Psi_blk = torch.bmm(U.transpose(-2, -1), Psi_blk).reshape(Nb, self.fac * rkm_used, s)

                self.Nbvec += [Nb]
                rkm = rk
            else:
                E = compute_final_E(Y_blk, Om_blk, device=self.device)
                self.Emats += [E]
                break
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
            v = torch.from_numpy(v).to(self.device)
        ref = self.Umats[0] if nlev else self.Emats[0].val
        v = v.to(ref.dtype)
        col_vec = (v.ndim == 1)
        vperm = v[self.perm, None] if col_vec else v[self.perm, :]
        cols = vperm.shape[1]

        downmats = self.Vmats if mode == 'N' else self.Umats
        upmats   = self.Umats if mode == 'N' else self.Vmats
        transpose = (mode == 'T')

        # ---- down-sweep: project + regroup fac children into a parent block 
        VV = [vperm.reshape(self.Nb, self.nl, cols)]
        for lvl in range(nlev):
            A = downmats[lvl]                                  # (Nb_lvl, b_lvl, rk)
            proj = torch.bmm(A.transpose(-2, -1), VV[lvl])     # (Nb_lvl, rk, cols)
            Nb_lvl, rk_lvl = A.shape[0], A.shape[2]
            VV.append(proj.reshape(Nb_lvl // self.fac, self.fac * rk_lvl, cols))

        # ---- dense base ----
        uperm = self.Emats[-1].matvec(VV[-1], transpose=transpose)

        # ---- up-sweep: split parent back to children, U(coarse) + E(this) ----
        for lvl in range(nlev - 1, -1, -1):
            U = upmats[lvl]                                    # (Nb_lvl, b_lvl, rk)
            Nb_lvl, rk_lvl = U.shape[0], U.shape[2]
            uperm = uperm.reshape(Nb_lvl, rk_lvl, cols)
            far  = torch.bmm(U, uperm)                          # (Nb_lvl, b_lvl, cols)
            near = self.Emats[lvl].matvec(VV[lvl], transpose=transpose)
            uperm = far + near

        # ---- flatten, un-permute, restore input shape ----
        uperm = uperm.reshape(-1, cols)
        u = torch.zeros_like(uperm)
        u[self.perm, :] = uperm
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