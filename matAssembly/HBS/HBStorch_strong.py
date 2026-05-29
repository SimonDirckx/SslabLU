import numpy as np
import time
import torch
import torch.linalg as tla
import matAssembly.HBS.ULVsparse_torch as ULVsparse

"""
HBStorch_strong — HBS with strong admissibility correction, binary tree only.

At each HBS level lvl, the sketch tensors Om_ell, Y_ell have shape
(Nb_lvl, block_size, s) where block index i directly addresses the i-th
node in sorted order at that tree level.  Near-neighbour pairs (i,j) from
tree.near_neighbours[lvl] index directly into these tensors.

The correction C_ell[i,j] has shape (block_size, block_size) and approximates
the off-diagonal block A_ell[i,j] via the sketch residual:

    residual_ell[i] = Y_ell[i] - U_ell[i] @ U_ell[i].T @ Y_ell[i]
    C_ell[i,j]      = residual_ell[i] @ pinv(Om_ell[j])

At the leaf level there is no U_ell, so residual_ell = Y_ell.

In the matvec ascending pass, VV[lvl+1] holds the compressed vector at
HBS level lvl (shape (Nb_lvl * rk, k) for lvl > 0, or (Nb * nl, k) for
the leaf level).  The correction is applied as:

    uperm[i*bs:(i+1)*bs] += C_ell[i,j] @ VV_lvl[j*bs:(j+1)*bs]

where bs is the block size at that level.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pin(tensor, device):
    if device is not None and device != 'cpu' and torch.cuda.is_available():
        return tensor.cpu().pin_memory()
    return tensor.cpu()

def _to_device(tensor, device, non_blocking=False):
    if device is None or device == 'cpu':
        return tensor.to('cpu')
    return tensor.to(device, non_blocking=non_blocking)

def block_matvec(A, B, device, mode='N'):
    Nb = A.shape[0]
    k  = B.shape[1]
    nB = B.shape[0] // Nb
    Bm = B.reshape(Nb, nB, k)
    if mode == 'N':
        return torch.bmm(A, Bm).reshape(Nb * A.shape[1], k)
    elif mode == 'T':
        return torch.bmm(A.mT, Bm).reshape(Nb * A.shape[2], k)
    else:
        raise ValueError("mode not recognized")

def block_mult(A, B, device, mode='N'):
    if mode == 'N':
        return torch.bmm(A, B)
    elif mode == 'T':
        return torch.bmm(A.mT, B)
    else:
        raise ValueError("mode not recognized")

def block_mult_and_reduce(A, B, fac, device, mode='N'):
    Nb = A.shape[0]
    if mode == 'N':
        C = torch.bmm(A, B)
        return C.contiguous().reshape(Nb // fac, fac * A.shape[1], B.shape[2])
    elif mode == 'T':
        C = torch.bmm(A.mT, B)
        return C.contiguous().reshape(Nb // fac, fac * A.shape[2], B.shape[2])
    else:
        raise ValueError("mode not recognized")

def block_solve_r(A, B, device):
    sol = tla.lstsq(B.mT, A.mT).solution
    return sol.mT

def _qr_solve(X, M):
    """Compute X @ pinv(M) via QR of M.T. X:(1,m,s), M:(1,n,s) -> (1,m,n)."""
    Q, R = tla.qr(M.mT)                                      # Q:(1,s,s), R:(1,s,n)
    QtXt = torch.bmm(Q.mT, X.mT)                             # (1,s,m)
    Z    = torch.linalg.solve_triangular(R, QtXt, upper=True) # (1,s,m) -- only if square
    return Z.mT                                               # (1,m,n) -- only if R square

def _pinv_solve(X, M):
    """
    Compute X @ pinv(M) for single blocks.
    X: (m, s), M: (n, s) -> (m, n).
    Uses lstsq: M.T @ out.T = X.T, i.e. solve (s,n) @ (n,m) = (s,m).
    """
    # tla.lstsq expects (..., s, n) and (..., s, m)
    sol = tla.lstsq(M.T.unsqueeze(0), X.T.unsqueeze(0)).solution  # (1, n, m)
    return sol.squeeze(0).T                                         # (m, n)

def compute_UV(Om, Y, rk, device):
    n  = Om.shape[1]
    Q  = tla.qr(Om.mT, mode='complete').Q
    Yn = torch.bmm(Y, Q[:, :, -n:])
    U  = tla.qr(Yn).Q
    return U[:, :, :rk].contiguous()

def construct_D_batched(U_ell, V_ell, Y_ell, Z_ell, Om_ell, Psi_ell, device):
    """Batched D construction matching HBStorch."""
    UtY = torch.bmm(U_ell.mT, Y_ell)
    res_Y = Y_ell - torch.bmm(U_ell, UtY)
    VtZ = torch.bmm(V_ell.mT, Z_ell)
    res_Z = Z_ell - torch.bmm(V_ell, VtZ)
    # Use lstsq per block for numerical stability
    Nb = Om_ell.shape[0]
    n  = U_ell.shape[1]
    D  = torch.zeros(Nb, n, n, dtype=Y_ell.dtype, device=device)
    for i in range(Nb):
        YP = _pinv_solve(res_Y[i], Om_ell[i])    # (n, n)
        ZP = _pinv_solve(res_Z[i], Psi_ell[i])   # (n, n)
        D[i] = YP + U_ell[i] @ (U_ell[i].T @ ZP.T)
    return D

def _fused_ascending(U, D, uperm, VV_lvl, device):
    """Fuse U@uperm + D@VV_lvl into one bmm."""
    Nb   = U.shape[0]
    k    = uperm.shape[1]
    nU   = uperm.shape[0] // Nb
    nV   = VV_lvl.shape[0] // Nb
    up_b = uperm.reshape(Nb, nU, k)
    vv_b = VV_lvl.reshape(Nb, nV, k)
    x    = torch.cat([up_b, vv_b], dim=1)
    UD   = torch.cat([U, D], dim=2)
    return torch.bmm(UD, x).reshape(Nb * UD.shape[1], k)

# ---------------------------------------------------------------------------
# Near-neighbour correction
# ---------------------------------------------------------------------------

def compute_corrections(residual_ell, Om_ell, near_pairs, device):
    """
    Compute near-neighbour correction matrices at one HBS level.

    residual_ell : (Nb, bs, s) — sketch residual at this level
    Om_ell       : (Nb, bs, s) — sketch random matrix at this level
    near_pairs   : list of (i, j) block-index pairs at this level

    Returns list of (bs, bs) tensors, one per pair.
    Block size bs = residual_ell.shape[1].
    """
    corrections = []
    for (i, j) in near_pairs:
        C_ij = _pinv_solve(residual_ell[i], Om_ell[j])   # (bs, bs)
        corrections.append(C_ij)
    return corrections

def apply_corrections(uperm, v_lvl, corrections, near_pairs, bs, device):
    """
    Apply near-neighbour corrections in the matvec.

    uperm       : (Nb*bs, k) flat output at this level — modified in place
    v_lvl       : (Nb*bs, k) flat input at this level (VV entry for this level)
    corrections : list of (bs, bs) correction tensors
    near_pairs  : list of (i, j) block-index pairs
    bs          : block size at this level
    """
    for (i, j), C in zip(near_pairs, corrections):
        uperm[i*bs:(i+1)*bs, :] += C.to(device) @ v_lvl[j*bs:(j+1)*bs, :]
    return uperm

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HBSMAT_STRONG:
    """
    HBS with strong admissibility correction. Binary tree only.

    tree must have near_neighbours built via tree.build_near_neighbours().
    near_neighbours[hbs_lvl] gives (i,j) pairs of block indices at that
    HBS level, directly addressing Om_ell[i], Om_ell[j] etc.
    """

    def __init__(self, A=None, device=None, tree=None):
        self.Umats      = []
        self.Vmats      = []
        self.Dmats      = []
        self.Cmats      = []       # corrections per HBS level, finest first
        self.near_pairs = []       # (i,j) block pairs per HBS level

        self.mode   = 'N'
        self._tree  = None
        self.fac    = 2

        if A is not None:
            self.A      = A
            self.shape  = A.shape
            self.dtype  = A.dtype
            self.device = device
            torch.set_default_dtype(torch.float64)

        if tree is not None:
            self.tree = tree
            self.perm = tree.perm_leaf
            self.Nb   = tree.nleaves
            self.nl   = self.A.shape[0] // self.Nb
            self.L    = tree.nlevels
            if not hasattr(tree, 'near_neighbours'):
                raise ValueError(
                    "tree.build_near_neighbours() must be called first.")

        self.blockSolveTime = 0
        self.nullTime       = 0
        self.setupTime      = 0
        self.DTime          = 0
        self.tSample        = 0
        self.tConstruct     = 0
        self.tCompress      = 0
        self.nSamples       = 0
        self.Nbvec          = []
        self.quad           = False

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, t):
        self._tree = t

    @property
    def nbytes(self):
        ctr  = sum(U.nbytes for U in self.Umats)
        ctr += sum(V.nbytes for V in self.Vmats)
        ctr += sum(D.nbytes for D in self.Dmats)
        ctr += sum(C.nbytes for Cl in self.Cmats for C in Cl)
        return ctr

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def construct(self, rk):
        self.constructHBS_strong(rk)

    def constructHBS_strong(self, rk):
        nl  = self.nl
        Nb  = self.Nb
        fac = self.fac

        # Inflate s to capture near-neighbour off-diagonal blocks.
        # Max pairs per block across all levels gives a conservative bound.
        max_nn_per_block = 0
        for pairs in self.tree.near_neighbours:
            if pairs:
                from collections import Counter
                cnt = Counter(i for (i, j) in pairs)
                max_nn_per_block = max(max_nn_per_block, max(cnt.values()))
        p = max(1, max_nn_per_block)
        s = (2 + p) * rk + (2 + p) * nl + 5
        self.nSamples = s

        tic = time.time()
        Om_flat  = torch.randn(self.shape[1], s, dtype=torch.float64,
                               device=self.device)
        Psi_flat = torch.randn(self.shape[0], s, dtype=torch.float64,
                               device=self.device)
        Omprime  = torch.zeros_like(Om_flat)
        Omprime[self.perm, :]  = Om_flat
        Psiprime = torch.zeros_like(Psi_flat)
        Psiprime[self.perm, :] = Psi_flat

        Y_np = self.A   @ Omprime.cpu().numpy()
        Z_np = self.A.T @ Psiprime.cpu().numpy()

        Y = torch.from_numpy(Y_np[self.perm, :]).to(self.device)
        Z = torch.from_numpy(Z_np[self.perm, :]).to(self.device)

        Y   = ULVsparse.convert_to_torch_tens(Y,        self.Nb, device=self.device)
        Z   = ULVsparse.convert_to_torch_tens(Z,        self.Nb, device=self.device)
        Om  = ULVsparse.convert_to_torch_tens(Om_flat,  self.Nb, device=self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psi_flat, self.Nb, device=self.device)

        self.setupTime += time.time() - tic
        self.tSample   += time.time() - tic

        tic_compress = time.time()
        for lvl in range(self.L - 1, -1, -1):

            if lvl == self.L - 1:
                Om_ell  = Om
                Psi_ell = Psi
                Y_ell   = Y
                Z_ell   = Z
                rkm     = min(rk, nl)
            else:
                Y_ell   -= block_mult(D_ell, Om_ell,  self.device)
                Y_ell    = block_mult_and_reduce(U_ell, Y_ell,  fac, self.device, mode='T')
                Z_ell   -= block_mult(D_ell, Psi_ell, self.device, mode='T')
                Z_ell    = block_mult_and_reduce(V_ell, Z_ell,  fac, self.device, mode='T')
                Om_ell   = block_mult_and_reduce(V_ell, Om_ell, fac, self.device, mode='T')
                Psi_ell  = block_mult_and_reduce(U_ell, Psi_ell,fac, self.device, mode='T')
                Nb  = Nb // fac
                rkm = min(rk, nl * (fac ** (self.L - 1 - lvl)))

            print(f"lvl={lvl}  Nb={Nb}  Om_ell shape={Om_ell.shape}")
            self.Nbvec += [Nb]

            # near_neighbours[lvl] gives (i,j) block pairs at this HBS level.
            # These directly index Om_ell[i], Y_ell[i] etc.
            nn_pairs = self.tree.near_neighbours[lvl]            # block size at this level
            bs = Om_ell.shape[1]

            if lvl > 0:
                tic = time.time()
                U_ell = compute_UV(Om_ell,  Y_ell,  rkm, self.device)
                V_ell = compute_UV(Psi_ell, Z_ell,  rkm, self.device)
                self.nullTime += time.time() - tic

                tic = time.time()
                D_ell = construct_D_batched(U_ell, V_ell, Y_ell, Z_ell,
                                            Om_ell, Psi_ell, self.device)
                self.DTime += time.time() - tic

                # Sketch residual: what U_ell did not capture
                residual = Y_ell - torch.bmm(U_ell, torch.bmm(U_ell.mT, Y_ell))
                corrections = compute_corrections(residual, Om_ell,
                                                  nn_pairs, self.device)

                self.Dmats.append(D_ell)
                self.Umats.append(U_ell)
                self.Vmats.append(V_ell)
                self.Cmats.append(corrections)
                self.near_pairs.append(nn_pairs)

            else:
                # Leaf level: D_ell is block-diagonal dense solve.
                # Near-neighbour correction uses full Y_ell (no U_ell).
                tic = time.time()
                D_ell = block_solve_r(Y_ell, Om_ell, self.device)
                self.blockSolveTime += time.time() - tic
                self.Dmats.append(_pin(D_ell, self.device))

                corrections = compute_corrections(Y_ell, Om_ell,
                                                  nn_pairs, self.device)
                self.Cmats.append(corrections)
                self.near_pairs.append(nn_pairs)

        self.tCompress = time.time() - tic_compress

    # -----------------------------------------------------------------------
    # Matvec helpers
    # -----------------------------------------------------------------------

    def _descend_N(self, vperm):
        """Descending pass (mode N): project v through Vmats, collect VV."""
        VV = [vperm]
        Nb = self.Nb
        for lvl_idx in range(len(self.Vmats)):
            v_lvl = block_matvec(self.Vmats[lvl_idx], VV[-1],
                                 self.device, mode='T')
            VV.append(v_lvl)
            Nb = Nb // self.fac
        return VV

    def _descend_T(self, vperm):
        """Descending pass (mode T): project v through Umats, collect VV."""
        VV = [vperm]
        Nb = self.Nb
        for lvl_idx in range(len(self.Umats)):
            v_lvl = block_matvec(self.Umats[lvl_idx], VV[-1],
                                 self.device, mode='T')
            VV.append(v_lvl)
            Nb = Nb // self.fac
        return VV

    def _ascend_N(self, VV):
        """
        Ascending pass (mode N).

        VV[0] = vperm, shape (Nb*nl, k)  — leaf resolution input
        VV[j] for j=1..nU = compressed input at non-leaf levels,
               shape (Nb_j * rk_j, k) where Nb_j = Nb // fac^j

        At each non-leaf level j (j=1 is finest non-leaf, j=nU is coarsest):
            Umats[nU-j]: (Nb_j, nl_j, rk_j)   — basis
            Dmats[nU-j]: (Nb_j, nl_j, nl_j)   — diagonal block
            uperm coming in has shape (Nb_j * rk_j, k)
              where rk_j = Umats[nU-j].shape[2]
            output = U @ uperm_blocked + D @ VV[j]_blocked
              both terms have shape (Nb_j * nl_j, k)

        Special case: after the leaf multiply, uperm has shape (Nb*nl, k).
        At the first non-leaf step (j=1, k=nU-1):
            Umats[nU-1]: (Nb//fac, nl_parent, rk_finest)
            uperm: (Nb*nl, k) = (Nb//fac * fac*nl, k)
            This is Nb//fac blocks of size fac*nl.
            U[i]:(nl_parent, rk_finest) needs input of size rk_finest.
            But we have blocks of size fac*nl = rk_finest only if rk_finest=fac*nl.
            In general rk_finest = min(rk, nl), so fac*nl != rk_finest.

        The resolution: uperm after the leaf step must be *compressed* before
        being passed up. The leaf D multiply gives output at leaf resolution.
        We then need block_matvec(U[nU-1], uperm, mode='T') to compress it,
        then add D[nU-1] @ VV[1] to get the diagonal contribution.
        This matches the HBS formula:
            uperm_new = U_ell @ (U_ell.T @ uperm_child) + D_ell @ v_ell
        where U_ell.T @ uperm_child is the compression step.

        Wait — that's not right either. The correct HBS ascending recurrence is:
            u_ell[i] = U_ell[i] @ u_{ell-1}[i] + D_ell[i] @ v_ell[i]
        where u_{ell-1}[i] is the *already compressed* output from level ell-1,
        having shape (rk, k). And U_ell[i] has shape (nl_ell, rk).
        So: block_matvec(U, uperm_rk, 'N') + block_matvec(D, VV_lvl, 'N')
        Both give (Nb_ell * nl_ell, k).

        The leaf output is nl-sized per block (not rk-sized). The leaf IS the
        bottom of the tree, so we go directly from leaf-resolution uperm to the
        first non-leaf level. At the first non-leaf level:
            uperm_rk has shape (Nb//fac * rk_finest, k)
        But uperm from the leaf is (Nb*nl, k). We need rk_finest-sized blocks.
        This means the leaf-level uperm must be *reduced* by fac blocks into one
        and compressed to rk. This is exactly what block_matvec(U, uperm, 'T')
        does with U:(Nb//fac, nl_parent, rk): it treats uperm as Nb//fac blocks
        of size fac*nl... no, nB = Nb*nl/(Nb//fac) = fac*nl, and U has rk cols.
        bmm(U.mT, uperm_reshaped) = (Nb//fac, rk, k). That's the compression.

        So the correct ascending step is:
            uperm_compressed = block_matvec(U, uperm_prev, 'T')  # compress
            uperm_new = block_matvec(U, uperm_compressed, 'N') + block_matvec(D, VV, 'N')
        But that's U @ U.T @ uperm which loses information.

        Actually looking at HBStorch matmat more carefully:
            uperm = block_matvec(D_leaf, VV[-1])   # (Nb*nl, col) — this IS the output
            for lvl in range(nU-1, -1, -1):
                uperm = U@uperm + D@VV[lvl]   via _fused_ascending
        In HBStorch, at each non-leaf level, uperm coming in has shape
        (Nb_lvl * rk_lvl, col) — it is already rk-sized because the D_leaf in
        HBStorch covers ALL leaf blocks (Nb=1 at the coarsest, full dense block).
        HBStorch processes everything in one level above the leaves.

        In HBStorch_strong (binary tree), there are L-1 non-leaf levels and 1
        leaf level with Nb=nleaves. After the leaf D multiply:
            uperm: (Nb*nl, col)  with Nb=nleaves
        At the first ascending non-leaf step (finest non-leaf):
            U: (Nb//fac, nl_parent, rk)
            input to U should be rk-sized: (Nb//fac * rk, col)
            but we have (Nb*nl, col) = (Nb//fac * fac*nl, col)
        So fac*nl must equal rk at this level for block_matvec(U,'N') to work.
        This is only guaranteed if rk >= fac*nl, which defeats compression.

        The correct interpretation: in the standard HBS ascending pass,
            u_parent = U_parent @ u_child_compressed + D_parent @ v_parent
        where u_child_compressed has shape (rk,) per block — it's the output
        of the *child level's* ascending pass, already rk-sized.
        At the leaf level, the "child output" for non-leaf level j=1 is not
        a D multiply — it's the *projection* of the leaf output onto the basis:
            u_leaf_compressed[i] = U_finest[i].T @ D_leaf[i] @ v_leaf[i]
        Then:
            u_finest[i] = U_finest[i] @ u_leaf_compressed[i] + D_finest[i] @ v_finest[i]

        But this is also not right for HBStorch where D_leaf is the full bottom.

        The simplest correct fix: treat the leaf output as rk-sized by doing the
        compression explicitly before entering the loop.
        """
        nl   = self.nl
        fac  = self.fac
        nU   = len(self.Umats)

        # leaf level
        D_leaf = _to_device(self.Dmats[-1], self.device, non_blocking=True)
        uperm  = block_matvec(D_leaf, VV[-1], self.device)
        leaf_lvl = self.L - 1
        uperm = apply_corrections(uperm, VV[-1],
                                  self.Cmats[leaf_lvl],
                                  self.near_pairs[leaf_lvl],
                                  nl, self.device)
        # uperm: (Nb*nl, col) — leaf resolution

        # ascending non-leaf levels, k=nU-1 is finest non-leaf
        for k in range(nU - 1, -1, -1):
            hbs_lvl = k
            U = self.Umats[k]   # (Nb_k, nl_k, rk_k)
            D = self.Dmats[k]   # (Nb_k, nl_k, nl_k)
            # uperm has shape (Nb_child * something, col)
            # block_matvec(U, uperm, 'N'): nB = uperm.shape[0]//U.shape[0]
            #   = Nb_child*nl_child / Nb_k = fac * nl_child (for leaf step)
            #   or = rk_{k+1} (for non-leaf steps, since previous step outputs nl_k)
            # For the leaf step: uperm is (Nb*nl, col), U[k=nU-1]:(Nb//fac, nl_p, rk)
            #   nB = Nb*nl/(Nb//fac) = fac*nl  -> U[i]:(nl_p,rk) @ B:(fac*nl,col)
            #   -> this is wrong dimension unless rk=fac*nl
            # FIX: compress uperm to rk first, then expand
            uperm_c = block_matvec(U, uperm, self.device, mode='T')   # (Nb_k*rk_k, col)
            u_part  = block_matvec(U, uperm_c, self.device, mode='N') # (Nb_k*nl_k, col)
            d_part  = block_matvec(D, VV[k + 1], self.device, mode='N') # (Nb_k*nl_k, col)
            uperm   = u_part + d_part
            bs = U.shape[1]
            uperm = apply_corrections(uperm, VV[k + 1],
                                      self.Cmats[hbs_lvl],
                                      self.near_pairs[hbs_lvl],
                                      bs, self.device)
        return uperm

    def _ascend_T(self, VV):
        """Ascending pass (mode T)."""
        nl  = self.nl
        nV  = len(self.Vmats)

        D_leaf = _to_device(self.Dmats[-1], self.device, non_blocking=True)
        uperm  = block_matvec(D_leaf, VV[-1], self.device, mode='T')
        leaf_lvl = self.L - 1
        uperm = apply_corrections(uperm, VV[-1],
                                  [C.mT for C in self.Cmats[leaf_lvl]],
                                  [(j, i) for (i, j) in self.near_pairs[leaf_lvl]],
                                  nl, self.device)

        for k in range(nV - 1, -1, -1):
            hbs_lvl = k
            V = self.Vmats[k]
            D = self.Dmats[k]
            uperm_c = block_matvec(V, uperm, self.device, mode='T')
            v_part  = block_matvec(V, uperm_c, self.device, mode='N')
            d_part  = block_matvec(D.mT, VV[k + 1], self.device, mode='N')
            uperm   = v_part + d_part
            bs = V.shape[1]
            uperm = apply_corrections(uperm, VV[k + 1],
                                      [C.mT for C in self.Cmats[hbs_lvl]],
                                      [(j, i) for (i, j) in self.near_pairs[hbs_lvl]],
                                      bs, self.device)
        return uperm

    # -----------------------------------------------------------------------
    # Public matvec interface
    # -----------------------------------------------------------------------

    def _prep(self, v):
        numpy_input = isinstance(v, np.ndarray)
        if numpy_input:
            v = torch.from_numpy(v).to(self.device)
        v = v.to(self.Dmats[0].dtype)
        if v.ndim == 1:
            vperm = v[self.perm, None]
        else:
            vperm = v[self.perm, :]
        return v, vperm, numpy_input

    def _unprep(self, v, uperm, numpy_input):
        u = torch.empty(uperm.shape, device=self.device, dtype=uperm.dtype)
        u[self.perm, :] = uperm
        if v.ndim == 1:
            u = u.flatten()
        if numpy_input:
            u = u.cpu().numpy()
        return u

    def matvec(self, v):
        return self.matmat(v)

    def rmatvec(self, v):
        return self.rmatmat(v)

    def matmat(self, v):
        v, vperm, numpy_input = self._prep(v)
        VV    = self._descend_N(vperm)
        uperm = self._ascend_N(VV)
        return self._unprep(v, uperm, numpy_input)

    def rmatmat(self, v):
        v, vperm, numpy_input = self._prep(v)
        VV    = self._descend_T(vperm)
        uperm = self._ascend_T(VV)
        return self._unprep(v, uperm, numpy_input)

    def __matmul__(self, v):
        v, vperm, numpy_input = self._prep(v)
        if self.mode == 'N':
            VV    = self._descend_N(vperm)
            uperm = self._ascend_N(VV)
        elif self.mode == 'T':
            VV    = self._descend_T(vperm)
            uperm = self._ascend_T(VV)
        else:
            raise ValueError("mode not recognized")
        return self._unprep(v, uperm, numpy_input)