import numpy as np
import scipy.linalg as splinalg
import time
import matAssembly.HBS.ULVsparse as ULVsparse
import matAssembly.HBS.ULVsparse_torch as ULVsparse_torch
import torch.linalg as tla
import scipy.linalg as sclinalg


# ---------------------------------------------------------------------------
# Utility functions — identical to HBSnew.py
# ---------------------------------------------------------------------------

def block_solve_r(A, B, Nb):
    nb = B.shape[0] // Nb
    n  = A.shape[0] // Nb
    C  = np.zeros(shape=(A.shape[0], nb))
    for i in range(Nb):
        C[i*n:(i+1)*n, :] = (A[i*n:(i+1)*n, :]
                              @ np.linalg.pinv(B[i*nb:(i+1)*nb, :], rcond=1e-15))
    return C


def compute_UV(Om, Y, rk, Nb):
    nloc = Om.shape[0] // Nb
    U    = np.zeros(shape=(Y.shape[0], rk))
    n    = Om.shape[0] // Nb
    nU   = U.shape[0]  // Nb
    for i in range(Nb):
        Q = np.linalg.qr(Om[i*n:(i+1)*n, :].T, mode='complete')[0]
        U[i*nU:(i+1)*nU, :] = (
            np.linalg.svd(Y[i*nU:(i+1)*nU, :] @ Q[:, -nloc:])[0])[:, :rk]
    return U


def compute_UV_far(Y, Z, rk, Nb, nl, adj_leaf_list):
    """
    Compute leaf-level bases U, V using the null space of the near-field
    sketch rows, so that U captures only the far-field range of each block.

    For leaf r with near-field J_r = {r} union adj[r]:
      - Stack Y[J_r, :] (shape |J_r|*nl x s) and compute its null space Q_perp.
      - U_r = left singular vectors of Y_r @ Q_perp  (far-field directions only).
      - V_r = left singular vectors of Z_r @ Q_perp_Z where Q_perp_Z is the
        null space of Z[J_r, :].

    Parameters
    ----------
    Y, Z          : (Nb*nl, s_samples) full sketches
    rk            : target rank
    Nb            : number of leaves
    nl            : DOFs per leaf
    adj_leaf_list : list of sets of adjacent leaf ranks

    Returns
    -------
    U, V : (Nb*nl, rk) basis matrices
    """
    U = np.zeros((Nb * nl, rk))
    V = np.zeros((Nb * nl, rk))

    for r in range(Nb):
        near_r = sorted([r] + list(adj_leaf_list[r]))

        Yr = Y[r*nl:(r+1)*nl, :]   # (nl, s)
        Zr = Z[r*nl:(r+1)*nl, :]   # (nl, s)

        # Null space of near-field Y rows: vectors in sample space
        # invisible to all near-field blocks of r.
        Y_near = np.vstack([Y[s*nl:(s+1)*nl, :] for s in near_r])  # (|J_r|*nl, s)
        Q_Y    = np.linalg.qr(Y_near.T, mode='complete')[0]         # (s, s)
        # Q_Y[:, |J_r|*nl:] spans null space of Y_near
        n_near_Y = Y_near.shape[0]
        Q_perp_Y = Q_Y[:, n_near_Y:]                                # (s, s-|J_r|*nl)

        # U_r from Y_r projected onto far-field sample directions
        U[r*nl:(r+1)*nl, :] = (
            np.linalg.svd(Yr @ Q_perp_Y)[0])[:, :rk]

        # Same for V using Z
        Z_near = np.vstack([Z[s*nl:(s+1)*nl, :] for s in near_r])  # (|J_r|*nl, s)
        Q_Z    = np.linalg.qr(Z_near.T, mode='complete')[0]         # (s, s)
        n_near_Z = Z_near.shape[0]
        Q_perp_Z = Q_Z[:, n_near_Z:]                                # (s, s-|J_r|*nl)

        V[r*nl:(r+1)*nl, :] = (
            np.linalg.svd(Zr @ Q_perp_Z)[0])[:, :rk]

    return U, V


def block_mult(A, B, Nb, mode='N'):
    if mode == 'N':
        C  = np.zeros(shape=(A.shape[0], B.shape[1]))
        kA = A.shape[1]
        n  = A.shape[0] // Nb
        for i in range(Nb):
            C[i*n:(i+1)*n, :] = A[i*n:(i+1)*n, :] @ B[i*kA:(i+1)*kA, :]
    elif mode == 'T':
        kA = A.shape[1]
        C  = np.zeros(shape=(kA * Nb, B.shape[1]))
        n  = A.shape[0] // Nb
        for i in range(Nb):
            C[i*kA:(i+1)*kA, :] = A[i*n:(i+1)*n, :].T @ B[i*n:(i+1)*n, :]
    else:
        raise ValueError("mode not recognized")
    return C


def construct_D(U_ell, V_ell, Y_ell, Z_ell, Om_ell, Psi_ell, Nb):
    """Block-diagonal D for levels strictly above the leaf — unchanged."""
    C = np.zeros(shape=(U_ell.shape[0], Om_ell.shape[0] // Nb))
    n = U_ell.shape[0] // Nb
    for i in range(Nb):
        Usub   = U_ell  [i*n:(i+1)*n, :]
        Vsub   = V_ell  [i*n:(i+1)*n, :]
        Ysub   = Y_ell  [i*n:(i+1)*n, :]
        Zsub   = Z_ell  [i*n:(i+1)*n, :]
        Omsub  = Om_ell [i*n:(i+1)*n, :]
        Psisub = Psi_ell[i*n:(i+1)*n, :]
        C[i*n:(i+1)*n, :] = (
            (Ysub - Usub @ (Usub.T @ Ysub)) @ np.linalg.pinv(Omsub)
            + Usub @ (Usub.T @ (
                ((Zsub - Vsub @ (Vsub.T @ Zsub))
                 @ np.linalg.pinv(Psisub)).T)))
    return C


# ---------------------------------------------------------------------------
# Leaf-level sparse-block D helpers
# ---------------------------------------------------------------------------

def construct_D_leaf(U_ell, V_ell, Y_ell, Z_ell, Om_ell, Psi_ell,
                     Nb, nl, adj_leaf_list):
    """
    Build the leaf-level near-field correction as a sparse block dict.

    For each leaf rank r, define its near-field
        J_r = {r} union adj_leaf_list[r]
    (sorted for determinism).  Stack Om_ell and Psi_ell for J_r, then apply:

        E_r = (I - U_r U_r^T) Y_r @ pinv(Om[J_r, :])
            + U_r U_r^T @ ((I - V_r V_r^T) Z_r @ pinv(Psi[J_r, :])).T

    which has shape (nl, |J_r|*nl).  Unpack into per-(r,s) blocks of shape
    (nl, nl) and store in the dict D_leaf[(r, s)] for all s in J_r.

    Both (r, s) and (s, r) entries are stored, each computed from block r's
    sketch; they are generally NOT transposes of each other for non-symmetric A.

    Parameters
    ----------
    U_ell, V_ell    : (Nb*nl, rk)
    Y_ell, Z_ell    : (Nb*nl, s_samples)
    Om_ell, Psi_ell : (Nb*nl, s_samples)
    Nb              : number of leaves
    nl              : DOFs per leaf
    adj_leaf_list   : list of sets, adj_leaf_list[r] = adjacent leaf ranks

    Returns
    -------
    D_leaf : dict {(r, s): np.ndarray shape (nl, nl)}
    """
    D_leaf = {}
    for r in range(Nb):
        near_r = sorted([r] + list(adj_leaf_list[r]))

        Ur = U_ell[r*nl:(r+1)*nl, :]
        Vr = V_ell[r*nl:(r+1)*nl, :]
        Yr = Y_ell[r*nl:(r+1)*nl, :]
        Zr = Z_ell[r*nl:(r+1)*nl, :]

        # Stack Om and Psi for the entire near-field  (|J_r|*nl, s_samples)
        Om_Jr  = np.vstack([Om_ell [s*nl:(s+1)*nl, :] for s in near_r])
        Psi_Jr = np.vstack([Psi_ell[s*nl:(s+1)*nl, :] for s in near_r])

        # First term:  (I - U_r U_r^T) Y_r @ pinv(Om_Jr)  -> (nl, |J_r|*nl)
        # Vectorised over all s in J_r at once.
        proj_Y = ((Yr - Ur @ (Ur.T @ Yr))
                  @ np.linalg.pinv(Om_Jr, rcond=1e-15))

        # Second term from proof, assembled per adjacent pair (r,s):
        #   U_r U_r^T A[r,s](I - V_s V_s^T)
        # recovered as:
        #   U_r U_r^T @ [(I - V_s V_s^T) Z_s @ pinv(Psi_r)]^T    shape (nl,nl)
        # where Z_s = A[:,s]^T Psi  so  Z_s @ pinv(Psi_r) ~ A[r,s]^T
        # and (I-Vs Vs^T) projects out the far-field of s.
        # Psi_r = Psi_ell[r*nl:(r+1)*nl, :], precomputed once for this r.
        Psi_r    = Psi_ell[r*nl:(r+1)*nl, :]
        pinv_Psi_r = np.linalg.pinv(Psi_r, rcond=1e-15)  # (s, nl)

        proj_Z = np.zeros((nl, len(near_r) * nl))
        for k, s in enumerate(near_r):
            Zs = Z_ell[s*nl:(s+1)*nl, :]                   # (nl, s_samples)
            Vs = V_ell[s*nl:(s+1)*nl, :]                   # (nl, rk)
            # (I - Vs Vs^T) Zs @ pinv(Psi_r)  -> (nl, nl)
            inner = (Zs - Vs @ (Vs.T @ Zs)) @ pinv_Psi_r  # (nl, nl)
            # U_r U_r^T @ inner^T               -> (nl, nl)
            proj_Z[:, k*nl:(k+1)*nl] = Ur @ (Ur.T @ inner.T)

        E_r = proj_Y + proj_Z   # (nl, |J_r|*nl)

        for k, s in enumerate(near_r):
            D_leaf[(r, s)] = E_r[:, k*nl:(k+1)*nl]

    return D_leaf


def D_leaf_mult(D_leaf, v, Nb, nl, adj_leaf_list, mode='N'):
    """
    Apply the sparse-block leaf D to a block vector v.

    mode='N' (forward):    out[r] = sum_{s in J_r}  D_leaf[(r,s)] @ v[s]
    mode='T' (transpose):  out[r] = sum_{s in J_r}  D_leaf[(s,r)].T @ v[s]

    For the transpose, (H^T)[r,s] = H[s,r]^T so we look up D_leaf[(s,r)].
    The loop is equivalent: for each r, gather from all s whose near-field
    contains r (i.e. r in J_s), which by symmetry of the admissibility
    condition equals {r} union adj_leaf_list[r] = J_r.
    """
    out = np.zeros_like(v)
    for r in range(Nb):
        near_r = [r] + list(adj_leaf_list[r])
        for s in near_r:
            vs = v[s*nl:(s+1)*nl, :]
            if mode == 'N':
                out[r*nl:(r+1)*nl, :] += D_leaf[(r, s)] @ vs
            else:
                out[r*nl:(r+1)*nl, :] += D_leaf[(s, r)].T @ vs
    return out


def peel_D_leaf(Y_ell, Z_ell, Om_ell, Psi_ell, D_leaf, Nb, nl, adj_leaf_list):
    """
    Subtract leaf near-field contributions from Y_ell and Z_ell in-place
    before coarsening to the next level.

    For each leaf r and each s in J_r = {r} union adj[r]:
        Y_ell[r] -= D_leaf[(r, s)] @ Om_ell[s]
        Z_ell[r] -= D_leaf[(s, r)].T @ Psi_ell[s]

    The Z peel uses D_leaf[(s,r)].T because Z[r] = A^T[:,r] Psi contains
    A[s,r]^T Psi[s] for adjacent s, and D_leaf[(s,r)] ~ A[s,r].
    """
    for r in range(Nb):
        near_r = [r] + list(adj_leaf_list[r])
        for s in near_r:
            Y_ell[r*nl:(r+1)*nl, :] -= D_leaf[(r, s)] @ Om_ell [s*nl:(s+1)*nl, :]
            Z_ell[r*nl:(r+1)*nl, :] -= D_leaf[(s, r)].T @ Psi_ell[s*nl:(s+1)*nl, :]


# ---------------------------------------------------------------------------
# HBSMAT_STRONG
# ---------------------------------------------------------------------------

class HBSMAT_STRONG:
    """
    HBS matrix with strong (arbitrary symmetric) admissibility at the leaf level.

    Storage layout — mirrors HBSMAT exactly except Dmats[0] = None:

        Umats[i], Vmats[i]  : basis matrices, leaf-first (i=0 is leaf level)
        Dmats[i]            : i=0  -> None  (leaf D is in self.Dleaf)
                              i=1..L-2 -> block-diagonal D (as in HBSMAT)
                              i=L-1    -> root dense solve
        Dleaf               : dict {(r,s): (nl,nl) array}, near-field at leaves

    matmat / rmatmat are identical to HBSMAT except that when the downward
    pass reaches lvl=0, it calls D_leaf_mult instead of block_mult(Dmats[0],...).
    """

    def __init__(self, A=None, tree=None, quad=False):
        self.Umats  = []
        self.Vmats  = []
        self.Dmats  = []   # Dmats[0]=None (sentinel for leaf), rest as HBSMAT
        self.Dleaf  = {}   # sparse block near-field dict at leaf level
        self.Qlist  = []
        self.Rlist  = []
        self.Wlist  = []
        self.Uulist = []

        self.mode = 'N'
        self.fac  = 4 if quad else 2
        self.quad = quad

        if A is not None:
            self.A     = A
            print("A shape =", A.shape)
            self.shape = A.shape
            self.dtype = A.dtype

        if tree is not None:
            if not hasattr(tree, 'level_adj_list'):
                raise AttributeError(
                    "tree must have level_adj_list. "
                    "Call tree._build_level_adjacency() first.")
            self.perm  = tree.perm_leaf
            self.Nb    = tree.nleaves
            self.nl    = A.shape[0] // tree.nleaves
            self.L     = tree.nlevels
            self.tree  = tree
            # adj_leaf_list[r] = set of leaf ranks adjacent to leaf rank r.
            # level_adj_list is indexed by tree depth; leaves are at depth L-1.
            self.adj_leaf_list = tree.level_adj_list[tree.nlevels - 1]

        self.blockSolveTime = 0
        self.nullTime       = 0
        self.setupTime      = 0
        self.DTime          = 0
        self.tSample        = 0
        self.tCompress      = 0
        self.Nbvec          = []

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

    @property
    def nbytes(self):
        ctr  = sum(U.nbytes for U in self.Umats)
        ctr += sum(V.nbytes for V in self.Vmats)
        ctr += sum(D.nbytes for D in self.Dmats if D is not None)
        ctr += sum(B.nbytes for B in self.Dleaf.values())
        return ctr

    def construct(self, rk, compute_ULV=False):
        if compute_ULV:
            raise NotImplementedError(
                "ULV not yet implemented for strong admissibility.")
        self.constructHBS(rk)

    # ------------------------------------------------------------------
    def constructHBS(self, rk):
        """
        Build the strongly-admissible HBS representation.

        Identical to HBSMAT.constructHBS except at the leaf level:
          - D_leaf (sparse block) replaces the block-diagonal D.
          - Dmats[0] = None is appended as a sentinel so that all
            subsequent index arithmetic mirrors the original exactly.
          - peel_D_leaf replaces block_mult(D,...) when coarsening from
            the leaf level.
        """
        if self.fac == 4:
            s = 6 * rk + 4 * self.nl + 5
        else:
            s = 4 * rk + 2 * self.nl + 5
        self.nSamples = s

        rng = np.random.default_rng()
        tic = time.time()

        Om  = rng.standard_normal(size=(self.shape[1], s))
        Psi = rng.standard_normal(size=(self.shape[0], s))

        Omprime            = np.zeros(shape=Om.shape)
        Omprime[self.perm, :]  = Om
        Psiprime           = np.zeros(shape=Psi.shape)
        Psiprime[self.perm, :] = Psi

        Y = self.A.matmat (Omprime)[self.perm, :]
        Z = self.A.rmatmat(Psiprime)[self.perm, :]
        del Omprime, Psiprime

        Nb = self.Nb
        nl = self.nl
        self.tSample += time.time() - tic

        tic_compress = time.time()

        for lvl in range(self.L - 1, -1, -1):

            if lvl == self.L - 1:
                # ---- leaf level -------------------------------------------
                Om_ell  = Om
                Psi_ell = Psi
                Y_ell   = Y.copy()
                Z_ell   = Z.copy()
                del Y, Z
                rkm = min(rk, nl)

                # Compute bases from far-field null space of near-field rows
                tic2  = time.time()
                U_ell, V_ell = compute_UV_far(
                    Y_ell, Z_ell, rkm, Nb, nl, self.adj_leaf_list)
                self.nullTime += time.time() - tic2

                tic2 = time.time()
                D_leaf = construct_D_leaf(
                    U_ell, V_ell, Y_ell, Z_ell, Om_ell, Psi_ell,
                    Nb, nl, self.adj_leaf_list)
                self.Dleaf = D_leaf
                self.DTime += time.time() - tic2

                # Sentinel so Dmats is parallel to Umats/Vmats (see docstring)
                self.Dmats += [None]
                self.Umats += [U_ell]
                self.Vmats += [V_ell]

            else:
                # ---- coarsen from finer level -----------------------------
                if lvl == self.L - 2:
                    # Coming from the leaf: use sparse peel
                    peel_D_leaf(Y_ell, Z_ell, Om_ell, Psi_ell,
                                D_leaf, Nb, nl, self.adj_leaf_list)
                else:
                    # Coming from a coarser interior level: use block_mult
                    Y_ell -= block_mult(D_ell, Om_ell, Nb)
                    Z_ell -= block_mult(D_ell, Psi_ell, Nb, mode='T')

                Y_ell   = block_mult(U_ell, Y_ell,   Nb, mode='T')
                Z_ell   = block_mult(V_ell, Z_ell,   Nb, mode='T')
                Om_ell  = block_mult(V_ell, Om_ell,  Nb, mode='T')
                Psi_ell = block_mult(U_ell, Psi_ell, Nb, mode='T')

                Nb  = Nb // self.fac
                rkm = min(rk, nl * (self.fac ** (self.L - 1 - lvl)))

                if lvl > 0:
                    tic2  = time.time()
                    U_ell = compute_UV(Om_ell,  Y_ell, rkm, Nb)
                    V_ell = compute_UV(Psi_ell, Z_ell, rkm, Nb)
                    self.nullTime += time.time() - tic2

                    tic2  = time.time()
                    D_ell = construct_D(U_ell, V_ell, Y_ell, Z_ell,
                                        Om_ell, Psi_ell, Nb)
                    self.DTime += time.time() - tic2

                    self.Dmats += [D_ell]
                    self.Umats += [U_ell]
                    self.Vmats += [V_ell]

                else:
                    tic2  = time.time()
                    D_ell = block_solve_r(Y_ell, Om_ell, Nb)
                    self.blockSolveTime += time.time() - tic2
                    self.Dmats += [D_ell]

            self.Nbvec += [Nb]

        self.tCompress = time.time() - tic_compress

    # ------------------------------------------------------------------
    # matmat / rmatmat — identical to HBSMAT except lvl=0 uses D_leaf_mult
    # ------------------------------------------------------------------

    def matmat(self, v):
        """Forward matvec: u = H v."""
        if v.ndim == 1:
            vperm = v[self.perm, np.newaxis]
        else:
            vperm = v[self.perm, :]

        VV = [vperm]
        Nb = self.Nb

        # Upward: compress through V (same as HBSMAT)
        for lvl in range(len(self.Vmats)):
            VV += [block_mult(self.Vmats[lvl], VV[lvl], Nb, mode='T')]
            Nb  = Nb // self.fac

        # Root solve
        uperm = block_mult(self.Dmats[-1], VV[-1], Nb)

        # Downward: same loop as HBSMAT; at lvl=0 dispatch to D_leaf_mult
        for lvl in range(len(self.Umats) - 1, -1, -1):
            if lvl == 0:
                uperm = (block_mult(self.Umats[0], uperm, self.fac * Nb)
                       + D_leaf_mult(self.Dleaf, VV[0], self.Nb, self.nl,
                                     self.adj_leaf_list, mode='N'))
            else:
                uperm = (block_mult(self.Umats[lvl], uperm, self.fac * Nb)
                       + block_mult(self.Dmats[lvl], VV[lvl], self.fac * Nb))
            Nb = Nb * self.fac

        u = np.zeros(shape=uperm.shape)
        u[self.perm, :] = uperm
        if v.ndim == 1:
            u = u.flatten()
        return u

    def rmatmat(self, v):
        """Transpose matvec: u = H^T v."""
        if v.ndim == 1:
            vperm = v[self.perm, np.newaxis]
        else:
            vperm = v[self.perm, :]

        VV = [vperm]
        Nb = self.Nb

        # Upward: compress through U^T
        for lvl in range(len(self.Umats)):
            VV += [block_mult(self.Umats[lvl], VV[lvl], Nb, mode='T')]
            Nb  = Nb // self.fac

        # Root solve (transposed)
        uperm = block_mult(self.Dmats[-1], VV[-1], Nb, mode='T')

        # Downward
        for lvl in range(len(self.Vmats) - 1, -1, -1):
            if lvl == 0:
                uperm = (block_mult(self.Vmats[0], uperm, self.fac * Nb)
                       + D_leaf_mult(self.Dleaf, VV[0], self.Nb, self.nl,
                                     self.adj_leaf_list, mode='T'))
            else:
                uperm = (block_mult(self.Vmats[lvl], uperm, self.fac * Nb)
                       + block_mult(self.Dmats[lvl], VV[lvl], self.fac * Nb,
                                    mode='T'))
            Nb = Nb * self.fac

        u = np.zeros(shape=uperm.shape)
        u[self.perm, :] = uperm
        if v.ndim == 1:
            u = u.flatten()
        return u

    def matvec(self, v):
        return self.matmat(v)

    def rmatvec(self, v):
        return self.rmatmat(v)

    def __matmul__(self, v):
        if self.mode == 'N':
            return self.matmat(v)
        elif self.mode == 'T':
            return self.rmatmat(v)
        else:
            raise ValueError("mode not recognized")

    def __repr__(self):
        return (f"HBSMAT_STRONG(N={self.shape[0]}, L={self.L}, "
                f"Nb={self.Nb}, nl={self.nl})")