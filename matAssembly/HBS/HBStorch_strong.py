"""
HBStorch_strong.py  (consolidated, tagging removed)
=====================================================================
Strong-admissibility HBS *compression* from black-box matvecs.

Adjacency convention:
  * The neighbour structure supplied EXCLUDES self.  A tree's
    level_adj_list[lvl][t] lists boxes adjacent to t at level lvl, NOT t.
  * Internally the compressor always uses the NEAR band
        near(lvl, t) = [t] + neighbours(lvl, t),
    adding the diagonal block back wherever the near band is used.  The
    diagonal block is always inadmissible (stored densely and corrected),
    just not carried in the adjacency list.

The near band appears in four places, all self-inclusive via _near():
  1. far-field basis : nullify stacked near rows (diagonal + neighbours).
  2. near recovery   : D[t][s] = A_ts - U_t U_t^* A_ts V_s V_s^*, all s in near(t).
  3. reduction       : subtract sum_{s in near(t)} D[t][s] Om_s (diagonal included).
  4. apply (upward)  : add sum_{s in near(t)} D[t][s] g_s (diagonal included).

Adjacency assumed SYMMETRIC (checked).  Compression only.  Variable neighbour
degree supported via per-box loops.

Operator interface: A may be a scipy LinearOperator (uses matmat / rmatmat) or
any object exposing matmat/rmatmat, matvec/rmatvec, or @ and conj().T.  No .mT
is required.  Torch<->numpy and host<->device transfers are handled at the two
sketch calls only (scipy operators run on host numpy).

Inputs:
    HBSStrong(A, tree=T)                          # T gives adjacency, perm, sizes
    HBSStrong(A, Nb=, L=, neighbors=fn|list|None) # explicit (self-excluded);
                                                  # None -> 1D rule |i-j|<=1
=====================================================================
"""
import numpy as np
import torch
import torch.linalg as tla
import time

torch.set_default_dtype(torch.float64)


def neighbors_1d_excl(i, nb):
    return [j for j in (i - 1, i + 1) if 0 <= j < nb]


class HBSStrong:
    def __init__(self, A, Nb=None, L=None, device='cpu',
                 tree=None, perm=None, neighbors=None):
        self.A = A
        self.N = A.shape[0]
        self.device = device
        self.fac = 2
        self._tree = tree

        if tree is not None:
            Nb = tree.nleaves
            L = tree.nlevels - 1
            perm = np.asarray(tree.perm_leaf)
            self._adj = [list(map(list, tree.level_adj_list[lvl]))
                         for lvl in range(tree.nlevels)]
            self._nbr_mode = 'list'
        elif neighbors is None:
            self._adj = None; self._nbr_mode = '1d'
        elif callable(neighbors):
            self._adj = neighbors; self._nbr_mode = 'callable'
        else:
            self._adj = neighbors; self._nbr_mode = 'list'

        self.Nb = Nb; self.nl = self.N // Nb; self.L = L
        self.perm = (torch.arange(self.N, dtype=torch.int64) if perm is None
                     else torch.as_tensor(perm, dtype=torch.int64))
        self.U = []; self.V = []; self.D = []; self.clevels = []
        self.Aroot = None; self._child_ranks_cache = []
        self._near_cache = None; self._invperm_cache = None

    # -- neighbour access (self-EXCLUDED) ----------------------------------
    def _nbrs(self, lvl, t):
        if self._nbr_mode == '1d':
            return neighbors_1d_excl(t, self._nb_at(lvl))
        if self._nbr_mode == 'callable':
            return [j for j in self._adj(lvl, t) if j != t]
        return [j for j in self._adj[lvl][t] if j != t]

    def _near(self, lvl, t):
        if self._near_cache is None:
            self._near_cache = {}
        key = (lvl, t)
        v = self._near_cache.get(key)
        if v is None:
            v = [t] + self._nbrs(lvl, t)
            self._near_cache[key] = v
        return v

    def _invperm(self):
        if self._invperm_cache is None:
            ip = torch.empty_like(self.perm)
            ip[self.perm] = torch.arange(self.N, device=self.device)
            self._invperm_cache = ip
        return self._invperm_cache

    def _nb_at(self, lvl):
        if self._tree is not None:
            return len(self._tree.get_boxes_level(lvl))
        return 2 ** lvl

    def _has_far(self, lvl):
        nb = self._nb_at(lvl)
        return any(len(self._nbrs(lvl, t)) < nb - 1 for t in range(nb))

    def _check_symmetry(self, lvl):
        nb = self._nb_at(lvl)
        for t in range(nb):
            for s in self._nbrs(lvl, t):
                if t not in self._nbrs(lvl, s):
                    raise ValueError(f"adjacency not symmetric at level {lvl}: "
                                     f"{s} in nbrs({t}) but {t} not in nbrs({s})")

    def _child_ranks(self, lvl):
        if self._tree is not None:
            T = self._tree
            boxes_l = T.get_boxes_level(lvl)
            rank_of = {g: r for r, g in enumerate(boxes_l)}
            out = []
            for g in T.get_boxes_level(lvl - 1):
                ch = [c.index for c in T.get_node(g).children]
                out.append([rank_of[c] for c in ch])
            return out
        nb = self._nb_at(lvl)
        return [[2 * P, 2 * P + 1] for P in range(nb // self.fac)]

    # -- operator application (scipy LinearOperator: matmat / rmatmat) -----
    def _apply_A(self, Xt, adjoint=False):
        """Apply A (or A^*) to a torch matrix Xt.  Handles the torch<->numpy and
        host<->device boundary: scipy LinearOperators require host numpy input.
        No .mT used."""
        Xnp = Xt.detach().cpu().numpy()
        Op = self.A
        if adjoint:
            if hasattr(Op, 'rmatmat'):
                Ynp = Op.rmatmat(Xnp)
            elif hasattr(Op, 'rmatvec'):
                Ynp = np.column_stack([Op.rmatvec(Xnp[:, j]) for j in range(Xnp.shape[1])])
            else:
                Ynp = (Op.conj().T) @ Xnp
        else:
            if hasattr(Op, 'matmat'):
                Ynp = Op.matmat(Xnp)
            elif hasattr(Op, 'matvec'):
                Ynp = np.column_stack([Op.matvec(Xnp[:, j]) for j in range(Xnp.shape[1])])
            else:
                Ynp = Op @ Xnp
        return torch.as_tensor(np.asarray(Ynp), dtype=Xt.dtype, device=Xt.device)

    # -- core linear algebra ----------------------------------------------
    @staticmethod
    def _far_basis_and_factor(Yt, On, rk):
        """Far-field basis of one box AND the reusable factorization of its near
        band On.  One complete QR of On^T serves both: trailing columns give
        null(On) for the basis; the (Q1, R1) leading part gives the
        least-squares solve against On for the near-band recovery (no pinv)."""
        gb = On.shape[0]
        QR = tla.qr(On.mT, mode='complete')        # On^T = Q R, Q:(s,s)
        Q = QR.Q; R = QR.R
        Pnull = Q[:, gb:]                          # null(On): (s, s-gb)
        M = Yt @ Pnull
        rr = min(rk, M.shape[1], M.shape[0])
        U = tla.svd(M, full_matrices=False).U[:, :rr]
        Q1 = Q[:, :gb]                             # (s, gb)
        R1 = R[:gb, :gb]                           # (gb, gb) upper-tri
        return U, (Q1, R1, gb)

    @staticmethod
    def _solve_band(Wt, fac):
        """Least-squares X with X @ On = Wt, using On = R1^T Q1^T (from the QR of
        On^T).  pinv(On) acts as Q1 R1^{-T}; robust triangular solve."""
        Q1, R1, gb = fac
        tmp = Wt @ Q1                              # (b, gb)
        # X = tmp @ R1^{-T}: solve X R1^T = tmp
        X = torch.linalg.solve_triangular(R1.mT, tmp, upper=False, left=False)
        return X                                   # (b, gb)

    def _recover_band(self, Yb, Zb, U, V, facU, facV, lvl):
        nb = len(Yb); b = Yb[0].shape[0]
        T1 = []
        for t in range(nb):
            near = self._near(lvl, t)
            Yperp = Yb[t] - U[t] @ (U[t].mT @ Yb[t])
            T1.append((self._solve_band(Yperp, facU[t]), near))
        T2 = []
        for sgn in range(nb):
            near = self._near(lvl, sgn)
            Zperp = Zb[sgn] - V[sgn] @ (V[sgn].mT @ Zb[sgn])
            T2.append((self._solve_band(Zperp, facV[sgn]), near))
        D = {t: {} for t in range(nb)}
        for t in range(nb):
            Ut = U[t]
            t1mat, near_t = T1[t]
            for jpos, s in enumerate(near_t):
                t1 = t1mat[:, jpos * b:(jpos + 1) * b]
                t2mat, near_s = T2[s]
                ipos = near_s.index(t)
                t2 = t2mat[:, ipos * b:(ipos + 1) * b].mT
                # D = (I-PU) t1 + PU t2 ; apply PU as U(U^* x), never form UU
                Pu_t1 = Ut @ (Ut.mT @ t1)
                Pu_t2 = Ut @ (Ut.mT @ t2)
                D[t][s] = (t1 - Pu_t1) + Pu_t2
        return D

    # -- construction ------------------------------------------------------
    def construct(self, rk, p=10):
        nl = self.nl
        if self._nbr_mode == '1d' or self._adj is None:
            maxdeg = 3
        else:
            maxdeg = max((max((len(self._near(lvl, t))
                               for t in range(self._nb_at(lvl))), default=1)
                          for lvl in range(2, self.L + 1)), default=1)
        s = min(maxdeg * max(nl, self.fac * rk) + rk + p, self.N)
        self.nSamples = s
        tic = time.time()
        Om = torch.randn(self.N, s, device=self.device)
        Psi = torch.randn(self.N, s, device=self.device)
        # sketch without allocating separate permuted full-size copies:
        # A acts in native order; scatter box-order samples into native order
        # by index, apply, then gather rows back to box order.
        Y = self._apply_A(Om[self._invperm(), :], adjoint=False)[self.perm, :]
        Z = self._apply_A(Psi[self._invperm(), :], adjoint=True)[self.perm, :]
        print("sample done in : ",time.time()-tic,"s")
        def split(M, nb):
            bb = M.shape[0] // nb
            return [M[i * bb:(i + 1) * bb, :] for i in range(nb)]

        lvl = self.L; nb = self._nb_at(lvl)
        Yb = split(Y, nb); Zb = split(Z, nb)
        Omb = split(Om, nb); Psib = split(Psi, nb)

        while True:
            nb = len(Yb)
            if not self._has_far(lvl):
                self.Aroot = torch.cat(Yb, dim=0) @ tla.pinv(torch.cat(Omb, dim=0))
                self.root_nb = nb; self.root_b = Yb[0].shape[0]
                break
            self._check_symmetry(lvl)

            U = []; V = []; facU = []; facV = []
            for t in range(nb):
                near = self._near(lvl, t)
                On = torch.cat([Omb[j] for j in near], dim=0)
                Pn = torch.cat([Psib[j] for j in near], dim=0)
                Ut, fu = self._far_basis_and_factor(Yb[t], On, rk)
                Vt, fv = self._far_basis_and_factor(Zb[t], Pn, rk)
                U.append(Ut); V.append(Vt); facU.append(fu); facV.append(fv)

            D = self._recover_band(Yb, Zb, U, V, facU, facV, lvl)
            self.U.append(U); self.V.append(V); self.D.append(D)
            self.clevels.append(lvl)

            r0 = U[0].shape[1]
            assert all(U[t].shape[1] == r0 for t in range(nb)), \
                f"non-uniform rank at level {lvl}"
            tY = [None]*nb; tZ=[None]*nb; tOm=[None]*nb; tPsi=[None]*nb
            for t in range(nb):
                near = self._near(lvl, t)
                Ynear = sum((D[t][s] @ Omb[s] for s in near), torch.zeros_like(Yb[t]))
                Znear = sum((D[s][t].mT @ Psib[s] for s in near), torch.zeros_like(Zb[t]))
                tY[t] = U[t].mT @ (Yb[t] - Ynear)
                tZ[t] = V[t].mT @ (Zb[t] - Znear)
                tOm[t] = V[t].mT @ Omb[t]
                tPsi[t] = U[t].mT @ Psib[t]

            cr = self._child_ranks(lvl); self._child_ranks_cache.append(cr)
            Yb = [torch.cat([tY[c] for c in ch], dim=0) for ch in cr]
            Zb = [torch.cat([tZ[c] for c in ch], dim=0) for ch in cr]
            Omb = [torch.cat([tOm[c] for c in ch], dim=0) for ch in cr]
            Psib = [torch.cat([tPsi[c] for c in ch], dim=0) for ch in cr]
            lvl -= 1
        return self

    # -- apply -------------------------------------------------------------
    def _apply(self, v, transpose=False):
        was_np = isinstance(v, np.ndarray)
        vt = (torch.as_tensor(v, dtype=torch.float64, device=self.device)
              if was_np else v.to(self.device))
        if vt.ndim == 1:
            vt = vt[:, None]; squeeze = True
        else:
            squeeze = False
        vp = vt[self.perm, :]
        Bdown = self.V if not transpose else self.U
        Bup = self.U if not transpose else self.V

        def split(M, nb):
            bb = M.shape[0] // nb
            return [M[i * bb:(i + 1) * bb, :] for i in range(nb)]

        g = [split(vp, self.Nb)]
        for i, lvl in enumerate(self.clevels):
            Bd = Bdown[i]; nb = len(g[i])
            proj = [Bd[t].mT @ g[i][t] for t in range(nb)]
            cr = self._child_ranks_cache[i]
            g.append([torch.cat([proj[c] for c in ch], dim=0) for ch in cr])

        groot = torch.cat(g[-1], dim=0)
        Aroot = self.Aroot if not transpose else self.Aroot.mT
        wroot = Aroot @ groot
        b = self.root_b
        w_parent = [wroot[i*b:(i+1)*b, :] for i in range(self.root_nb)]

        for i in reversed(range(len(self.clevels))):
            lvl = self.clevels[i]
            Bu = Bup[i]; D = self.D[i]; nb = len(g[i])
            cr = self._child_ranks_cache[i]
            wnew = [None]*nb
            for pidx, ch in enumerate(cr):
                off = 0
                for c in ch:
                    rc = Bu[c].shape[1]
                    wpar_c = w_parent[pidx][off:off+rc, :]
                    far = Bu[c] @ wpar_c
                    near = sum(((D[c][s] if not transpose else D[s][c].mT) @ g[i][s]
                                for s in self._near(lvl, c)),
                               torch.zeros_like(far))
                    wnew[c] = far + near
                    off += rc
            w_parent = wnew

        u = torch.cat(w_parent, dim=0)
        out = torch.zeros_like(u); out[self.perm, :] = u
        if squeeze:
            out = out[:, 0]
        if was_np:
            return out.detach().cpu().numpy()
        return out

    def matvec(self, v): return self._apply(v, False)
    def rmatvec(self, v): return self._apply(v, True)
    def __matmul__(self, v): return self._apply(v, False)


def _build_exact_hbs_1d(L, m, r, seed=0):
    rng = np.random.default_rng(seed)
    nleaf = 2 ** L; N = nleaf * m
    nbr = lambda i, j: abs(i - j) <= 1
    def admissible(P, Q, lvl):
        return lvl != 0 and (not nbr(P, Q)) and nbr(P >> 1, Q >> 1)
    def leaf_range(box, lvl):
        span = 2 ** (L - lvl); a0 = box * span
        return a0 * m, (a0 + span) * m
    orth = lambda a, b: np.linalg.qr(rng.standard_normal((a, b)))[0]
    Uf = {}; Vf = {}
    for lvl in range(L, 0, -1):
        for box in range(2 ** lvl):
            sz = m if lvl == L else 2 * r
            Uf[(lvl, box)] = orth(sz, r); Vf[(lvl, box)] = orth(sz, r)
    hatU = {}; hatV = {}
    for box in range(2 ** L):
        hatU[(L, box)] = Uf[(L, box)]; hatV[(L, box)] = Vf[(L, box)]
    for lvl in range(L - 1, 0, -1):
        for box in range(2 ** lvl):
            c1, c2 = 2*box, 2*box+1
            h1, h2 = hatU[(lvl+1, c1)], hatU[(lvl+1, c2)]
            bU = np.zeros((h1.shape[0]+h2.shape[0], 2*r))
            bU[:h1.shape[0], :r] = h1; bU[h1.shape[0]:, r:] = h2
            hatU[(lvl, box)] = bU @ Uf[(lvl, box)]
            k1, k2 = hatV[(lvl+1, c1)], hatV[(lvl+1, c2)]
            bV = np.zeros((k1.shape[0]+k2.shape[0], 2*r))
            bV[:k1.shape[0], :r] = k1; bV[k1.shape[0]:, r:] = k2
            hatV[(lvl, box)] = bV @ Vf[(lvl, box)]
    A = np.zeros((N, N))
    for lvl in range(1, L + 1):
        for P in range(2 ** lvl):
            for Q in range(2 ** lvl):
                if admissible(P, Q, lvl):
                    S = rng.standard_normal((r, r))
                    r0, r1 = leaf_range(P, lvl); c0, c1 = leaf_range(Q, lvl)
                    A[r0:r1, c0:c1] = hatU[(lvl, P)] @ S @ hatV[(lvl, Q)].T
    for a in range(nleaf):
        for b in range(nleaf):
            if nbr(a, b):
                A[a*m:(a+1)*m, b*m:(b+1)*m] = rng.standard_normal((m, m))
    return torch.from_numpy(A)


if __name__ == "__main__":
    from scipy.sparse.linalg import aslinearoperator
    print("regression: 1D exact-HBS via scipy LinearOperator (no .mT)")
    A = _build_exact_hbs_1d(3, 100, 5, seed=0); N = A.shape[0]
    LinOp = aslinearoperator(A.numpy())
    H = HBSStrong(LinOp, Nb=8, L=3).construct(rk=5, p=10)
    relF = (H @ torch.eye(N) - A).norm() / torch.linalg.matrix_norm(A, ord='fro')
    print(f"  full ||A_h - A||_F / ||A||_F = {relF:.3e}  "
          f"({'machine precision' if relF < 1e-10 else 'FAIL'})")