"""
HBStorch_strong.py  (consolidated, batched QR)
=====================================================================
Strong-admissibility HBS *compression* from black-box matvecs.

Adjacency: supplied EXCLUDING self; internally the near band is self-inclusive
near(lvl,t) = [t] + nbrs(lvl,t).  The diagonal block is inadmissible (stored
densely, corrected), just not carried in the adjacency list.  Symmetric (checked).

Batching:  at each level every box's near band is padded to a uniform degree
g_max with ZERO blocks (and a 0/1 mask), so the near-band QR factorizations of
all boxes run as a SINGLE batched torch.linalg.qr.  The far-field bases come out
as one tensor U[i], V[i] of shape (nb, b, rk) per level; the near corrections as
one tensor D[i] of shape (nb, g_max, b, b).  No s x s factor is ever formed
(reduced QR + economy SVD), so peak memory is nb * s_lvl * g_max*b.  An optional
chunk= bounds that by processing boxes in groups.

Zero-padded near rows make On rank-deficient; the recovery solve therefore uses
a min-norm pseudo-inverse, computed with batched SVD so it runs on CPU AND GPU
(torch.linalg.lstsq's rank-revealing drivers are CPU-only):
    fast=False : Wt @ tla.pinv(On)          (forms the pinv matrix, then applies)
    fast=True  : SVD-solve applied to the RHS directly (no (s x gb) pinv matrix;
                 same SVD, fewer flops / less memory in the apply).
Both give the min-norm solution on the padded system.

Operator A: scipy LinearOperator (matmat/rmatmat) or @/conj().T; no .mT on A.
Internal storage (U,V,D,index/mask) is private; matvec/rmatvec/@ + constructor
are the public surface.

Inputs:
    HBSStrong(A, tree=T)
    HBSStrong(A, Nb=, L=, neighbors=fn|list|None)   # None -> 1D |i-j|<=1
=====================================================================
"""
import numpy as np
import torch
import torch.linalg as tla

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
                     else torch.as_tensor(perm, dtype=torch.int64)).to(self.device)
        self.U = []; self.V = []; self.D = []; self.clevels = []
        self.Aroot = None
        self._cidx = []          # per clevel: (npar, fac) child box ranks
        self._nidx = []          # per clevel: (nb, g_max) near source idx
        self._nmsk = []          # per clevel: (nb, g_max) near mask
        self._nbpos = []         # per clevel: (nb, g_max) back-position
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
        v = self._near_cache.get((lvl, t))
        if v is None:
            v = [t] + self._nbrs(lvl, t)
            self._near_cache[(lvl, t)] = v
        return v

    def _invperm(self):
        if self._invperm_cache is None:
            ip = torch.empty_like(self.perm)
            ip[self.perm] = torch.arange(self.N, device=self.perm.device)
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
                    raise ValueError(f"adjacency not symmetric at level {lvl}")

    def _build_near_index(self, lvl):
        """(idx, msk, backpos) for the self-inclusive near band padded to g_max.
        idx (nb,g) source box per slot (pad->self); msk (nb,g) 1/0; backpos
        (nb,g) position of t within source s's near list (adjoint gather)."""
        nb = self._nb_at(lvl)
        near = [self._near(lvl, t) for t in range(nb)]
        g = max(len(n) for n in near)
        idx = torch.zeros(nb, g, dtype=torch.int64)
        msk = torch.zeros(nb, g)
        slot = [dict() for _ in range(nb)]
        for t in range(nb):
            for j, s in enumerate(near[t]):
                idx[t, j] = s; msk[t, j] = 1.0
                slot[t].setdefault(s, j)
            for j in range(len(near[t]), g):
                idx[t, j] = t
        backpos = torch.zeros(nb, g, dtype=torch.int64)
        for t in range(nb):
            for j in range(g):
                if msk[t, j]:
                    s = int(idx[t, j]); backpos[t, j] = slot[s].get(t, 0)
        return (idx.to(self.device), msk.to(self.device), backpos.to(self.device))

    def _child_ranks(self, lvl):
        if self._tree is not None:
            T = self._tree
            rank_of = {gg: r for r, gg in enumerate(T.get_boxes_level(lvl))}
            return [[rank_of[c.index] for c in T.get_node(gg).children]
                    for gg in T.get_boxes_level(lvl - 1)]
        nb = self._nb_at(lvl)
        return [[2 * P, 2 * P + 1] for P in range(nb // self.fac)]

    # -- operator application (scipy LinearOperator) -----------------------
    def _apply_A(self, Xt, adjoint=False):
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

    # -- batched gather of the near band -----------------------------------
    @staticmethod
    def _gather(blocks, idx, msk):
        """blocks (nb,b,c) -> (nb,g,b,c) gathered over near idx, pad slots zeroed.
        The advanced index already returns a fresh copy, so the mask is applied
        in place to avoid allocating a second full-size tensor."""
        nb, b, c = blocks.shape
        g = idx.shape[1]
        out = blocks[idx.reshape(-1)].reshape(nb, g, b, c)
        out.mul_(msk.reshape(nb, g, 1, 1))
        return out

    @staticmethod
    def _auto_chunk(chunk, nb, per_box_elems, dev, frac=0.2):
        """Boxes-per-tile for the batched factorizations.  An explicit positive
        `chunk` always wins.  Otherwise, on CUDA, pick the largest tile whose
        working set (per_box_elems fp64 numbers per box) fits in `frac` of the
        currently-free memory; on CPU or if the query fails, do not tile."""
        if chunk is not None and chunk > 0:
            return min(int(chunk), nb)
        if getattr(dev, 'type', 'cpu') == 'cuda':
            try:
                free, _ = torch.cuda.mem_get_info(dev)
                ch = int(frac * free) // max(8 * int(per_box_elems), 1)
                return max(1, min(ch, nb))
            except Exception:
                return nb
        return nb

    @staticmethod
    def _band_apply(D, src, idx, chunk=None):
        """Contract a near-band operator with gathered sources WITHOUT ever
        materializing the (nb,g,b,*) gather:
            out[t] = sum_j D[t,j] @ src[idx[t,j]]            -> (nb,b,s)
        Equivalent to einsum('tgbc,tgcs->tbs', D, gather(src)).  D (nb,g,b,b)
        must already be masked (padded slots zero); src (nb,b,s).  Peak working
        set is (nb,b,s) per slot instead of (nb,g,b,s); `chunk` further tiles
        the box axis so the per-tile transient is (chunk,b,s)."""
        nb, g, b, _ = D.shape
        out = torch.zeros(nb, b, src.shape[2], device=src.device, dtype=src.dtype)
        ch = nb if (chunk is None or chunk <= 0) else int(chunk)
        for a in range(0, nb, ch):
            sl = slice(a, a + ch)
            for j in range(g):
                out[sl] += torch.bmm(D[sl, j], src[idx[sl, j]])
        return out

    @staticmethod
    def _gather_pos(T, idx, backpos):
        """T (nb,b,g,b) -> (nb,g,b,b): out[t,j] = T[s,:,pos,:], s=idx[t,j]."""
        nb, b, g, _ = T.shape
        out = T[idx.reshape(-1)].reshape(nb, g, b, g, b)
        pj = backpos.reshape(nb, g, 1, 1, 1).expand(nb, g, b, 1, b)
        return torch.gather(out, 3, pj).squeeze(3)

    def _transpose_band(self, D, idx, backpos, msk):
        """Dt (nb,g,b,b): Dt[t,j] = D[s][t]^T, s=idx[t,j] (adjoint near block)."""
        nb, g, b, _ = D.shape
        Dg = D[idx.reshape(-1)].reshape(nb, g, g, b, b)
        pj = backpos.reshape(nb, g, 1, 1, 1).expand(nb, g, 1, b, b)
        Dg = torch.gather(Dg, 2, pj).squeeze(2)
        return (Dg.transpose(-1, -2) * msk.reshape(nb, g, 1, 1)).contiguous()

    # -- batched basis + recovery for one level ----------------------------
    def _factor_level(self, Yb, Zb, Omb, Psib, idx, msk, backpos,
                      rk, s_lvl, fast, chunk):
        """Yb,Zb,Omb,Psib: (nb,b,s).  Returns U,V (nb,b,rk), D (nb,g,b,b).

        Far-field basis: padded near band, batched reduced-QR + economy SVD.
        Recovery solve (T = perp @ pinv(near)):
          fast=False : SVD pseudo-inverse on the (zero-padded) near band.
          fast=True  : boxes are bucketed by their TRUE near-degree; each bucket
                       has a uniform full-rank near band (no padding), solved by
                       a single batched reduced-QR + triangular solve (no SVD).
                       This removes the zero-padding rank deficiency that forced
                       the SVD, and is ~4x faster on the dominant recovery cost.
        """
        nb, b, _ = Yb.shape
        g = idx.shape[1]
        dev = Yb.device
        # a b-row block has rank <= b, so the far-field basis has at most b
        # columns; cap the requested rank accordingly (uniform b per level).
        reff = min(rk, b)
        Yc = Yb[:, :, :s_lvl]; Zc = Zb[:, :, :s_lvl]
        Oc = Omb[:, :, :s_lvl]; Pc = Psib[:, :, :s_lvl]
        if not fast:
            On = self._gather(Oc, idx, msk).reshape(nb, g * b, s_lvl)
            Pn = self._gather(Pc, idx, msk).reshape(nb, g * b, s_lvl)

        # boxes-per-tile: worst-case per-box working set in the batched QR is
        # ~5 copies of the near band / Q factor (s_lvl x g*b).  Sized to free mem
        # when chunk is None so the default does not OOM on large levels.
        ch = self._auto_chunk(chunk, nb, 5 * s_lvl * g * b, dev)

        # true near-degree per box (number of real, unmasked slots)
        deg = msk.sum(dim=1).round().to(torch.int64)            # (nb,)

        def far_basis_padded(NN, samp):
            """fast=False basis: padded near band, reduced QR for the null-space
            projector, economy SVD for the leading-rk basis.  Chunked."""
            U_parts = []
            for a in range(0, nb, ch):
                NNc = NN[a:a+ch]; sc = samp[a:a+ch]
                Q1 = tla.qr(NNc.transpose(1, 2), mode='reduced').Q
                M = sc - torch.bmm(torch.bmm(sc, Q1), Q1.transpose(1, 2))
                Uc = tla.svd(M, full_matrices=False).U[:, :, :reff].contiguous()
                U_parts.append(Uc)
            return torch.cat(U_parts, 0)

        def recover_svd(NN, samp, U):
            """T = perp @ pinv(NN) via SVD (handles zero-padded rank deficiency).
            Returns (nb, b, g*b).  fast=False path."""
            out = torch.empty(nb, b, g * b, device=dev, dtype=NN.dtype)
            for a in range(0, nb, ch):
                NNc = NN[a:a+ch]; sc = samp[a:a+ch]; Uc = U[a:a+ch]
                perp = sc - torch.bmm(Uc, torch.bmm(Uc.transpose(1, 2), sc))
                out[a:a+ch] = torch.bmm(perp, tla.pinv(NNc))
            return out

        def basis_and_recover_bucketed(src, samp):
            """fast=True: bucket boxes by TRUE near-degree.  ONE reduced QR of
            the unpadded near band per bucket serves BOTH:
              * the far-field basis  (null-space projector I - Q1 Q1^*),
              * the recovery solve   (T = perp @ pinv(near) = (perp Q1) R1^{-T}).
            src: (nb,b,s) source blocks (Oc/Pc); samp: (nb,b,s) samples (Yc/Zc).
            The near band is gathered PER BUCKET straight from src using the
            first d (real) slots of idx -- no padded (nb,g,b,s) tensor and no
            mask, since a degree-d box has exactly d real slots at [:d].
            The leading-reff left basis is taken from an eigh of the b x b Gram
            M M^* (b is small) rather than an SVD over the long sample axis.
            Returns U (nb,b,reff) and T (nb,b,g*b) (pad cols zero)."""
            U = torch.empty(nb, b, reff, device=dev, dtype=samp.dtype)
            T = torch.zeros(nb, b, g * b, device=dev, dtype=samp.dtype)
            for d in torch.unique(deg).tolist():
                sel_all = (deg == d).nonzero(as_tuple=True)[0]
                if sel_all.numel() == 0:
                    continue
                for a in range(0, sel_all.numel(), ch):
                    sel = sel_all[a:a + ch]
                    nd = sel.numel()
                    # gather only the d real neighbour source blocks (no pad/mask)
                    src_idx = idx[sel, :d].reshape(-1)          # (nd*d,)
                    NNb = src[src_idx].reshape(nd, d * b, s_lvl)
                    sc = samp[sel]                              # (nd, b, s)
                    # single reduced QR of the unpadded near band^T: (s, d*b)
                    QR = tla.qr(NNb.transpose(1, 2), mode='reduced')
                    Q1, R1 = QR.Q, QR.R                         # Q1:(nd,s,d*b)
                    # far-field basis: project near row space out of the sample
                    M = sc - torch.bmm(torch.bmm(sc, Q1), Q1.transpose(1, 2))
                    # leading reff left singular vectors of M == top-reff eigen-
                    # vectors of the b x b SPD Gram M M^* (eigh ascending: flip).
                    G = torch.bmm(M, M.transpose(1, 2))         # (nd,b,b)
                    evecs = torch.linalg.eigh(G).eigenvectors
                    U[sel] = evecs[:, :, -reff:].flip(-1).contiguous()
                    # recovery: perp wrt the just-found basis, triangular solve
                    Uc = U[sel]
                    perp = sc - torch.bmm(Uc, torch.bmm(Uc.transpose(1, 2), sc))
                    tmp = torch.bmm(perp, Q1)                    # (nd,b,d*b)
                    Td = torch.linalg.solve_triangular(
                        R1.transpose(1, 2), tmp, upper=False, left=False)
                    T[sel, :, :d * b] = Td
            return U, T

        if fast:
            # near band is gathered per bucket inside the routine straight from
            # Oc/Pc; no (nb,g,b,s) tensor is formed on the fast path.
            U, T1 = basis_and_recover_bucketed(Oc, Yc)
            V, T2 = basis_and_recover_bucketed(Pc, Zc)
        else:
            U = far_basis_padded(On, Yc)
            V = far_basis_padded(Pn, Zc)
            T1 = recover_svd(On, Yc, U)
            T2 = recover_svd(Pn, Zc, V)

        T1 = T1.reshape(nb, b, g, b).permute(0, 2, 1, 3).contiguous()   # (nb,g,b,b)
        T2 = T2.reshape(nb, b, g, b)                                    # (nb,b,g,b)

        # term1 = (I - U U^*) T1 ; term2 = U U^* (adjoint block of T2)^T
        PU_T1 = torch.einsum('tbr,tgrc->tgbc', U,
                             torch.einsum('trb,tgbc->tgrc', U.transpose(1, 2), T1))
        term1 = T1 - PU_T1
        nbr_T2 = self._gather_pos(T2, idx, backpos).transpose(-1, -2)
        term2 = torch.einsum('tbr,tgrc->tgbc', U,
                             torch.einsum('trb,tgbc->tgrc', U.transpose(1, 2), nbr_T2))
        D = (term1 + term2) * msk.reshape(nb, g, 1, 1)
        return U, V, D

    # -- construction ------------------------------------------------------
    def _check_uniform_boxes(self):
        """The batched path reshapes each level's stacked samples into
        (nb, b, s) with a single b = N/nb per level.  This requires every box
        at a level to hold the same number of DOFs.  Verify up front and raise
        a legible error otherwise (e.g. a ragged tree), instead of failing with
        a cryptic reshape mismatch deep in the level merge."""
        if self._tree is None:
            # explicit / 1D path: blocks are N/nb by construction, nothing ragged
            if self.N % self.Nb != 0:
                raise ValueError(
                    f"non-uniform boxes: N={self.N} is not divisible by "
                    f"Nb={self.Nb}; the batched compressor needs equal-size boxes.")
            return
        T = self._tree
        for lvl in range(T.nlevels):
            boxes = T.get_boxes_level(lvl)
            if not boxes:
                continue
            sizes = set()
            for bdex in boxes:
                node = T.get_node(bdex)
                if getattr(node, 'point_inds', None) is not None:
                    sizes.add(len(node.point_inds))
            if len(sizes) > 1:
                raise ValueError(
                    f"non-uniform boxes at tree level {lvl}: found box sizes "
                    f"{sorted(sizes)}.  The batched compressor requires all boxes "
                    f"at a level to have equal size (N/nb).  Fix the tree so "
                    f"leaves are uniform, or use the per-box build.")
        if self.N % self.Nb != 0:
            raise ValueError(
                f"non-uniform boxes: N={self.N} not divisible by nleaves="
                f"{self.Nb}; leaves are ragged.")

    def construct(self, rk, p=10, fast=False, chunk=None):
        self._check_uniform_boxes()
        nl = self.nl
        if self._nbr_mode == '1d' or self._adj is None:
            maxdeg = 3
        else:
            maxdeg = max((max((len(self._near(lvl, t))
                               for t in range(self._nb_at(lvl))), default=1)
                          for lvl in range(2, self.L + 1)), default=1)
        s = min(maxdeg * max(nl, self.fac * rk) + rk + p, self.N)
        self.nSamples = s

        Om = torch.randn(self.N, s, device=self.device)
        Psi = torch.randn(self.N, s, device=self.device)
        Y = self._apply_A(Om[self._invperm(), :], adjoint=False)[self.perm, :]
        Z = self._apply_A(Psi[self._invperm(), :], adjoint=True)[self.perm, :]

        def to_blocks(M, nb):
            b = M.shape[0] // nb
            return M.reshape(nb, b, M.shape[1]).contiguous()

        lvl = self.L; nb = self._nb_at(lvl)
        Yb = to_blocks(Y, nb); Zb = to_blocks(Z, nb)
        Omb = to_blocks(Om, nb); Psib = to_blocks(Psi, nb)

        while True:
            nb = Yb.shape[0]
            if not self._has_far(lvl):
                Yf = Yb.reshape(nb * Yb.shape[1], Yb.shape[2])
                Of = Omb.reshape(nb * Omb.shape[1], Omb.shape[2])
                self.Aroot = Yf @ tla.pinv(Of)
                self.root_nb = nb; self.root_b = Yb.shape[1]
                break
            self._check_symmetry(lvl)
            idx, msk, backpos = self._build_near_index(lvl)

            b_lvl = Yb.shape[1]; g = idx.shape[1]
            s_lvl = min(g * b_lvl + rk + p, Yb.shape[2])

            U, V, D = self._factor_level(Yb, Zb, Omb, Psib, idx, msk, backpos,
                                         rk, s_lvl, fast, chunk)
            self.U.append(U); self.V.append(V); self.D.append(D)
            self.clevels.append(lvl)
            self._nidx.append(idx); self._nmsk.append(msk); self._nbpos.append(backpos)

            # reduction (batched): subtract near band (diag incl), project, merge.
            # The near-band contraction is slot-by-slot (no (nb,g,b,s) gather) and
            # tiled over boxes; peak per tile is (chunk,b,s).
            red_ch = self._auto_chunk(chunk, nb, 3 * Yb.shape[2] * b_lvl, Yb.device)
            Ynear = self._band_apply(D, Omb, idx, chunk=red_ch)
            Dt = self._transpose_band(D, idx, backpos, msk)
            Znear = self._band_apply(Dt, Psib, idx, chunk=red_ch)
            tY = torch.bmm(U.transpose(1, 2), Yb - Ynear)
            tZ = torch.bmm(V.transpose(1, 2), Zb - Znear)
            tOm = torch.bmm(V.transpose(1, 2), Omb)
            tPsi = torch.bmm(U.transpose(1, 2), Psib)

            cr = self._child_ranks(lvl)
            ci = torch.tensor(cr, dtype=torch.int64, device=self.device)
            self._cidx.append(ci)
            r = tY.shape[1]; npar, fc = ci.shape

            def merge(t):
                gth = t[ci.reshape(-1)].reshape(npar, fc, r, t.shape[2])
                return gth.reshape(npar, fc * r, t.shape[2]).contiguous()
            Yb = merge(tY); Zb = merge(tZ); Omb = merge(tOm); Psib = merge(tPsi)
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

        nb0 = self.Nb; b0 = vp.shape[0] // nb0
        g = [vp.reshape(nb0, b0, vp.shape[1]).contiguous()]
        for i in range(len(self.clevels)):
            proj = torch.bmm(Bdown[i].transpose(1, 2), g[i])     # (nb,r,s)
            ci = self._cidx[i]; r = proj.shape[1]; npar, fc = ci.shape
            gth = proj[ci.reshape(-1)].reshape(npar, fc, r, proj.shape[2])
            g.append(gth.reshape(npar, fc * r, proj.shape[2]).contiguous())

        groot = g[-1].reshape(self.root_nb * self.root_b, g[-1].shape[2])
        Aroot = self.Aroot if not transpose else self.Aroot.transpose(0, 1)
        wroot = Aroot @ groot
        b = self.root_b
        w_par = wroot.reshape(self.root_nb, b, wroot.shape[1]).contiguous()

        for i in reversed(range(len(self.clevels))):
            Bu = Bup[i]; D = self.D[i]; ci = self._cidx[i]
            idx = self._nidx[i]; msk = self._nmsk[i]; backpos = self._nbpos[i]
            npar, fc = ci.shape; r = Bu.shape[2]; nb = g[i].shape[0]
            wp = w_par.reshape(npar, fc, r, w_par.shape[2])
            child = ci.reshape(-1)
            wp_flat = wp.reshape(npar * fc, r, w_par.shape[2])
            far = torch.bmm(Bu[child], wp_flat)                  # (nb,b,s) child order
            far_box = torch.empty(nb, Bu.shape[1], w_par.shape[2],
                                  device=self.device, dtype=far.dtype)
            far_box[child] = far
            Duse = D if not transpose else self._transpose_band(D, idx, backpos, msk)
            ap_ch = self._auto_chunk(None, nb, 3 * g[i].shape[2] * Bu.shape[1],
                                     g[i].device)
            near = self._band_apply(Duse, g[i], idx, chunk=ap_ch)
            w_par = far_box + near
            g[i] = None

        u = w_par.reshape(w_par.shape[0] * w_par.shape[1], w_par.shape[2])
        out = torch.empty_like(u); out[self.perm, :] = u
        if squeeze:
            out = out[:, 0]
        return out.detach().cpu().numpy() if was_np else out

    def matvec(self, v): return self._apply(v, False)
    def rmatvec(self, v): return self._apply(v, True)
    def __matmul__(self, v): return self._apply(v, False)


def _build_exact_hbs_1d(L, m, r, seed=0):
    rng = np.random.default_rng(seed)
    nleaf = 2 ** L; N = nleaf * m
    nbr = lambda i, j: abs(i - j) <= 1
    def adm(P, Q, lvl): return lvl != 0 and (not nbr(P, Q)) and nbr(P >> 1, Q >> 1)
    def lr(box, lvl):
        span = 2 ** (L - lvl); a0 = box * span
        return a0 * m, (a0 + span) * m
    orth = lambda a, b: np.linalg.qr(rng.standard_normal((a, b)))[0]
    Uf = {}; Vf = {}
    for lvl in range(L, 0, -1):
        for box in range(2 ** lvl):
            sz = m if lvl == L else 2 * r
            Uf[(lvl, box)] = orth(sz, r); Vf[(lvl, box)] = orth(sz, r)
    hU = {}; hV = {}
    for box in range(2 ** L):
        hU[(L, box)] = Uf[(L, box)]; hV[(L, box)] = Vf[(L, box)]
    for lvl in range(L - 1, 0, -1):
        for box in range(2 ** lvl):
            c1, c2 = 2*box, 2*box+1
            h1, h2 = hU[(lvl+1, c1)], hU[(lvl+1, c2)]
            bU = np.zeros((h1.shape[0]+h2.shape[0], 2*r)); bU[:h1.shape[0], :r]=h1; bU[h1.shape[0]:, r:]=h2
            hU[(lvl, box)] = bU @ Uf[(lvl, box)]
            k1, k2 = hV[(lvl+1, c1)], hV[(lvl+1, c2)]
            bV = np.zeros((k1.shape[0]+k2.shape[0], 2*r)); bV[:k1.shape[0], :r]=k1; bV[k1.shape[0]:, r:]=k2
            hV[(lvl, box)] = bV @ Vf[(lvl, box)]
    A = np.zeros((N, N))
    for lvl in range(1, L + 1):
        for P in range(2 ** lvl):
            for Q in range(2 ** lvl):
                if adm(P, Q, lvl):
                    S = rng.standard_normal((r, r))
                    r0, r1 = lr(P, lvl); c0, c1 = lr(Q, lvl)
                    A[r0:r1, c0:c1] = hU[(lvl, P)] @ S @ hV[(lvl, Q)].T
    for a in range(nleaf):
        for bb in range(nleaf):
            if nbr(a, bb):
                A[a*m:(a+1)*m, bb*m:(bb+1)*m] = rng.standard_normal((m, m))
    return torch.from_numpy(A)


if __name__ == "__main__":
    from scipy.sparse.linalg import aslinearoperator
    A = _build_exact_hbs_1d(3, 100, 5, seed=0); N = A.shape[0]
    LinOp = aslinearoperator(A.numpy())
    for fast in (False, True):
        H = HBSStrong(LinOp, Nb=8, L=3).construct(rk=5, p=10, fast=fast)
        e = (torch.from_numpy(H @ np.eye(N)) - A).norm() / torch.linalg.matrix_norm(A, ord='fro')
        print(f"1D exact-HBS (batched, fast={fast}): {e:.3e} "
              f"({'OK' if e < 1e-10 else 'FAIL'})")