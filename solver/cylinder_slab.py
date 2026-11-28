"""
Local double-slab solver for OMS on a straight cylinder, discretized by the
glued 5-patch HPS cylinder of stellerator.py (square core + 4 arched patches).

Coordinates
-----------
stellerator.py works in cylinder coordinates (x, y, z): cross-section (x, y),
axis z.  OMS stacks slabs along coordinate 0.  This wrapper therefore exposes
points in OMS coordinates

        w = (z, x, y)          (axial first, then the cross-section)

and builds the patches of the double slab  w0 in [geom[0][0], geom[1][0]]
(i.e. z in that range).  The PDO handed to it is an hpsmultidomain PDO_3d in
OMS coordinates w (e.g. from param_geom(...).transform_helmholtz_pdo, as for
the cube / twisted torus); it is converted to a stellerator.PDOPhys in
cylinder coordinates and pulled back to each patch.  A PDOPhys in cylinder
coordinates is accepted as well and used as is.

DOFs and blocks
---------------
After gluing (physical flux continuity across every leaf face and every patch
interface), the unknowns are the leaf-face nodes:
    counts == 2 : shared by two leaves      -> Ii  (incl. the midplane w0 = xc)
    counts == 1 : on the slab surface        -> Ib  (end caps + cylinder wall)
and   A[Ii] u = 0   is the glued (flux-balance) system, so
    Aii = A[Ii, Ii],  Aib = A[Ii, Ib]  (and Abi, Abb for completeness).

DOF order is canonical: sorted by cross-section position (x, then y), then by
z.  Every constant-w0 face set, taken in index order, is then ordered by its
cross-section coordinates, identically in every slab -- so slab i's left face
Il lines up with slab i-1's midplane Ic, as OMS assumes.  (Face nodes at a
given (x, y) carry bitwise-identical coordinates in every plane and slab: they
come from the same patch reference nodes.)
"""
import numpy as np
import scipy.sparse as sp
import torch

from stellerator import make_cylinder, build_patch, glue, PDOPhys

def _memoize_pdo(hps):
    """Cache the PDO coefficient values per evaluation point set.  hpsmultidomain
    re-evaluates every coefficient on every get_DtNs call (reduce/solve); with
    the metric of the 5-patch map that is the dominant per-call cost and does
    not change between calls."""
    pdo = hps.pdo
    for name in ("c11", "c22", "c33", "c12", "c13", "c23", "c1", "c2", "c3", "c"):
        f = getattr(pdo, name)
        if f is None:
            continue
        store = []

        def g(xx, f=f, store=store):
            for k, v in store:
                if k.shape == xx.shape and k.device == xx.device and torch.equal(k, xx):
                    return v.clone()
            v = f(xx)
            store.append((xx.detach().clone(), v.detach().clone()))
            return v
        setattr(pdo, name, g)


def _tensor_eval(Ls, idxs, ph, max_bytes=1 << 28):
    """v[l, s] = sum_n prod_a Ls[a][l, s, idxs[a][n]] ph[l, n]: leaf polynomials
    (nodal values ph, leaf l) at points s, from 1D Lagrange weights Ls[a]
    (nb, se, n_a).  When the nodes form the full tensor grid (the leaf case),
    ph is scattered to (nb, n0, n1, n2) and contracted one axis at a time,
    O(se n^3) per leaf with small temporaries; otherwise the (se, npl) weight
    matrix is formed explicitly.  Both are chunked over leaves (temporaries
    <= max_bytes), so memory stays bounded whatever the patch size."""
    nb, se = Ls[0].shape[:2]
    na = [L.shape[2] for L in Ls]
    npl = ph.shape[1]
    flat = (idxs[0] * na[1] + idxs[1]) * na[2] + idxs[2]
    tensor = np.prod(na) == npl and len(np.unique(flat)) == npl
    per_leaf = se * (na[0] * na[1] if tensor else npl) * 8
    ch = max(1, int(max_bytes // max(per_leaf, 1)))
    out = np.empty((nb, se))
    for a in range(0, nb, ch):
        b = min(nb, a + ch)
        L0, L1, L2 = (L[a:b] for L in Ls)
        if tensor:
            T = np.zeros((b - a, na[0] * na[1] * na[2]))
            T[:, flat] = ph[a:b]
            T = T.reshape(b - a, *na)
            tmp = np.einsum("lsk,lijk->lsij", L2, T)
            tmp = np.einsum("lsj,lsij->lsi", L1, tmp)
            out[a:b] = np.einsum("lsi,lsi->ls", L0, tmp)
        else:
            B = L0[:, :, idxs[0]] * L1[:, :, idxs[1]] * L2[:, :, idxs[2]]
            out[a:b] = np.einsum("lsn,ln->ls", B, ph[a:b])
    return out


def patch_order(p, kind):
    """Per-axis order of a patch.  p: int (all axes), or (p_n, p_m, p_z) with
        p_n : tangential -- core x and y, arch eta   (shared faces, n leaves)
        p_m : radial     -- arch xi                  (m leaves)
        p_z : axial      -- all patches              (nz leaves)
    Patch reference axes: core (x, y, z), arch (xi, eta, z)."""
    if np.ndim(p) == 0:
        return (int(p),) * 3
    pn, pm, pz = (int(v) for v in p)
    return (pn, pn, pz) if kind == "core" else (pm, pn, pz)


def slab_patches(geom, Rc, s, n, m, nz, stretch=0.0):
    """The five patches of the double slab w0 in [geom[0][0], geom[1][0]].
    stretch: radial clustering of the arch leaves (arch_stretch.py)."""
    z0, z1 = float(geom[0][0]), float(geom[1][0])
    patches = make_cylinder(s=s, L=z1 - z0, n=n, m=m, nz=nz, Rc=Rc, stretch=stretch)
    for P in patches:
        P.box[2] = [z0, z1]
        P.a = (P.box[:, 1] - P.box[:, 0]) / (2 * P.nleaf)
    return patches


_W_OF_CYL = [1, 2, 0]      # cylinder coordinate i (x, y, z) is OMS coordinate _W_OF_CYL[i]


def pdo3d_to_cyl(pdo):
    """PDO_3d in OMS coordinates w = (z, x, y), hpsmultidomain convention
        -(c11 u_11 + c22 u_22 + c33 u_33 + c12 u_12 + c13 u_13 + c23 u_23)
        + c1 u_1 + c2 u_2 + c3 u_3 + c u,
    -> PDOPhys (non-divergence form) in cylinder coordinates (x, y, z):
        a_kk = c_kk,  a_kl = c_kl / 2,  b_k = c_k,  then permuted w -> (x, y, z)."""
    cache = {}

    def ev(f, W, N):
        if f is None:
            return np.zeros(N)
        v = f(W)
        v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        return np.asarray(v, dtype=np.float64) * np.ones(N)

    def _points(X):
        X = np.asarray(X, dtype=np.float64)
        if not ("X" in cache and cache["X"].shape == X.shape and np.array_equal(cache["X"], X)):
            cache.clear()
            cache["X"] = X.copy()
            cache["W"] = torch.from_numpy(np.ascontiguousarray(X[:, [2, 0, 1]]))
        return cache["W"], len(X)

    def coeffs_ab(X):
        W, N = _points(X)
        if "ab" not in cache:
            aw = np.zeros((N, 3, 3))
            for k, f in enumerate((pdo.c11, pdo.c22, pdo.c33)):
                aw[:, k, k] = ev(f, W, N)
            for (k, l), f in (((0, 1), pdo.c12), ((0, 2), pdo.c13), ((1, 2), pdo.c23)):
                aw[:, k, l] = aw[:, l, k] = 0.5 * ev(f, W, N)
            bw = np.stack([ev(f, W, N) for f in (pdo.c1, pdo.c2, pdo.c3)], axis=1)
            p = _W_OF_CYL
            cache["ab"] = (aw[:, p][:, :, p], bw[:, p])
        return cache["ab"]

    def coeff_c(X):
        # evaluated only when asked for: the flux conversion at face slots needs
        # a only, and c may be defined only at leaf nodes (pb.coarse.NodalField)
        W, N = _points(X)
        if "c" not in cache:
            cache["c"] = ev(pdo.c, W, N)
        return cache["c"]

    return PDOPhys(a=lambda X: coeffs_ab(X)[0], b=lambda X: coeffs_ab(X)[1],
                   c=coeff_c, form="nondiv")


class CylinderSlab:
    def __init__(self, geom, pde, p, Rc=1.0, s=0.5, n=4, m=1, nz=2,
                 flux="normal", device=torch.device("cpu"), glue_tol=1e-9,
                 keep_leaf=False, stretch=0.0):
        geom = np.asarray(geom, dtype=np.float64)
        z0, z1 = float(geom[0][0]), float(geom[1][0])
        if not isinstance(pde, PDOPhys):
            pde = pdo3d_to_cyl(pde)
        if nz % 2:
            raise ValueError("nz must be even: the midplane w0 = xc must be a "
                             "leaf face (OMS takes Ic there)")

        # ---- patches of this double slab: z in [z0, z1] --------------------
        # p: int, or (p_n, p_m, p_z) per direction (see patch_order)
        patches = slab_patches(geom, Rc, s, n, m, nz, stretch)
        built = [build_patch(P, patch_order(p, P.kind), pde, device=device, flux=flux)
                 for P in patches]
        gl = glue(built, tol=glue_tol * max(1.0, Rc, abs(z0), abs(z1)))
        if keep_leaf:
            # keep the leaf operators (hps objects) and the gluing maps for
            # reduce_body / reconstruct; the patch matrices are not needed
            for b in built:
                b.pop("A"); b.pop("h")
                _memoize_pdo(b["hps"])
            self._built, self._G, self._offsets = built, gl["G"].tocsr(), gl["offsets"]
        del built                                  # (kept above if keep_leaf)

        counts = gl["counts"]
        if counts.max() > 2:
            raise RuntimeError("gluing produced a node shared by >2 leaves; "
                               "check glue_tol")

        # coordinates from the FIRST slot of each cluster (deterministic,
        # bitwise identical across planes / slabs for a given (x, y))
        _, first = np.unique(gl["labels"], return_index=True)
        Xcyl = gl["Xslot"][first]

        # canonical order: x, then y, then z  (lexsort: last key is primary)
        order = np.lexsort((Xcyl[:, 2], Xcyl[:, 1], Xcyl[:, 0]))
        Xcyl = Xcyl[order]
        counts = counts[order]
        A = gl["A"].tocsr()[order][:, order].tocsr()
        if np.iscomplexobj(A.data) and not np.any(A.data.imag):
            A = sp.csr_matrix((A.data.real, A.indices, A.indptr), shape=A.shape)

        self.XX = np.ascontiguousarray(Xcyl[:, [2, 0, 1]])   # OMS coords (z, x, y)
        self.Ii = np.flatnonzero(counts == 2)
        self.Ib = np.flatnonzero(counts == 1)
        Ai, Ab = A[self.Ii], A[self.Ib]
        self.Aii = Ai[:, self.Ii].tocsc()
        self.Aib = Ai[:, self.Ib].tocsr()
        self.Abi = Ab[:, self.Ii].tocsr()
        self.Abb = Ab[:, self.Ib].tocsr()
        self.zbounds = (z0, z1)
        self.device = device
        self.keep_leaf = keep_leaf
        if keep_leaf:
            self._order = order                    # canonical dof k = glue label order[k]

    # ---------------------------------------------------------------- leaf level
    # Leaf-grid fields ("nodes") are the concatenation over patches of each
    # patch's hps.xx_tot (nboxes * p^3 Chebyshev nodes per patch, face nodes of
    # neighbouring leaves duplicated).  Body loads are read at the leaf-interior
    # nodes only (interior_mask()); the rest of the entries are ignored.

    def _need_leaf(self):
        if not self.keep_leaf:
            raise RuntimeError("build CylinderSlab(..., keep_leaf=True)")

    def nodes(self):
        """Leaf-grid nodes in OMS coordinates (z, x, y), fixed order."""
        self._need_leaf()
        if not hasattr(self, "_nodes"):
            Xc = np.vstack([b["P"].to_phys(b["hps"].xx_tot.numpy()) for b in self._built])
            self._nodes = np.ascontiguousarray(Xc[:, [2, 0, 1]])
            sizes = [b["hps"].xx_tot.shape[0] for b in self._built]
            self._node_off = np.concatenate([[0], np.cumsum(sizes)])
        return self._nodes

    def interior_mask(self):
        """True at leaf-interior nodes (where body loads / PDE rows live)."""
        self._need_leaf()
        if not hasattr(self, "_imask"):
            m = []
            for b in self._built:
                hps = b["hps"]
                loc = np.zeros(int(np.prod(hps.p)), bool)
                loc[np.asarray(hps.H.JJ.Jc)] = True
                m.append(np.tile(loc, int(hps.nboxes)))
            self._imask = np.concatenate(m)
        return self._imask

    def _patch_vec(self, v, i):
        self.nodes()
        a, b = self._node_off[i], self._node_off[i + 1]
        return torch.from_numpy(np.ascontiguousarray(v[a:b], dtype=np.float64))

    def reduce_body(self, h, b=None):
        """Body load h on the leaf grid -> skeleton right-hand side.
        Returns (Qh) on all canonical dofs (sign convention of build_patch's h,
        i.e.  A x = Qh  is the glued system), and, if b (data on Ib) is given,
        beta = (Qh)[Ii] - Aib b  as well:   x_I = Aii^{-1} beta."""
        self._need_leaf()
        slots = []
        for i, bl in enumerate(self._built):
            r = bl["hps"].get_DtNs(self.device, mode="reduce_body",
                                   ff_body_vec=self._patch_vec(h, i))
            slots.append(bl["w"] * r.detach().cpu().numpy().reshape(-1))
        Qh = (self._G.T @ np.concatenate(slots))[self._order]
        if b is None:
            return Qh
        return Qh, Qh[self.Ii] - self.Aib @ b

    def reconstruct(self, x, h=None):
        """Skeleton values x (all canonical dofs) and body load h (leaf grid)
        -> solution on the leaf grid (same order as nodes())."""
        self._need_leaf()
        lab = np.empty_like(x)
        lab[self._order] = x
        uslots = self._G @ lab
        out = []
        for i, bl in enumerate(self._built):
            hps = bl["hps"]
            us = uslots[self._offsets[i]:self._offsets[i + 1]]
            uu = torch.from_numpy(np.ascontiguousarray(us[hps.I_unique.numpy()])).unsqueeze(-1)
            hv = None if h is None else self._patch_vec(h, i)
            sol, _ = hps.solve(self.device, uu, ff_body_vec=hv)
            out.append(sol[:, 0].detach().cpu().numpy())
        return np.concatenate(out)

    def midplane(self, tol=1e-10):
        """Canonical dof indices on the midplane w0 = (z0 + z1) / 2 (all in Ii)."""
        zc = 0.5 * sum(self.zbounds)
        I = np.flatnonzero(np.abs(self.XX[:, 0] - zc) <= tol * max(1.0, abs(zc)))
        assert np.all(np.isin(I, self.Ii))
        return I

    def leaf_grids(self):
        """Per patch: (Patch, hps.grid_xx (nboxes, p^3, 3) in patch reference
        coordinates, node offset) -- for off-grid interpolation."""
        self.nodes()
        return [(b["P"], b["hps"].grid_xx.numpy(), self._node_off[i])
                for i, b in enumerate(self._built)]

    @staticmethod
    def to_cyl(w):
        """OMS coordinates (z, x, y) -> cylinder coordinates (x, y, z)."""
        w = w.detach().cpu().numpy() if torch.is_tensor(w) else np.asarray(w)
        return w[:, [1, 2, 0]]


# =============================================================================
#  Geometry built once, zero-order coefficient updated cheaply
# =============================================================================
class CylinderSlabOps:
    """Same discretization, DOF order and blocks as CylinderSlab(keep_leaf=True),
    restructured for repeated solves and coefficient updates.

    Leaf operator: K_l + diag(c) on the interior nodes, K_l from the PDO given
    here (geometry; its own zero-order term, normally none) and c set by set_c.
    Built ONCE (independent of c), per patch:
        K_II = Aloc[Jc, Jc],   E = Aloc[Jc, Jxun] @ Iu        (per leaf, on device)
        Nc   = Ir @ Nxc[:, Jc], Nf = Ir @ Nxc[:, Jxun] @ Iu   (shared by the leaves)
        w, B : physical-flux conversion (stellerator.flux_conversion)
        gluing, canonical order, Ii / Ib, CSR patterns of Aii, Aib and the
        scatter maps leaf-DtN entry -> CSR data position.
    set_c(c), per patch (batched dense) + one scatter:
        LU of K_II + diag(c_I),  X = Acc^{-1} E,  DtN = Nf - Nc X,  Y = -X
        Aii.data = scatter(w * DtN) + (B part)
    Per solve (one leaf solve, shared by reduction and reconstruction):
        z = Acc^{-1} h_I,  flux = -w * Nc z  -> glue -> beta = (Qh)_I - Aib b
        phi_I = z + Y x_F,  phi_F(Chebyshev) = Iu x_F
    (Iu: Gauss -> Chebyshev face interpolation, Ir: Chebyshev -> Gauss.)
    """

    def __init__(self, geom, pde, p, Rc=1.0, s=0.5, n=4, m=1, nz=2,
                 flux="conormal", device=torch.device("cpu"), glue_tol=1e-9,
                 chunk_bytes=2e8, periodic=False, stretch=0.0):
        """periodic=True: glue the end faces w0 = geom[0][0] and geom[1][0]
        (periodic tube, e.g. the whole torus for the coarse problem); Ib is
        then the wall only."""
        from scipy.spatial import cKDTree
        from scipy.sparse.csgraph import connected_components
        from stellerator import flux_conversion
        from solver.hpsmultidomain.hpsmultidomain.hps_multidomain_disc import HPS_Multidomain
        import solver.hpsmultidomain.hpsmultidomain.hps_parallel_leaf_ops as leaf_ops

        geom = np.asarray(geom, dtype=np.float64)
        z0, z1 = float(geom[0][0]), float(geom[1][0])
        if nz % 2:
            raise ValueError("nz must be even (midplane must be a leaf face)")
        if flux not in ("normal", "conormal"):
            raise ValueError("CylinderSlabOps needs a physical flux ('normal'/'conormal')")
        if not isinstance(pde, PDOPhys):
            pde = pdo3d_to_cyl(pde)
        self.device, self.zbounds, self.keep_leaf = device, (z0, z1), True
        dt = torch.float64

        # ---- per patch: geometry-only leaf operators ------------------------
        self.patches, Xslot = [], []
        for P in slab_patches(geom, Rc, s, n, m, nz, stretch):
            pv = np.array(patch_order(p, P.kind))
            hps = HPS_Multidomain(P.pdo(pde), torch.tensor(P.box), P.a, pv, 3)
            assert hps.interpolate, "Gauss face nodes required for gluing"
            H = hps.H
            Jc = np.asarray(H.JJ.Jc).ravel(); Jxun = np.asarray(H.JJ.Jxunique).ravel()
            Iu = np.asarray(H.Interp_mat_unique, float)      # (|Jxun|, size_ext)
            Ir = np.asarray(H.Interp_mat_reverse, float)     # (size_ext, n_face_cheb)
            Nxc = np.asarray(H.Nxc, float)
            nb, npl, nI, se = int(hps.nboxes), int(np.prod(pv)), len(Jc), Iu.shape[1]
            T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
            Iu_t = T(Iu)
            Nc, Nf = T(Ir @ Nxc[:, Jc]), T(Ir @ Nxc[:, Jxun] @ Iu)
            xx, Ds = hps.grid_xx.to(device), H.Ds.to(device)
            Jc_t = torch.as_tensor(Jc, device=device); Jx_t = torch.as_tensor(Jxun, device=device)
            K_II = torch.empty(nb, nI, nI, dtype=dt, device=device)
            E = torch.empty(nb, nI, se, dtype=dt, device=device)
            chunk = max(1, int(chunk_bytes // (npl * npl * 8)))
            for b0 in range(0, nb, chunk):
                b1 = min(nb, b0 + chunk)
                Arow = leaf_ops.get_Aloc_3d(pv, xx, Ds, hps.pdo, b0, b1, device)[:, Jc_t]
                K_II[b0:b1] = Arow[:, :, Jc_t]
                E[b0:b1] = Arow[:, :, Jx_t] @ Iu_t
                del Arow
            X = hps.xx_ext.numpy()
            w, B = flux_conversion(P, X, np.array(hps.q), nb, pde, flux)
            self.patches.append(dict(P=P, pv=pv, nb=nb, npl=npl, nI=nI, se=se, Xref=X,
                                     Jc=Jc, Jxun=Jxun, Jc_t=Jc_t, Jx_t=Jx_t, Iu=Iu_t,
                                     Nc=Nc, Nf=Nf, K_II=K_II, E=E, w=w,
                                     w_t=T(w.reshape(nb, se)), grid=hps.grid_xx.numpy()))
            Xslot.append(P.to_phys(X))
            self.patches[-1]["B"] = B
            del hps

        # ---- gluing topology (same clusters / order as CylinderSlab) ---------
        Xphys = np.vstack(Xslot)
        self._slot_off = np.concatenate([[0], np.cumsum([len(x) for x in Xslot])])
        tol = glue_tol * max(1.0, Rc, abs(z0), abs(z1))
        self.periodic = bool(periodic)
        if self.periodic:                      # identify w0 = z1 with w0 = z0
            Xphys = Xphys.copy()
            top = Xphys[:, 2] > z1 - tol
            Xphys[top, 2] -= (z1 - z0)
        pairs = cKDTree(Xphys).query_pairs(tol, output_type="ndarray")
        Ns = len(Xphys)
        adj = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(Ns, Ns))
        _, labels = connected_components(adj, directed=False)
        counts = np.bincount(labels)
        if counts.max() > 2:
            raise RuntimeError("gluing produced a node shared by >2 leaves; check glue_tol")
        _, first = np.unique(labels, return_index=True)
        Xcyl = Xphys[first]
        order = np.lexsort((Xcyl[:, 2], Xcyl[:, 1], Xcyl[:, 0]))
        canon = np.empty_like(order); canon[order] = np.arange(len(order))
        self.dof_of_slot = canon[labels]                       # slot -> canonical dof
        counts = counts[order]
        self.XX = np.ascontiguousarray(Xcyl[order][:, [2, 0, 1]])
        self.Ii = np.flatnonzero(counts == 2)
        self.Ib = np.flatnonzero(counts == 1)
        self.ndof = len(counts)
        Ni, Nb = len(self.Ii), len(self.Ib)
        posI = np.full(self.ndof, -1); posI[self.Ii] = np.arange(Ni)
        posB = np.full(self.ndof, -1); posB[self.Ib] = np.arange(Nb)

        # ---- CSR patterns and scatter maps for Aii, Aib ----------------------
        keys = {"ii": [], "ib": []}; vals_B = {"ii": [], "ib": []}
        self._sel = []
        for i, d in enumerate(self.patches):
            nb, se = d["nb"], d["se"]
            dof = self.dof_of_slot[self._slot_off[i] + np.arange(nb * se)].reshape(nb, se)
            r = posI[dof][:, :, None]
            sel = {}
            for blk, cpos, N in (("ii", posI, Ni), ("ib", posB, Nb)):
                c = cpos[dof][:, None, :]
                m_ = (r >= 0) & (c >= 0)
                flat = np.flatnonzero(m_)
                sel[blk] = flat
                keys[blk].append((np.broadcast_to(r, m_.shape).ravel()[flat] * N
                                  + np.broadcast_to(c, m_.shape).ravel()[flat]))
            self._sel.append(sel)
            Bc = d.pop("B").tocoo()
            rd, cd = dof.ravel()[Bc.row], dof.ravel()[Bc.col]
            for blk, cpos, N in (("ii", posI, Ni), ("ib", posB, Nb)):
                ok = (posI[rd] >= 0) & (cpos[cd] >= 0)
                keys[blk].append(posI[rd[ok]] * N + cpos[cd[ok]])
                vals_B[blk].append(Bc.data[ok])
        self._csr = {}
        for blk, shape in (("ii", (Ni, Ni)), ("ib", (Ni, Nb))):
            allk = np.concatenate(keys[blk])
            uk, inv = np.unique(allk, return_inverse=True)
            rows, cols = uk // shape[1], uk % shape[1]
            indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=shape[0]))])
            sizes = [len(k) for k in keys[blk]]
            parts = np.split(inv, np.cumsum(sizes)[:-1])
            # parts: [dtn patch 0, B patch 0, dtn patch 1, B patch 1, ...]
            dtn_pos = parts[0::2]; B_pos = parts[1::2]
            base = np.zeros(len(uk))
            for bp, bv in zip(B_pos, vals_B[blk]):
                base += np.bincount(bp, bv, minlength=len(uk))
            self._csr[blk] = dict(shape=shape, indptr=indptr, indices=cols.astype(np.int32),
                                  base=base, dtn_pos=dtn_pos)
        self.Aii = self.Aib = None
        self.Abi = self.Abb = None
        self.c_set = False

    def release_geometry(self):
        """Free the c-independent leaf data (K_II, E): set_c is no longer
        possible, solves still are.  For slabs factored once."""
        for d in self.patches:
            d["K_II"] = d["E"] = None
        self._csr = self._sel = None           # assembly maps (only set_c needs them)
        self._geometry_released = True

    def skeleton_from_field(self, phi):
        """Values at all canonical skeleton dofs of a leaf-grid field phi
        (nodes() order): each dof's leaf polynomial evaluated at its Gauss node."""
        self.nodes()
        vals = np.empty(self.ndof)
        for i, d in enumerate(self.patches):
            grid, X, nb, se, npl = d["grid"], d["Xref"], d["nb"], d["se"], d["npl"]
            leaf = np.arange(nb * se) // se
            lo, hi = grid.min(axis=1), grid.max(axis=1)
            rel0 = (grid[0] - lo[0]) / (hi[0] - lo[0])
            r = (X - lo[leaf]) / (hi[leaf] - lo[leaf])
            Ls, idxs = [], []                      # 1D barycentric Lagrange weights per axis
            for a in range(3):
                u, idx = np.unique(np.round(rel0[:, a], 12), return_inverse=True)
                wb = 1.0 / np.prod(u[:, None] - u[None, :] + np.eye(len(u)), axis=1)
                dd = r[:, a][:, None] - u[None, :]
                hit = np.abs(dd) < 1e-13
                dd[hit] = 1.0
                t = wb / dd
                La = t / t.sum(axis=1, keepdims=True)
                rows = hit.any(axis=1)
                La[rows] = hit[rows]
                Ls.append(La.reshape(nb, se, len(u)))
                idxs.append(idx.reshape(-1))
            ph = np.asarray(phi[self._node_off[i]:self._node_off[i + 1]], float).reshape(nb, npl)
            v = _tensor_eval(Ls, idxs, ph).reshape(-1)
            vals[self.dof_of_slot[self._slot_off[i]:self._slot_off[i + 1]]] = v
        return vals

    # ------------------------------------------------------------ coefficient
    def set_c(self, c):
        """Zero-order coefficient c at the leaf-grid nodes (nodes() order; only
        leaf-interior values are used).  Rebuilds leaf factors, DtNs, Aii, Aib."""
        if getattr(self, "_geometry_released", False):
            raise RuntimeError("set_c after release_geometry()")
        c = np.asarray(c, dtype=np.float64)
        self.nodes()
        data = {blk: d["base"].copy() for blk, d in self._csr.items()}
        for i, d in enumerate(self.patches):
            cI = torch.as_tensor(c[self._node_off[i]:self._node_off[i + 1]].reshape(d["nb"], d["npl"])
                                 [:, d["Jc"]], dtype=torch.float64, device=self.device)
            A = d["K_II"].clone()
            A.diagonal(dim1=1, dim2=2).add_(cI)
            LU, piv = torch.linalg.lu_factor(A)
            del A
            X = torch.linalg.lu_solve(LU, piv, d["E"])          # Acc^{-1} E
            DtN = d["Nf"].unsqueeze(0) - d["Nc"].unsqueeze(0) @ X
            d["LU"], d["piv"], d["Y"] = LU, piv, X.neg_()
            V = (d["w_t"].unsqueeze(-1) * DtN).reshape(-1).cpu().numpy()
            del DtN
            for blk in ("ii", "ib"):
                data[blk] += np.bincount(self._csr[blk]["dtn_pos"][i], V[self._sel[i][blk]],
                                         minlength=len(data[blk]))
        mk = lambda blk: sp.csr_matrix((data[blk], self._csr[blk]["indices"],
                                        self._csr[blk]["indptr"]), shape=self._csr[blk]["shape"])
        self.Aii, self.Aib = mk("ii"), mk("ib")
        self.c_set = True

    # ------------------------------------------------------------ leaf grid
    def nodes(self):
        """Leaf-grid nodes in OMS coordinates (z, x, y), fixed order (as CylinderSlab)."""
        if not hasattr(self, "_nodes"):
            Xc = np.vstack([d["P"].to_phys(d["grid"].reshape(-1, 3)) for d in self.patches])
            self._nodes = np.ascontiguousarray(Xc[:, [2, 0, 1]])
            self._node_off = np.concatenate([[0], np.cumsum([d["nb"] * d["npl"] for d in self.patches])])
        return self._nodes

    def interior_mask(self):
        if not hasattr(self, "_imask"):
            m = []
            for d in self.patches:
                loc = np.zeros(d["npl"], bool); loc[d["Jc"]] = True
                m.append(np.tile(loc, d["nb"]))
            self._imask = np.concatenate(m)
        return self._imask

    def leaf_grids(self):
        self.nodes()
        return [(d["P"], d["grid"], self._node_off[i]) for i, d in enumerate(self.patches)]

    @property
    def grid(self):
        """LeafGrid-compatible view (for pb.prolong.Prolongation)."""
        from types import SimpleNamespace
        self.nodes()
        return SimpleNamespace(
            patches=[dict(P=d["P"], grid=d["grid"], nleaf=np.array(d["P"].nleaf, dtype=int),
                          pv=d["pv"]) for d in self.patches],
            nodes=self._nodes, offsets=self._node_off)

    def midplane(self, tol=1e-10):
        zc = 0.5 * sum(self.zbounds)
        I = np.flatnonzero(np.abs(self.XX[:, 0] - zc) <= tol * max(1.0, abs(zc)))
        assert np.all(np.isin(I, self.Ii))
        return I

    # ------------------------------------------------------------ solves
    def _hI(self, h, i):
        d = self.patches[i]
        v = np.asarray(h[self._node_off[i]:self._node_off[i + 1]], dtype=np.float64)
        return torch.as_tensor(v.reshape(d["nb"], d["npl"])[:, d["Jc"]],
                               dtype=torch.float64, device=self.device).unsqueeze(-1)

    def reduce_body(self, h, b=None, return_z=False):
        """As CylinderSlab.reduce_body; with return_z also the leaf solves
        z = Acc^{-1} h_I, to be passed to reconstruct (no second leaf solve)."""
        if not self.c_set:
            raise RuntimeError("call set_c first")
        self.nodes()
        flux = np.empty(self._slot_off[-1])
        zs = []
        for i, d in enumerate(self.patches):
            z = torch.linalg.lu_solve(d["LU"], d["piv"], self._hI(h, i))
            q = -(d["w_t"] * (d["Nc"].unsqueeze(0) @ z)[..., 0])
            flux[self._slot_off[i]:self._slot_off[i + 1]] = q.reshape(-1).cpu().numpy()
            zs.append(z)
        Qh = np.bincount(self.dof_of_slot, flux, minlength=self.ndof)
        out = (Qh,) if b is None else (Qh, Qh[self.Ii] - self.Aib @ b)
        if return_z:
            out = out + (zs,)
        return out if len(out) > 1 else out[0]

    def reconstruct(self, x, h=None, z=None):
        """Skeleton values x (all canonical dofs) and the leaf solves z (from
        reduce_body(..., return_z=True)) or the body load h -> leaf-grid field."""
        self.nodes()
        uslots = np.asarray(x, dtype=np.float64)[self.dof_of_slot]
        out = []
        for i, d in enumerate(self.patches):
            xf = torch.as_tensor(uslots[self._slot_off[i]:self._slot_off[i + 1]].reshape(d["nb"], d["se"], 1),
                                 dtype=torch.float64, device=self.device)
            zi = (z[i] if z is not None else
                  torch.linalg.lu_solve(d["LU"], d["piv"], self._hI(h, i)) if h is not None else 0.0)
            u = torch.zeros(d["nb"], d["npl"], dtype=torch.float64, device=self.device)
            u[:, d["Jc_t"]] = (d["Y"] @ xf + zi)[..., 0]
            u[:, d["Jx_t"]] = (d["Iu"].unsqueeze(0) @ xf)[..., 0]
            out.append(u.reshape(-1).cpu().numpy())
        return np.concatenate(out)
