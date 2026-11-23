"""
circInterfaceTree -- binary tree over the DOFs of one cross-section ("interface")
of the 5-patch cylinder discretisation (square core + 4 arched patches), built so
that HBSMAT (HBStorch_new.HBSMAT) can use it directly.

Geometry of the cross-section (radius Rc, core half-width s*Rc):
    core     : (x, y) in [-s Rc, s Rc]^2, n_core x n_core HPS leaves
    arch k   : unit-disk blending map, rotated by k*pi/2 (k = 0: east, 1: north,
               2: west, 3: south), reference (xi, eta) in [-1, 1]^2 with
               m_arch HPS leaves radially (xi) and n_core leaves tangentially (eta);
               leaves are uniform in xi.  With stretch beta != 0 the radial blend
               parameter is x = g(xi) (arch_stretch.py): leaves graded physically
               toward the wall, but the leaf-index structure -- and so this whole
               tree -- is unchanged.  The tree must be given the SAME stretch as
               the discretisation (DOF -> HPS leaf goes through the inverse map);
               a mismatch is caught by the leaf-boundary and equal-count checks.
    DOFs     : the points on the cross-section faces of the HPS leaves (e.g. the
               q x q Gauss nodes per leaf), passed in physical coordinates.

Tree ("pie" top levels, then per-patch bisection):
    level 0  whole cross-section
    level 1  x < 0 | x > 0
    level 2  y < 0 | y > 0                          -> 4 quarters
    level 3  core quarter | the two half arched patches of that quarter
    level 4  core quarter: bisection;  arched pair: split along the diagonal
    deeper   each node is a box of HPS leaves in ONE patch: ordinary binary
             dissection in leaf-index space -- halve the side with more leaves,
             alternating directions on ties -- down to single HPS leaves, and
             optionally further inside an HPS leaf on its q x q point grid
             (same rule on the point grid).

HBSMAT requirements (all enforced here):
    * complete binary tree, all leaves at depth L-1, nleaves = 2**(L-1);
    * every node at a level holds the same number of DOFs (HBSMAT uses equal
      block sizes per level);
    * perm_leaf lists the leaves' DOFs leaf by leaf, leaves in left-to-right
      order, so the fac = 2 consecutive blocks of a level form their parent.
For the pie levels this forces n_core = 4 * m_arch with m_arch a power of two
(then the core quarter and the arched pair of a quarter hold equally many HPS
leaves), and the same number of DOFs on every HPS leaf face.

Below the pie levels every split is by HPS-leaf (or point-grid) INDEX, never by a
geometric midpoint, so a non-uniform physical leaf layout (radial stretch) still
gives equal-size children; the pie levels split by x = 0, y = 0 and patch id,
which the radial stretch does not move.

Interface used by HBSMAT: perm_leaf, nleaves, nlevels.  slabTree-style queries
(get_leaves, get_boxes_level, get_nodes_level, get_node, get_box_inds) are
provided as well; node indices follow slabTree: children of b are 2b+1, 2b+2.
"""
import numpy as np

from arch_stretch import blend_of_xi, xi_of_blend

CORE = 0
ARCH_NAMES = {1: "arch E", 2: "arch N", 3: "arch W", 4: "arch S"}


# ---------------------------------------------------------------------------
# geometry of the 5-patch cross-section
# ---------------------------------------------------------------------------
class CircCrossSection:
    """Maps between reference and physical coordinates of the 5 patches.
    Patch ids: 0 = core, 1..4 = arches rotated by (id-1)*pi/2."""

    def __init__(self, Rc=1.0, s=0.5, n_core=4, m_arch=1, stretch=0.0):
        if not 0 < s < 1 / np.sqrt(2):
            raise ValueError("need 0 < s < 1/sqrt(2) for a valid arch map")
        self.Rc, self.s, self.n, self.m = float(Rc), float(s), int(n_core), int(m_arch)
        self.stretch = float(stretch)          # arch radial clustering (arch_stretch.py)

    # leaf counts per patch, as (count along ref dir 0, count along ref dir 1)
    def leaf_shape(self, patch):
        return (self.n, self.n) if patch == CORE else (self.m, self.n)

    @staticmethod
    def _rot(k):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        return np.array([[c, -s], [s, c]])

    def to_phys(self, patch, ref):
        """ref (N, 2): core (x, y); arch (xi, eta).  Returns (N, 2)."""
        ref = np.atleast_2d(ref)
        if patch == CORE:
            return ref.copy()
        xi, eta = blend_of_xi(ref[:, 0], self.stretch), ref[:, 1]   # xi: blend parameter x
        th = np.pi * eta / 4
        u = (1 - xi) / 2 * self.s + (1 + xi) / 2 * np.cos(th)
        v = (1 - xi) / 2 * self.s * eta + (1 + xi) / 2 * np.sin(th)
        return self.Rc * np.c_[u, v] @ self._rot(patch - 1).T

    def classify(self, X):
        """Patch id of physical points X (N, 2)."""
        x, y = X[:, 0], X[:, 1]
        a = self.s * self.Rc
        pid = np.zeros(len(X), int)
        out = (np.abs(x) > a) | (np.abs(y) > a)
        ang = np.mod(np.arctan2(y, x) + np.pi / 4, 2 * np.pi)   # sector of +-45 deg
        pid[out] = 1 + np.floor(ang[out] / (np.pi / 2)).astype(int) % 4
        return pid

    def to_ref(self, patch, X, tol=1e-13, maxit=50):
        """Inverse map for points X (N, 2) known to lie in `patch`: Newton for the
        (unstretched) blend parameter x and eta, then xi = g^{-1}(x)."""
        X = np.atleast_2d(X)
        if patch == CORE:
            return X.copy()
        uv = (X @ self._rot(patch - 1)) / self.Rc        # rotate back, unit disk
        u, v = uv[:, 0], uv[:, 1]
        th = np.arctan2(v, u)
        eta = np.clip(4 * th / np.pi, -1, 1)
        r, r_in = np.hypot(u, v), self.s / np.maximum(np.cos(th), 1e-3)
        xi = np.clip(2 * (r - r_in) / np.maximum(1 - r_in, 1e-3) - 1, -1, 1)
        s = self.s
        for _ in range(maxit):
            t = np.pi * eta / 4
            fu = (1 - xi) / 2 * s + (1 + xi) / 2 * np.cos(t) - u
            fv = (1 - xi) / 2 * s * eta + (1 + xi) / 2 * np.sin(t) - v
            a11 = (np.cos(t) - s) / 2
            a12 = -(1 + xi) / 2 * np.sin(t) * np.pi / 4
            a21 = (np.sin(t) - s * eta) / 2
            a22 = (1 - xi) / 2 * s + (1 + xi) / 2 * np.cos(t) * np.pi / 4
            det = a11 * a22 - a12 * a21
            dxi = (a22 * fu - a12 * fv) / det
            deta = (-a21 * fu + a11 * fv) / det
            xi, eta = xi - dxi, eta - deta
            if max(np.abs(dxi).max(initial=0), np.abs(deta).max(initial=0)) < tol:
                break
        return np.c_[xi_of_blend(xi, self.stretch), eta]

    def ref_box(self, patch):
        """Reference-coordinate box of a patch: ((lo0, hi0), (lo1, hi1))."""
        a = self.s * self.Rc
        return ((-a, a), (-a, a)) if patch == CORE else ((-1.0, 1.0), (-1.0, 1.0))

    def leaf_of(self, patch, ref):
        """HPS leaf indices (i, j) of reference points in a patch."""
        (l0, h0), (l1, h1) = self.ref_box(patch)
        c0, c1 = self.leaf_shape(patch)
        i = np.floor((ref[:, 0] - l0) / (h0 - l0) * c0).astype(int)
        j = np.floor((ref[:, 1] - l1) / (h1 - l1) * c1).astype(int)
        return np.clip(i, 0, c0 - 1), np.clip(j, 0, c1 - 1)

    def extent(self, patch, box, axis, N=24):
        """Physical length of a leaf box's midline along reference `axis`.
        box = (i0, i1, j0, j1) in HPS-leaf index space of the patch."""
        (l0, h0), (l1, h1) = self.ref_box(patch)
        c0, c1 = self.leaf_shape(patch)
        i0, i1, j0, j1 = box
        r0 = l0 + (h0 - l0) * np.array([i0, i1]) / c0
        r1 = l1 + (h1 - l1) * np.array([j0, j1]) / c1
        t = np.linspace(0, 1, N)
        if axis == 0:
            ref = np.c_[r0[0] + t * (r0[1] - r0[0]), np.full(N, r1.mean())]
        else:
            ref = np.c_[np.full(N, r0.mean()), r1[0] + t * (r1[1] - r1[0])]
        P = self.to_phys(patch, ref)
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    # -- convenience: the Gauss nodes of every HPS leaf face of a cross-section
    def gauss_nodes(self, q):
        """Physical (N, 2) points: q x q Gauss-Legendre nodes per HPS leaf."""
        g = (np.polynomial.legendre.leggauss(q)[0] + 1) / 2
        pts = []
        for patch in range(5):
            (l0, h0), (l1, h1) = self.ref_box(patch)
            c0, c1 = self.leaf_shape(patch)
            for i in range(c0):
                for j in range(c1):
                    a = l0 + (h0 - l0) * (i + g) / c0
                    b = l1 + (h1 - l1) * (j + g) / c1
                    A, B = np.meshgrid(a, b, indexing="ij")
                    pts.append(self.to_phys(patch, np.c_[A.ravel(), B.ravel()]))
        return np.vstack(pts)


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------
class _Node:
    """Tree node.  kind: 'multi' (several patches, pie levels), 'box' (a box
    of HPS leaves in one patch) or 'sub' (part of one HPS leaf's point grid).
    patch: patch id or -1; box: (i0, i1, j0, j1) leaf-index box ('box') or
    point-grid index box inside the leaf ('sub'); leaf: (patch, i, j) for
    'sub' nodes; point_inds: indices into XX."""
    __slots__ = ("index", "level", "kind", "patch", "box", "leaf", "label",
                 "point_inds", "children", "bounds", "axis")

    def __init__(self, index, level, kind, point_inds, patch=-1, box=None,
                 leaf=None, label=""):
        self.index, self.level, self.kind = index, level, kind
        self.patch, self.box, self.leaf, self.label = patch, box, leaf, label
        self.point_inds = point_inds
        self.children = []
        self.bounds = None
        self.axis = None          # axis of the split that created this node (box/sub)

    @property
    def is_leaf(self):
        return len(self.children) == 0


class circInterfaceTree:
    """Binary tree over the DOFs of one cross-section of the 5-patch cylinder.

    Parameters
    ----------
    XX : (N, 2) or (N, 3) physical DOF coordinates (a z column is ignored).
    Rc, s, n_core, m_arch : the cross-section discretisation (see module doc).
    extra_levels : bisect each HPS leaf's point grid this many more times
                   (needs the grid to be a regular q0 x q1 tensor grid with
                   enough factors of two).  Default 0: tree leaves = HPS leaves.
    max_level : stop splitting at this level (coarser tree leaves); default:
                go down to single HPS leaves (+ extra_levels).
    patch_id, ref : optional per-DOF patch ids and reference coordinates; if
                    not given they are computed from XX (inverse arch map).
    """

    def __init__(self, XX, Rc=1.0, s=0.5, n_core=4, m_arch=1, extra_levels=0,
                 max_level=None, patch_id=None, ref=None, tol=1e-10, stretch=0.0):
        if hasattr(XX, "detach"):                    # torch tensor (e.g. stMap.XXI)
            XX = XX.detach().cpu().numpy()
        XX = np.asarray(XX, float)
        if XX.ndim != 2 or XX.shape[1] not in (2, 3):
            raise ValueError("XX must be (N, 2) or (N, 3)")
        self._XX = XX
        self.geom = G = CircCrossSection(Rc, s, n_core, m_arch, stretch)
        n, m = G.n, G.m
        if n != 4 * m or m & (m - 1):
            raise ValueError(
                f"HBS needs equal block sizes per level; with the pie top levels "
                f"this requires n_core = 4 * m_arch, m_arch a power of two "
                f"(got n_core={n}, m_arch={m}).  E.g. (4, 1), (8, 2), (16, 4).")
        X2 = XX[:, :2]

        # -- patch, reference coordinates and HPS leaf of every DOF ---------
        pid = G.classify(X2) if patch_id is None else np.asarray(patch_id, int)
        R = np.empty((len(X2), 2))
        if ref is None:
            for p in range(5):
                sel = pid == p
                if sel.any():
                    R[sel] = G.to_ref(p, X2[sel])
            err = np.abs(np.vstack([G.to_phys(p, R[pid == p]) for p in range(5)])
                         - np.vstack([X2[pid == p] for p in range(5)])).max()
            if err > 1e-8 * Rc:
                raise RuntimeError(f"inverse patch map did not converge ({err:.1e})")
        else:
            R[:] = ref
        li, lj = np.empty(len(X2), int), np.empty(len(X2), int)
        for p in range(5):
            sel = pid == p
            if sel.any():
                li[sel], lj[sel] = G.leaf_of(p, R[sel])
        self.patch_id, self.ref, self.leaf_i, self.leaf_j = pid, R, li, lj
        self._check_leaf_interior(pid, R)

        # DOFs per HPS leaf must all be equal (equal HBS block sizes)
        key = pid * 10 ** 6 + li * 10 ** 3 + lj
        _, cnt = np.unique(key, return_counts=True)
        nhps = n * n + 4 * m * n
        if len(cnt) != nhps or cnt.min() != cnt.max():
            raise ValueError(f"every one of the {nhps} HPS leaves must carry the "
                             f"same number of DOFs (found {len(cnt)} leaves, "
                             f"counts {cnt.min()}..{cnt.max()})")
        self.dofs_per_hps_leaf = int(cnt[0])
        self._extra = int(extra_levels)
        self._max_level = np.inf if max_level is None else int(max_level)
        self._tol = tol

        # -- build ------------------------------------------------------------
        self._boxes = {}
        root = _Node(0, 0, "multi", np.arange(len(X2)), label="cross-section")
        self._boxes[0] = root
        self._build(root)
        self._check_complete()
        self.build_perm()

    # ------------------------------------------------------------------ build
    def _child(self, parent, which, kind, mask, **kw):
        idx = 2 * parent.index + 1 + which
        node = _Node(idx, parent.level + 1, kind, parent.point_inds[mask], **kw)
        self._boxes[idx] = node
        return node

    def _split(self, node, mask0, kind0, kind1, kw0, kw1):
        a = self._child(node, 0, kind0, mask0, **kw0)
        b = self._child(node, 1, kind1, ~mask0, **kw1)
        node.children = [a, b]
        return a, b

    def _build(self, node):
        if node.level >= self._max_level:
            return
        G, X = self.geom, self._XX
        pts = node.point_inds
        pid = self.patch_id[pts]
        lvl = node.level

        if node.kind == "multi":
            if lvl == 0:                                   # x = 0
                m0 = X[pts, 0] < 0
                kids = self._split(node, m0, "multi", "multi",
                                   dict(label="x<0"), dict(label="x>0"))
            elif lvl == 1:                                 # y = 0
                m0 = X[pts, 1] < 0
                kids = self._split(node, m0, "multi", "multi",
                                   dict(label=node.label + ",y<0"),
                                   dict(label=node.label + ",y>0"))
            elif lvl == 2:                                 # core quarter | arched pair
                m0 = pid == CORE
                box = self._leaf_box(pts[m0], CORE)
                kids = self._split(node, m0, "box", "multi",
                                   dict(patch=CORE, box=box, label="core quarter"),
                                   dict(label="arched pair"))
            else:                                          # arched pair: diagonal
                ps = np.unique(pid)
                if len(ps) != 2:
                    raise RuntimeError("arched pair must span two patches")
                m0 = pid == ps[0]
                kids = self._split(
                    node, m0, "box", "box",
                    dict(patch=int(ps[0]), box=self._leaf_box(pts[m0], ps[0]),
                         label=ARCH_NAMES[int(ps[0])] + " half"),
                    dict(patch=int(ps[1]), box=self._leaf_box(pts[~m0], ps[1]),
                         label=ARCH_NAMES[int(ps[1])] + " half"))
            for c in kids:
                self._build(c)
            return

        if node.kind == "box":
            i0, i1, j0, j1 = node.box
            cand = [ax for ax, c in ((0, i1 - i0), (1, j1 - j0)) if c >= 2 and c % 2 == 0]
            if not cand:                                    # a single HPS leaf
                if i1 - i0 == 1 and j1 - j0 == 1 and self._extra > 0:
                    node.kind = "sub"
                    node.leaf = (node.patch, i0, j0)
                    self._init_sub(node)
                    self._build(node)
                return
            ax = self._dissect_axis(node, (i1 - i0, j1 - j0), cand)
            if ax == 0:
                mid = (i0 + i1) // 2
                m0 = self.leaf_i[pts] < mid
                b0, b1 = (i0, mid, j0, j1), (mid, i1, j0, j1)
            else:
                mid = (j0 + j1) // 2
                m0 = self.leaf_j[pts] < mid
                b0, b1 = (i0, i1, j0, mid), (i0, i1, mid, j1)
            kids = self._split(node, m0, "box", "box",
                               dict(patch=node.patch, box=b0, label=node.label),
                               dict(patch=node.patch, box=b1, label=node.label))
            for c in kids:
                c.axis = ax
                self._build(c)
            return

        # kind == "sub": bisect the point grid inside one HPS leaf
        if node.level >= self._sub_stop:
            return
        i0, i1, j0, j1 = node.box
        cand = [ax for ax, c in ((0, i1 - i0), (1, j1 - j0)) if c >= 2 and c % 2 == 0]
        if not cand:
            raise ValueError("extra_levels too large for the point grid of an HPS "
                             "leaf (needs even counts to bisect)")
        ax = self._dissect_axis(node, (i1 - i0, j1 - j0), cand)
        gi, gj = self._grid_idx[pts, 0], self._grid_idx[pts, 1]
        if ax == 0:
            mid = (i0 + i1) // 2
            m0 = gi < mid
            b0, b1 = (i0, mid, j0, j1), (mid, i1, j0, j1)
        else:
            mid = (j0 + j1) // 2
            m0 = gj < mid
            b0, b1 = (i0, i1, j0, mid), (i0, i1, mid, j1)
        kids = self._split(node, m0, "sub", "sub",
                           dict(patch=node.patch, box=b0, leaf=node.leaf, label=node.label),
                           dict(patch=node.patch, box=b1, leaf=node.leaf, label=node.label))
        for c in kids:
            c.axis = ax
            self._build(c)

    @staticmethod
    def _dissect_axis(node, counts, cand):
        """Ordinary binary dissection in index space: halve the side with more
        cells (HPS leaves, or grid points inside a leaf); on a tie, cut the
        other way than the split that created this node (axis 0 first)."""
        if len(cand) == 1:
            return cand[0]
        if counts[0] != counts[1]:
            return 0 if counts[0] > counts[1] else 1
        return 0 if node.axis in (None, 1) else 1

    def _leaf_box(self, pts, patch):
        i, j = self.leaf_i[pts], self.leaf_j[pts]
        return (int(i.min()), int(i.max()) + 1, int(j.min()), int(j.max()) + 1)

    def _init_sub(self, node):
        """Index the regular point grid of one HPS leaf (by reference coords)."""
        if not hasattr(self, "_grid_idx"):
            self._grid_idx = np.zeros((len(self._XX), 2), int)
            self._sub_h = {}
            self._sub_stop = node.level + self._extra
        pts = node.point_inds
        R = self.ref[pts]
        u0, i = np.unique(np.round(R[:, 0], 12), return_inverse=True)
        u1, j = np.unique(np.round(R[:, 1], 12), return_inverse=True)
        if len(u0) * len(u1) != len(pts):
            raise ValueError("extra_levels needs a tensor point grid on each HPS leaf")
        self._grid_idx[pts, 0], self._grid_idx[pts, 1] = i, j
        # physical spacing per grid step in each reference direction
        P = self.geom.to_phys(node.patch, R)
        c = np.argmin(np.abs(i - len(u0) // 2) + np.abs(j - len(u1) // 2))
        hs = []
        for ax, (uu, ii) in enumerate(((u0, i), (u1, j))):
            span = self.geom.to_phys(node.patch, np.array(
                [[uu[0], R[c, 1]], [uu[-1], R[c, 1]]] if ax == 0 else
                [[R[c, 0], uu[0]], [R[c, 0], uu[-1]]]))
            hs.append(np.linalg.norm(span[1] - span[0]) / max(len(uu) - 1, 1))
        self._sub_h[node.leaf] = hs
        node.box = (0, len(u0), 0, len(u1))

    # ------------------------------------------------------------------ checks
    def _check_leaf_interior(self, pid, R, rel_tol=1e-6):
        """Every DOF must lie strictly inside one HPS leaf in reference space
        (Gauss face nodes never sit on a leaf boundary).  A DOF on a boundary
        means the reference coordinates are wrong -- typically the tree was
        given a different arch stretch than the discretisation -- and the
        leaf assignment (hence the whole tree) would be ambiguous."""
        G = self.geom
        for p in range(5):
            sel = pid == p
            if not sel.any():
                continue
            (l0, h0), (l1, h1) = G.ref_box(p)
            c0, c1 = G.leaf_shape(p)
            for a, (lo, hi, c) in enumerate(((l0, h0, c0), (l1, h1, c1))):
                f = (R[sel, a] - lo) / (hi - lo) * c
                d = np.abs(f - np.rint(f))
                if d.min() < rel_tol:
                    raise ValueError(
                        f"patch {p}: DOF on an HPS leaf boundary along reference axis {a} "
                        f"(distance {d.min():.1e} leaf widths); wrong stretch "
                        f"(tree has {G.stretch})?")

    def _check_complete(self):
        leaves = [b for b in self._boxes.values() if b.is_leaf]
        depth = {b.level for b in leaves}
        if len(depth) != 1:
            raise RuntimeError(f"tree is not complete: leaves at levels {sorted(depth)}")
        for lvl in range(self.nlevels):
            sizes = {len(b.point_inds) for b in self.get_nodes_level(lvl)}
            if len(sizes) != 1:
                raise RuntimeError(f"unequal node sizes {sorted(sizes)} at level {lvl}")
        if self.nleaves != 2 ** (self.nlevels - 1):
            raise RuntimeError("leaf count is not a power of two")

    def build_perm(self):
        """perm_leaf: DOFs leaf by leaf, leaves left to right; inside a leaf
        sorted by (patch, reference coordinates)."""
        out = []
        for idx in self.get_leaves():
            p = self._boxes[idx].point_inds
            order = np.lexsort((self.ref[p, 1], self.ref[p, 0], self.patch_id[p]))
            out.append(p[order])
        self.perm_leaf = np.concatenate(out).astype(np.int64)
        if not np.array_equal(np.sort(self.perm_leaf), np.arange(len(self._XX))):
            raise RuntimeError("perm_leaf is not a permutation of the DOFs")

    # ------------------------------------------------------------------ queries
    def get_leaves(self):
        """Sorted leaf indices; for this complete tree that is left-to-right."""
        return sorted(b.index for b in self._boxes.values() if b.is_leaf)

    def get_boxes_level(self, lvl):
        return sorted(b.index for b in self._boxes.values() if b.level == lvl)

    def get_nodes_level(self, lvl):
        return [self._boxes[i] for i in self.get_boxes_level(lvl)]

    def get_node(self, idx):
        return self._boxes[idx]

    def get_box_inds(self, box):
        if box not in self._boxes:
            raise KeyError(f"Box {box} does not exist in this tree.")
        return self._boxes[box].point_inds.copy()

    @property
    def nlevels(self):
        return max(b.level for b in self._boxes.values()) + 1

    @property
    def nleaves(self):
        return len(self.get_leaves())

    @property
    def is_quad(self):
        return False

    @property
    def is_line_input(self):
        return False

    def __repr__(self):
        return (f"circInterfaceTree(N={len(self._XX)}, levels={self.nlevels}, "
                f"leaves={self.nleaves}, leaf size={len(self.perm_leaf) // self.nleaves}, "
                f"n_core={self.geom.n}, m_arch={self.geom.m}, s={self.geom.s}, "
                f"stretch={self.geom.stretch})")
