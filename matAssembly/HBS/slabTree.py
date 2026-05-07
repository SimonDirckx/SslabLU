"""
slabTree — a unified binary / quad spatial tree.

Splitting rules
---------------
Line input (intrinsic dimension 1):
  - Always binary (quad=True raises ValueError).
  - Each box is an interval [lo, hi] along the line's axis.
  - Split at the midpoint: left child gets [lo, mid], right child [mid, hi].

Rectangle input (intrinsic dimension 2), binary:
  - Each box is an axis-aligned rectangle [x_lo, x_hi] x [y_lo, y_hi].
  - Levels alternate splitting axis: even levels split x, odd levels split y.
  - Left/bottom child gets the lower half, right/top child the upper half.

Rectangle input (intrinsic dimension 2), quad:
  - Each box is an axis-aligned rectangle.
  - Every split divides both x and y at their midpoints simultaneously,
    producing 4 children: SW, SE, NW, NE.

Box index conventions
---------------------
  Binary  : children of box b -> 2b+1 (left/bottom), 2b+2 (right/top).
  Quad    : children of box b -> 4b+1 (SW), 4b+2 (SE), 4b+3 (NW), 4b+4 (NE).

All 2 (binary) or 4 (quad) children are always created at each split, even
if empty, so that the index formula is always valid. Empty boxes are leaves
that contain no points.

Geometry detection
------------------
XX shape (N,) or (N,1)     -> line in 1-D.
XX shape (N,d) with d >= 2 -> SVD test: if the second singular value is
                               negligible relative to the first (ratio <
                               line_tol), the cloud is a line; otherwise a
                               rectangle. The two splitting axes for the
                               rectangle case are always the raw coordinate
                               axes 0 and 1 of XX (the rectangle is assumed
                               axis-aligned).
"""

from __future__ import annotations

import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Geometry detection
# ---------------------------------------------------------------------------

def _is_line(XX: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True when XX lies (essentially) on a 1-D line."""
    if XX.shape[1] == 1:
        return True
    Xc = XX - XX.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    if sv[0] < tol:
        return True          # all points coincide
    return float(sv[1] / sv[0]) < tol


# ---------------------------------------------------------------------------
# Internal node
# ---------------------------------------------------------------------------

class _Box:
    """
    A single node in slabTree.

    Attributes
    ----------
    index      : unique integer identifier (follows parent/child formula)
    level      : depth in the tree (root = 0)
    bounds     : geometry of this box --
                   line   : (lo, hi)
                   rect   : (x_lo, x_hi, y_lo, y_hi)
    point_inds : indices into XX of the points owned by this box
    children   : list of child _Box objects (empty list -> leaf)
    """

    __slots__ = ("index", "level", "bounds", "point_inds", "children")

    def __init__(self, index: int, level: int, bounds: tuple,
                 point_inds: np.ndarray):
        self.index      = index
        self.level      = level
        self.bounds     = bounds
        self.point_inds = point_inds
        self.children: List[_Box] = []

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# slabTree
# ---------------------------------------------------------------------------

class slabTree:
    """
    Spatial slab tree -- binary or quad, for lines and axis-aligned rectangles.

    Splitting continues recursively until every leaf holds at most
    *min_leaf_size* points.

    Parameters
    ----------
    XX : np.ndarray
        Points, shape (N,), (N,1), (N,2), or (N,3).
        For rectangle input the rectangle is assumed axis-aligned; the two
        axes with the largest variance are used as the x- and y-splitting axes.

    quad : bool
        False -> binary tree.
        True  -> quad tree (rectangle input only; raises ValueError for lines).

    min_leaf_size : int
        Splitting stops when a box holds <= this many points.  Default 1
        (split all the way down to individual points).

    line_tol : float
        SVD singular-value ratio below which the cloud is declared a line.
        Default 1e-8.

    Examples
    --------
    >>> import numpy as np

    Binary tree on a 1-D line, leaves of size 1:
    >>> XX = np.linspace(0, 1, 32)
    >>> t = slabTree(XX, quad=False, min_leaf_size=1)
    >>> len(t.get_leaves())    # 32

    Binary tree on a 1-D line, leaves of size 4:
    >>> t = slabTree(XX, quad=False, min_leaf_size=4)
    >>> len(t.get_leaves())    # 8

    Binary tree on an axis-aligned rectangle in 2-D:
    >>> rng = np.random.default_rng(0)
    >>> XX = rng.random((64, 2))
    >>> t = slabTree(XX, quad=False, min_leaf_size=4)

    Quad tree on a rectangle:
    >>> t = slabTree(XX, quad=True, min_leaf_size=4)
    >>> t.get_boxes_level(1)   # [1, 2, 3, 4]  (SW, SE, NW, NE)

    quad=True on a line raises ValueError:
    >>> slabTree(np.linspace(0,1,20), quad=True)
    """

    def __init__(self, XX: np.ndarray, quad: bool = False,
                 min_leaf_size: int = 1, line_tol: float = 1e-8):

        # -- normalise input --------------------------------------------------
        XX = np.asarray(XX, dtype=float)
        if XX.ndim == 1:
            XX = XX[:, None]
        if XX.ndim != 2:
            raise ValueError("XX must be 1-D or 2-D.")

        # -- detect geometry --------------------------------------------------
        self._line = _is_line(XX, tol=line_tol)

        if quad and self._line:
            raise ValueError(
                "quad=True is not valid for line input. "
                "Use quad=False, or provide a 2-D (rectangle) point cloud.")

        self._XX            = XX
        self._quad          = quad
        self._min_leaf_size = min_leaf_size
        self._boxes: dict[int, _Box] = {}

        # -- find the two (or one) columns that actually vary -----------------
        # For an axis-aligned rectangle embedded in 2-D or 3-D, exactly two
        # columns have non-trivial spread; for a line, exactly one does.
        # We pick columns by descending variance so col_x has the larger
        # spread and col_y has the second-largest (rectangle only).
        col_vars  = XX.var(axis=0)
        sorted_cols = np.argsort(col_vars)[::-1]   # descending variance
        self._col_x = int(sorted_cols[0])           # primary axis
        self._col_y = int(sorted_cols[1]) if XX.shape[1] > 1 else None

        # -- root bounding box ------------------------------------------------
        if self._line:
            vals = XX[:, self._col_x]
            lo, hi = float(vals.min()), float(vals.max())
            eps = (hi - lo) * 1e-10 if hi > lo else 1e-10
            root_bounds = (lo - eps, hi + eps)
        else:
            x, y = XX[:, self._col_x], XX[:, self._col_y]
            ex = (x.max() - x.min()) * 1e-10 if x.max() > x.min() else 1e-10
            ey = (y.max() - y.min()) * 1e-10 if y.max() > y.min() else 1e-10
            root_bounds = (float(x.min()) - ex, float(x.max()) + ex,
                           float(y.min()) - ey, float(y.max()) + ey)

        root = _Box(index=0, level=0, bounds=root_bounds,
                    point_inds=np.arange(len(XX)))
        self._boxes[0] = root

        # -- build ------------------------------------------------------------
        self.perm_leaf  = np.empty(len(XX), dtype=int)
        self._perm_pos  = 0          # write cursor into perm_leaf

        if self._line:
            self._build_binary_line(root)
        elif quad:
            self._build_quad(root)
        else:
            self._build_binary_rect(root)

    # -- Line binary build ----------------------------------------------------

    def _write_leaf(self, box: _Box):
        """Write box.point_inds into perm_leaf at the current cursor."""
        n = len(box.point_inds)
        self.perm_leaf[self._perm_pos : self._perm_pos + n] = box.point_inds
        self._perm_pos += n

    def _build_binary_line(self, box: _Box):
        """Split interval [lo, hi] at its midpoint."""
        if len(box.point_inds) <= self._min_leaf_size:
            self._write_leaf(box)
            return

        lo, hi = box.bounds
        mid    = 0.5 * (lo + hi)
        vals   = self._XX[box.point_inds, self._col_x]

        left_mask  = vals <= mid
        right_mask = ~left_mask

        li = 2 * box.index + 1
        ri = 2 * box.index + 2

        left  = _Box(li, box.level + 1, (lo,  mid), box.point_inds[left_mask])
        right = _Box(ri, box.level + 1, (mid, hi),  box.point_inds[right_mask])

        box.children    = [left, right]
        self._boxes[li] = left
        self._boxes[ri] = right

        self._build_binary_line(left)
        self._build_binary_line(right)

    # -- Rectangle binary build -----------------------------------------------

    def _build_binary_rect(self, box: _Box):
        """
        Split rectangle alternating axes:
          even level -> split along x  (left  = lower x half, right = upper x half)
          odd  level -> split along y  (left  = lower y half, right = upper y half)
        """
        if len(box.point_inds) <= self._min_leaf_size:
            self._write_leaf(box)
            return

        x_lo, x_hi, y_lo, y_hi = box.bounds
        split_x = (box.level % 2 == 0)

        if split_x:
            mid  = 0.5 * (x_lo + x_hi)
            vals = self._XX[box.point_inds, self._col_x]
            left_bounds  = (x_lo, mid,  y_lo, y_hi)
            right_bounds = (mid,  x_hi, y_lo, y_hi)
        else:
            mid  = 0.5 * (y_lo + y_hi)
            vals = self._XX[box.point_inds, self._col_y]
            left_bounds  = (x_lo, x_hi, y_lo, mid)
            right_bounds = (x_lo, x_hi, mid,  y_hi)

        left_mask  = vals <= mid
        right_mask = ~left_mask

        li = 2 * box.index + 1
        ri = 2 * box.index + 2

        left  = _Box(li, box.level + 1, left_bounds,  box.point_inds[left_mask])
        right = _Box(ri, box.level + 1, right_bounds, box.point_inds[right_mask])

        box.children    = [left, right]
        self._boxes[li] = left
        self._boxes[ri] = right

        self._build_binary_rect(left)
        self._build_binary_rect(right)

    # -- Quad build -----------------------------------------------------------

    def _build_quad(self, box: _Box):
        """
        Split rectangle into 4 quadrants by halving both x and y:
          4b+1 = SW, 4b+2 = SE, 4b+3 = NW, 4b+4 = NE.
        All four children are always created (possibly empty).
        """
        if len(box.point_inds) <= self._min_leaf_size:
            self._write_leaf(box)
            return

        x_lo, x_hi, y_lo, y_hi = box.bounds
        x_mid = 0.5 * (x_lo + x_hi)
        y_mid = 0.5 * (y_lo + y_hi)

        px = self._XX[box.point_inds, self._col_x]
        py = self._XX[box.point_inds, self._col_y]

        masks = [
            (px <= x_mid) & (py <= y_mid),   # SW -> 4b+1
            (px >  x_mid) & (py <= y_mid),   # SE -> 4b+2
            (px <= x_mid) & (py >  y_mid),   # NW -> 4b+3
            (px >  x_mid) & (py >  y_mid),   # NE -> 4b+4
        ]
        child_bounds = [
            (x_lo,  x_mid, y_lo,  y_mid),    # SW
            (x_mid, x_hi,  y_lo,  y_mid),    # SE
            (x_lo,  x_mid, y_mid, y_hi),     # NW
            (x_mid, x_hi,  y_mid, y_hi),     # NE
        ]

        children = []
        for k, (mask, cb) in enumerate(zip(masks, child_bounds)):
            ci    = 4 * box.index + 1 + k
            child = _Box(ci, box.level + 1, cb, box.point_inds[mask])
            self._boxes[ci] = child
            children.append(child)

        box.children = children
        for child in children:
            self._build_quad(child)

    # -- Public API -----------------------------------------------------------

    def get_leaves(self) -> List[int]:
        """Return sorted list of box indices that are leaves."""
        return sorted(b.index for b in self._boxes.values() if b.is_leaf)

    def get_boxes_level(self, lvl: int) -> List[int]:
        """Return sorted list of box indices at depth *lvl*."""
        return sorted(b.index for b in self._boxes.values()
                      if b.level == lvl)

    def get_box_inds(self, box: int) -> np.ndarray:
        """
        Return the indices into XX of the points in box *box*.

        Parameters
        ----------
        box : int   Box index (0 = root).

        Returns
        -------
        np.ndarray of int  (empty array if the box holds no points)
        """
        if box not in self._boxes:
            raise KeyError(f"Box {box} does not exist in this tree.")
        return self._boxes[box].point_inds.copy()

    # -- Properties / repr ----------------------------------------------------

    @property
    def is_quad(self) -> bool:
        """True if this is a quad tree, False if binary."""
        return self._quad

    @property
    def is_line_input(self) -> bool:
        """True if the input was detected as a 1-D line."""
        return self._line

    @property
    def nlevels(self) -> int:
        """Number of levels in the tree (root counts as level 1)."""
        return max(b.level for b in self._boxes.values()) + 1
    @property
    def nleaves(self) -> int:
        return len(self.get_leaves())


    def __repr__(self) -> str:
        mode     = "quad" if self._quad else "binary"
        geom     = "line" if self._line else "rectangle"
        depth    = max(b.level for b in self._boxes.values())
        n_leaves = len(self.get_leaves())
        return (f"slabTree(mode={mode}, geometry={geom}, N={len(self._XX)}, "
                f"min_leaf_size={self._min_leaf_size}, depth={depth}, "
                f"leaves={n_leaves})")