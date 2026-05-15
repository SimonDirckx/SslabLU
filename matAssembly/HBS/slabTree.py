"""
slabTree — a unified binary / quad spatial tree.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def ldur(nx,ny):
    xpts = np.linspace(0,nx,nx)
    ypts = np.linspace(0,ny,ny)
    xy = np.zeros((nx*ny,2))
    xy[:,0] = np.kron(xpts,np.ones_like(ypts))
    xy[:,1] = np.kron(np.ones_like(xpts),ypts)
    l = np.where((xy[:,0]==0) & (xy[:,1]>0) & (xy[:,1]<ny))[0]
    d = np.where((xy[:,1]==0) & (xy[:,0]>0) & (xy[:,0]<nx-1))[0]
    u = np.where((xy[:,1]==ny-1) & (xy[:,0]>0) & (xy[:,0]<nx-1))[0]
    r = np.where((xy[:,0]==nx-1) & (xy[:,1]>0) & (xy[:,1]<ny))[0]
    return l,d,u,r

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


class _Node:
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
    children   : list of child _Node objects (empty list -> leaf)
    bDOFs      : dict with keys 'left','down','up','right', each a numpy array
                 of indices into XX giving the boundary DOFs, in order
                 Set during tree construction for rect binary trees.
    iDOFs      : np.ndarray of indices into XX for the interface DOFs
                 (interior points on the split line, in order).
                 None for leaves.
    """

    __slots__ = ("index", "level", "bounds", "point_inds", "children",
                 "bDOFs", "iDOFs","u")

    def __init__(self, index: int, level: int, bounds: tuple,
                 point_inds: np.ndarray):
        self.index      = index
        self.level      = level
        self.bounds     = bounds
        self.point_inds = point_inds
        self.children: list[_Node] = []
        self.bDOFs      = None   # populated by _build_binary_rect
        self.iDOFs      = None   # populated by _build_binary_rect (None for leaves)
        self.u          = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
def _outer_of_pair(p_node, q_node, direction):
    """
    Outer boundary indices and 4-segment sizes for an adjacent pair (p, q).
    direction: 'horizontal' (p LEFT of q) or 'vertical' (p BELOW q).
    Returns (outer_idxs, n_l, n_d, n_u, n_r).
    """
    if direction == 'horizontal':
        outer = np.concatenate([p_node.bDOFs['left'],
                                p_node.bDOFs['down'], q_node.bDOFs['down'],
                                p_node.bDOFs['up'],   q_node.bDOFs['up'],
                                q_node.bDOFs['right']])
        n_l = len(p_node.bDOFs['left'])
        n_d = len(p_node.bDOFs['down']) + len(q_node.bDOFs['down'])
        n_u = len(p_node.bDOFs['up'])   + len(q_node.bDOFs['up'])
        n_r = len(q_node.bDOFs['right'])
    else:
        outer = np.concatenate([p_node.bDOFs['left'],  q_node.bDOFs['left'],
                                p_node.bDOFs['down'],
                                q_node.bDOFs['up'],
                                p_node.bDOFs['right'], q_node.bDOFs['right']])
        n_l = len(p_node.bDOFs['left'])  + len(q_node.bDOFs['left'])
        n_d = len(p_node.bDOFs['down'])
        n_u = len(q_node.bDOFs['up'])
        n_r = len(p_node.bDOFs['right']) + len(q_node.bDOFs['right'])
    return outer, n_l, n_d, n_u, n_r


class slabTree:
    """
    Spatial slab tree -- binary or quad, for lines and axis-aligned rectangles/lines.

    Top-down; splitting continues recursively until min_leaf size or max_level reached

    Parameters
    ----------
    XX : np.ndarray
        Points, shape (N,), (N,1), (N,2), or (N,3).
        For rectangle input the rectangle is assumed axis-aligned; the two
        axes with the largest variance are used as the x- and y-splitting axes.
        Lex ordering is maintained

    quad : bool
        False -> binary tree.
        True  -> quad tree (rectangle input only; raises ValueError for lines).

    min_leaf_size : int
        minimum leaf size (default = 1)

    line_tol : float
        SVD singular-value ratio below which the cloud is declared a line.
        Default 1e-8.

    """

    def __init__(self, XX: np.ndarray, quad: bool = False,
                 min_leaf_size: int = 1,max_level=np.inf, line_tol: float = 1e-8):

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
        self._max_level = max_level
        self._boxes: dict[int, _Node] = {}
        vars = []
        print("XX shape = ",XX.shape)
        for d in range(XX.shape[1]):
            if XX[:,d].var()>line_tol:#delete constant vars for rect in 3D
                vars+=[d]
        
        sorted_cols = vars
        print("sorted_cols = ",sorted_cols)
        self._col_x = int(sorted_cols[0])
        self._col_y = int(sorted_cols[1]) if len(sorted_cols) > 1 else None
        if self._line:
            vals = XX[:, self._col_x]
            lo, hi = float(vals.min()), float(vals.max())
            root_bounds = (lo, hi)
        else:
            x, y = XX[:, self._col_x], XX[:, self._col_y]
            root_bounds = (float(x.min()), float(x.max()),
                           float(y.min()), float(y.max()))

        root = _Node(index=0, level=0, bounds=root_bounds,
                    point_inds=np.arange(len(XX)))
        self._boxes[0] = root

        # -- build ------------------------------------------------------------
        self.perm_leaf  = np.zeros(shape=(XX.shape[0],), dtype=int)
        self._perm_pos  = 0          # write cursor into perm_leaf
        # adjacent_pairs: list of (left_or_bot_idx, right_or_top_idx, orientation)
        # orientation is 'horizontal' (shared vertical boundary) or
        # 'vertical' (shared horizontal boundary).
        
        if self._line:
            self._build_binary_line(root)
        elif quad:
            self._build_quad(root)
        else:
            self._build_binary_rect(root)
        
        self._build_adjacency()
        #self.build_perm()
    # -- Line binary build ----------------------------------------------------
    def build_perm(self):
        leaves = self.get_leaves()
        start = 0
        print("NLEAVES = ",len(leaves))
        for leaf in leaves:
            print(start,",",leaf,",",len(self.get_box_inds(leaf)),"//",len(self.perm_leaf))
            self.perm_leaf[start:start+len(self.get_box_inds(leaf))] = self.get_box_inds(leaf)
            start+=len(self.get_box_inds(leaf))
    def _build_binary_line(self, box: _Node):
        """Split interval [lo, hi] at its midpoint."""
        if len(box.point_inds) <= self._min_leaf_size:
            return

        lo, hi = box.bounds
        mid    = 0.5 * (lo + hi)
        vals   = self._XX[box.point_inds, self._col_x]

        left_mask  = vals <= mid+1e-10
        right_mask = vals>=mid-1e-10

        li = 2 * box.index + 1
        ri = 2 * box.index + 2

        left  = _Node(li, box.level + 1, (lo,  mid), box.point_inds[left_mask])
        right = _Node(ri, box.level + 1, (mid, hi),  box.point_inds[right_mask])

        box.children    = [left, right]
        self._boxes[li] = left
        self._boxes[ri] = right

        self._build_binary_line(left)
        self._build_binary_line(right)

    # -- Rectangle binary build -----------------------------------------------

    def _build_binary_rect(self, box: _Node):
        """
        Split rectangle alternating axes:
          odd  level -> split along x  (left child = lower x, right = upper x)
          even level -> split along y  (left child = lower y, right = upper y)

        Assigns bDOFs and iDOFs to every node.

        bDOFs : dict with keys 'left','down','up','right' — corner-excluded
                boundary DOF indices into XX, ordered along each side.
        iDOFs : interface DOF indices (interior points on the split line,
                ordered along the line). None for leaves.

        """
        tol = 1e-10
        x_lo, x_hi, y_lo, y_hi = box.bounds

        # ----- LEAF -----
        if box.level == self._max_level or len(box.point_inds) <= self._min_leaf_size:
            pts = box.point_inds
            XX  = self._XX[pts,:]
            
            left  = pts[np.where((abs(XX[:,self._col_x]-x_lo)<tol) &
                                 (XX[:,self._col_y]>y_lo+tol) &
                                 (XX[:,self._col_y]<y_hi-tol))[0]]
            right = pts[np.where((abs(XX[:,self._col_x]-x_hi)<tol) &
                                 (XX[:,self._col_y]>y_lo+tol) &
                                 (XX[:,self._col_y]<y_hi-tol))[0]]
            down  = pts[np.where((abs(XX[:,self._col_y]-y_lo)<tol) &
                                 (XX[:,self._col_x]>x_lo+tol) &
                                 (XX[:,self._col_x]<x_hi-tol))[0]]
            up    = pts[np.where((abs(XX[:,self._col_y]-y_hi)<tol) &
                                 (XX[:,self._col_x]>x_lo+tol) &
                                 (XX[:,self._col_x]<x_hi-tol))[0]]
            box.bDOFs = {'left': left, 'down': down, 'up': up, 'right': right}
            box.iDOFs = None
            return {'left': [box.index], 'right': [box.index],
                    'down': [box.index], 'up': [box.index]}

        # ----- INTERNAL NODE -----
        split_x = (box.level % 2 == 1)

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

        left_mask  = vals <= mid+tol
        right_mask = vals>=mid-tol

        li = 2 * box.index + 1
        ri = 2 * box.index + 2

        left  = _Node(li, box.level + 1, left_bounds,  box.point_inds[left_mask])
        right = _Node(ri, box.level + 1, right_bounds, box.point_inds[right_mask])

        box.children    = [left, right]
        self._boxes[li] = left
        self._boxes[ri] = right

        left_sides  = self._build_binary_rect(left)
        right_sides = self._build_binary_rect(right)
        #XX  = self._XX
        pts = box.point_inds

        if split_x:
            # x-split: left child LEFT, right child RIGHT
            box.iDOFs = left.bDOFs['right']
            box.bDOFs = {
                'left':  left.bDOFs['left'],
                'right': right.bDOFs['right'],
                'down':  np.concatenate([left.bDOFs['down'],  right.bDOFs['down']]),
                'up':    np.concatenate([left.bDOFs['up'],    right.bDOFs['up']]),
            }
            
            return {
                'left':   left_sides['left'],
                'right':  right_sides['right'],
                'down': left_sides['down'] + right_sides['down'],
                'up':    left_sides['up']    + right_sides['up'],
            }
        else:
            box.iDOFs = left.bDOFs['up']
            box.bDOFs = {
                'left':  np.concatenate([left.bDOFs['left'],  right.bDOFs['left']]),
                'right': np.concatenate([left.bDOFs['right'], right.bDOFs['right']]),
                'down':  left.bDOFs['down'],
                'up':    right.bDOFs['up'],
            }
            
            return {
                'left':   left_sides['left']   + right_sides['left'],
                'right':  left_sides['right']  + right_sides['right'],
                'down': left_sides['down'],
                'up':    right_sides['up'],
            }
    def _build_adjacency(self):
        adjacency = []
        for lvl in range(self.nlevels):
            split_x = (lvl % 2 == 1)
            if split_x:
                split_dir = 'vertical'
                opp_dir = 'horizontal'
            else:
                split_dir = 'horizontal'
                opp_dir = 'vertical'
            adj_lvl = []
            if lvl == 0:
                adjacency+=[adj_lvl]
            else:
                for node in self.get_nodes_level(lvl-1):
                    adj_lvl+=[(node.children[0].index,node.children[1].index,split_dir)]
                for gr in adjacency[lvl-1]:
                    if gr[2]==split_dir:
                        adj_lvl+=[(self.get_node(gr[0]).children[1].index,self.get_node(gr[1]).children[0].index,split_dir)]
                    else:
                        adj_lvl+=[(self.get_node(gr[0]).children[0].index,self.get_node(gr[1]).children[0].index,opp_dir)]
                        adj_lvl+=[(self.get_node(gr[0]).children[1].index,self.get_node(gr[1]).children[1].index,opp_dir)]
                adjacency+=[adj_lvl]
        self.adjacency = adjacency



    # -- Quad build -----------------------------------------------------------

    def _build_quad(self, box: _Node):
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
            child = _Node(ci, box.level + 1, cb, box.point_inds[mask])
            self._boxes[ci] = child
            children.append(child)

        box.children = children
        for child in children:
            self._build_quad(child)

    def get_leaves(self) -> list[int]:
        """Return sorted list of box indices that are leaves."""
        return sorted(b.index for b in self._boxes.values() if b.is_leaf)

    def get_boxes_level(self, lvl: int) -> list[int]:
        """Return sorted list of box indices at depth *lvl*."""
        return sorted(b.index for b in self._boxes.values()
                      if b.level == lvl)
    def get_nodes_level(self, lvl: int) -> list[int]:
        """Return sorted list of box indices at depth *lvl*."""
        return [b for b in self._boxes.values() if b.level == lvl]
    
    def get_node(self, idx: int):
        
        return self._boxes[idx]
    

    def solve(self,node:_Node,interface_map,dir):
        if node.is_leaf:
            return
        else:
            tau = node.children[0]
            sig = node.children[1]
            #print("nodes are : ",tau.index,"//",sig.index)
            imap = interface_map(tau.index,sig.index)
            S = imap.S
            b = node.u
            uc = S@b
            if dir=='vertical':
                _,nl,nd,nu,nr = _outer_of_pair(tau,sig,'vertical')
                ut = np.concatenate([   b[:nl//2],
                                        b[nl:nl+nd],
                                        uc,
                                        b[nl+nd+nu:nl+nd+nu+nr//2]
                                        ])
                us = np.concatenate([   b[nl//2:nl],
                                        uc,
                                        b[nl+nd:nl+nd+nu],
                                        b[nl+nd+nu+nr//2:]
                                    ])
                self.set_u(tau,ut)
                self.set_u(sig,us)
                self.solve(tau,interface_map,'horizontal')
                self.solve(sig,interface_map,'horizontal')
            else:
                _,nl,nd,nu,nr = _outer_of_pair(tau,sig,'horizontal')
                ut = np.concatenate([   b[:nl],
                                        b[nl:nl+nd//2],
                                        b[nl+nd:nl+nd+nu//2],
                                        uc
                                        ])
                us = np.concatenate([   uc,
                                        b[nl+nd//2:nl+nd],
                                        b[nl+nd+nu//2:nl+nd+nu],
                                        b[nl+nd+nu:]
                                    ])
                self.set_u(tau,ut)
                self.set_u(sig,us)
                self.solve(tau,interface_map,'vertical')
                self.solve(sig,interface_map,'vertical')




    def set_u(self,node :_Node,u):
        node.u = u


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