"""
Adapter between SOMS3D_csr and solver.solverWrapper.

SOMS3D_csr works on the box [0, Lx] x [0, Ly] x [0, Lz] and returns the
interior system

    Sii u_i + Sib u_b = ftild[Ii]

on Legendre face dofs.  somsSolver places that box at the slab's position and
exposes it with the attribute names solverWrapper reads from every local
solver:

    XX     (n_dofs, 3) dof coordinates, global frame
    _Ji    interior dof indices        (SOMS Ii)
    _Jx    boundary dof indices        (SOMS Ib)
    Aii    Sii   (csr)
    Aix    Sib   (csr)
    Axi    None  (boundary rows: only needed for problem_type='mixed')
    Axx    None

The PDE is SOMS's own coefficient dict

    L u = sum_ij c_ij d_i d_j u + sum_i c_i d_i u + c u

with keys 'c11', 'c12', 'c13', 'c21', 'c22', 'c23', 'c31', 'c32', 'c33',
'c1', 'c2', 'c3', 'c'.  Each value is a real scalar or a vectorized callable
f(x, y, z) of numpy arrays in GLOBAL coordinates.  Missing keys are zero.
"""

from collections.abc import Mapping
import numbers

import numpy as np

from . import SOMS3D_csr as soms

COEFF_KEYS = ('c11', 'c12', 'c13', 'c21', 'c22', 'c23', 'c31', 'c32', 'c33',
              'c1', 'c2', 'c3', 'c')


def _three_ints(name, vals, minimum):
    try:
        vals = tuple(vals)
    except TypeError:
        raise ValueError("%s must be a sequence of 3 integers, got %r"
                         % (name, vals)) from None
    if len(vals) != 3:
        raise ValueError("%s must have 3 entries (x, y, z), got %d"
                         % (name, len(vals)))
    out = []
    for v in vals:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) \
                or v < minimum:
            raise ValueError("%s entries must be integers >= %d, got %r"
                             % (name, minimum, vals))
        out.append(int(v))
    return tuple(out)


def normalize_ord_tiling(ord, tiling):
    """((px, py, pz), (nbx, nby, nbz)) as int tuples, or ValueError."""
    if tiling is None:
        raise ValueError("solver type 'soms' needs tiling=(nbx, nby, nbz)")
    return _three_ints("ord", ord, 2), _three_ints("tiling", tiling, 1)


def _shifted_coeff(f, lo):
    """f evaluated at global coordinates, always returning one float per point.

    SOMS calls coefficients with LOCAL coordinates of its [0, L] box.  The
    result is also broadcast to the point count: a callable returning a bare
    scalar would otherwise break SOMS's per-row scaling c[:, None] * D.
    """
    x0, y0, z0 = (float(v) for v in lo)

    def g(x, y, z):
        v = np.asarray(f(x + x0, y + y0, z + z0))
        if np.iscomplexobj(v):
            raise TypeError("SOMS coefficients must be real; a callable "
                            "returned complex values")
        return np.broadcast_to(v.astype(np.float64, copy=False), np.shape(x))
    return g


def local_coeffs(pde, lo):
    """
    SOMS coefficient dict -> the dict SOMS_solver_sparse is called with:
    scalars unchanged (as float), callables shifted to global coordinates.
    Returns (coeffs, all_constant).
    """
    if not isinstance(pde, Mapping):
        raise TypeError(
            "solver type 'soms' takes the SOMS coefficient dict as its PDE "
            "(keys %s), got %s" % (", ".join(COEFF_KEYS), type(pde).__name__))
    unknown = sorted(set(pde) - set(COEFF_KEYS))
    if unknown:
        raise KeyError("unknown SOMS coefficient key(s) %s; valid keys: %s"
                       % (unknown, ", ".join(COEFF_KEYS)))
    coeffs, constant = {}, True
    for k, v in pde.items():
        if callable(v):
            coeffs[k] = _shifted_coeff(v, lo)
            constant = False
        elif isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)):
            coeffs[k] = float(v)
        else:
            raise TypeError("SOMS coefficient %r must be a real scalar or a "
                            "callable f(x, y, z), got %r" % (k, v))
    if not coeffs:
        raise ValueError("the SOMS coefficient dict is empty")
    return coeffs, constant


class somsSolver:
    """
    One SOMS slab discretization on the box `geom` = [[x0, y0, z0],
    [x1, y1, z1]], with Legendre order ord = (px, py, pz) per face and
    tiling = (nbx, nby, nbz) boxes.

    Constant coefficients (all scalars) use SOMS's ct_pde path: three local
    S matrices for the whole slab.  Any callable coefficient switches to the
    per-face path.
    """

    def __init__(self, pde, geom, ord, tiling, verbose=False):
        geom = np.asarray(geom, dtype=np.float64)
        if geom.shape != (2, 3):
            raise ValueError("SOMS is 3D only: geom must have shape (2, 3), "
                             "got %r" % (geom.shape,))
        lo, hi = geom.min(axis=0), geom.max(axis=0)
        L = hi - lo
        if np.any(L <= 0):
            raise ValueError("degenerate SOMS box %r" % (geom.tolist(),))

        (px, py, pz), (nbx, nby, nbz) = normalize_ord_tiling(ord, tiling)
        coeffs, constant = local_coeffs(pde, lo)

        Sii, Sib, _ftild, XYtot, Ii, Ib = soms.SOMS_solver_sparse(
            px, py, pz, nbx, nby, nbz, Lx=L[0], Ly=L[1], Lz=L[2],
            coeffs=coeffs, ct_pde=constant, forcing=None,
            dbg=1 if verbose else 0)

        self.ord = (px, py, pz)
        self.tiling = (nbx, nby, nbz)
        self.box_size = L / np.array([nbx, nby, nbz], dtype=np.float64)
        self.lo, self.hi = lo, hi
        self.ct_pde = constant

        self.XX = XYtot + lo
        self._Ji = np.asarray(Ii, dtype=np.int64)
        self._Jx = np.asarray(Ib, dtype=np.int64)
        self.Aii = Sii.tocsr()
        self.Aix = Sib.tocsr()
        # boundary (flux) rows exist only for the Neumann closure, which the
        # wrapper does not use yet (problem_type='mixed' is rejected)
        self.Axi = None
        self.Axx = None
