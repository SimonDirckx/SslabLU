import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sparse
from scipy.sparse import block_diag
from solver.pde_solver import AbstractPDESolver
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
from solver import sparse_utils
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Low-level stencil helpers (kept for potential external use / 1-D problems)
# ---------------------------------------------------------------------------

def stencilD(pts):
    h = pts[1] - pts[0]
    D = np.eye(len(pts))
    e = np.ones(shape=(len(pts) - 1,))
    D = D - np.diag(e, -1)
    return D / h

def stencilD2(pts):
    D  = stencilD(pts)
    D2 = D.T @ D
    D2[0, 0] = -D2[0, 1]
    return -D2

def stencilDxy(ptsx, ptsy):
    hx = ptsx[1] - ptsx[0]
    hy = ptsy[1] - ptsy[0]
    ex = np.ones(shape=(len(ptsx),))
    ey = np.ones(shape=(len(ptsy),))
    Dx   = -np.diag(ex, -1) - np.diag(ex, 1)
    Dy   = -np.diag(ey, -1) - np.diag(ey, 1)
    DxDy = sparse.kron(Dx, Dy) / (4. * hx * hy)
    return DxDy


# ---------------------------------------------------------------------------
# Direct CSR assembly – 5-point stencil (2-D)
# ---------------------------------------------------------------------------
# Operator:  L u = c11 u_xx + c22 u_yy [+ c1 u_y] [+ c u]   (coeffs as given)
#
# Flat index:  k = ix * Ny + iy,   ix in [0, Nx),  iy in [0, Ny)
#
# Interior row contributions:
#   diag        += -c11[k]*2/hx^2  -  c22[k]*2/hy^2   [+ c[k]]
#   (k, k - Ny) +=  c11[k] / hx^2          (x-left  neighbour)
#   (k, k + Ny) +=  c11[k] / hx^2          (x-right neighbour)
#   (k, k -  1) +=  c22[k] / hy^2          (y-left  neighbour)
#   (k, k +  1) +=  c22[k] / hy^2          (y-right neighbour)
#
# Boundary rows:  identity  (row k, col k) = 1
#
# We build Aii and Aix simultaneously, never forming the full N×N matrix.
# ---------------------------------------------------------------------------

def _constructPDO2D_csr(op, xpts, ypts, XX,
                        Ji, Jx, Ni, Nx_bnd):
    """
    Build Aii (Ni×Ni) and Aix (Ni×Nx_bnd) directly in CSR format.

    Parameters
    ----------
    op      : PDO object with fields c11, c22, (optional) c1, c
    xpts    : 1-D array, length Nx
    ypts    : 1-D array, length Ny
    XX      : (Nx*Ny, 2) array of all grid points  (row-major: x varies slow)
    Ji      : interior point indices into the flat grid  (length Ni)
    Jx      : boundary point indices into the flat grid  (length Nx_bnd)
    Ni, Nx_bnd : len(Ji), len(Jx)
    """
    Nx, Ny = len(xpts), len(ypts)
    hx = xpts[1] - xpts[0]
    hy = ypts[1] - ypts[0]
    ax = 1.0 / hx**2          # second-difference coefficient in x
    ay = 1.0 / hy**2          # second-difference coefficient in y

    # Precompute PDO coefficients at every grid point
    c11 = op.c11(XX)          # shape (Nx*Ny,)
    c22 = op.c22(XX)
    c1  = op.c1(XX)  if op.c1 else None
    c0  = op.c(XX)   if op.c  else None

    # Map flat index -> compressed row index in Aii / Aix  (-1 = not interior)
    # Map flat index -> compressed col index in Aix        (-1 = not boundary)
    row_of = np.full(Nx * Ny, -1, dtype=np.intp)
    col_of_bnd = np.full(Nx * Ny, -1, dtype=np.intp)
    row_of[Ji] = np.arange(Ni,     dtype=np.intp)
    col_of_bnd[Jx] = np.arange(Nx_bnd, dtype=np.intp)

    # Pre-allocate COO arrays with an upper bound of 5 entries per interior row
    max_nnz = 5 * Ni
    ii_row  = np.empty(max_nnz, dtype=np.intp)
    ii_col  = np.empty(max_nnz, dtype=np.intp)
    ii_val  = np.empty(max_nnz, dtype=np.float64)
    ix_row  = np.empty(max_nnz, dtype=np.intp)
    ix_col  = np.empty(max_nnz, dtype=np.intp)
    ix_val  = np.empty(max_nnz, dtype=np.float64)
    nii = 0
    nix = 0

    def _add(flat_row, flat_col, value):
        """Route (flat_row, flat_col, value) to Aii or Aix."""
        nonlocal nii, nix
        r = row_of[flat_row]       # compressed row (always valid: flat_row is interior)
        ci = row_of[flat_col]      # is flat_col interior?
        cx = col_of_bnd[flat_col]  # is flat_col boundary?
        if ci >= 0:
            ii_row[nii] = r;  ii_col[nii] = ci;  ii_val[nii] = value;  nii += 1
        elif cx >= 0:
            ix_row[nix] = r;  ix_col[nix] = cx;  ix_val[nix] = value;  nix += 1
        # neighbours that fall completely outside the grid are ignored

    for ix in range(Nx):
        for iy in range(Ny):
            k = ix * Ny + iy
            if row_of[k] < 0:       # boundary point: not a row of Aii/Aix
                continue

            # Coefficients are used exactly as supplied (no sign flip):
            # this assembles  +c11 u_xx + c22 u_yy [+ c1 u_y] [+ c u],
            # i.e. a Laplacian (Delta), not -Delta.
            # Standard 2nd difference: diag = -2/h^2,  off-diag = +1/h^2.
            diag_val  = -c11[k] * 2.0 * ax - c22[k] * 2.0 * ay
            if c0  is not None: diag_val += c0[k]

            _add(k, k, diag_val)

            # x-neighbours  (stride Ny in flat indexing)
            if ix > 0:
                _add(k, k - Ny, c11[k] * ax)
            if ix < Nx - 1:
                _add(k, k + Ny, c11[k] * ax)

            # y-neighbours  (stride 1)
            if iy > 0:
                _add(k, k - 1, c22[k] * ay)
            if iy < Ny - 1:
                _add(k, k + 1, c22[k] * ay)

            # first-order y-derivative  (c1 * u_y)  – backward difference
            # (u_i - u_{i-1})/h : diag += c1/h, (k-1) += -c1/h
            if c1 is not None:
                if iy > 0:
                    _add(k, k - 1, -c1[k] / hy)
                _add(k, k,          c1[k] / hy)

    Aii = sparse.csr_matrix(
        (ii_val[:nii], (ii_row[:nii], ii_col[:nii])), shape=(Ni, Ni))
    Aix = sparse.csr_matrix(
        (ix_val[:nix], (ix_row[:nix], ix_col[:nix])), shape=(Ni, Nx_bnd))
    return Aii, Aix


# ---------------------------------------------------------------------------
# Direct CSR assembly – 7-point stencil (3-D)
# ---------------------------------------------------------------------------
# Operator:  L u = c11 u_xx + c22 u_yy + c33 u_zz [+ c u]   (coeffs as given)
#
# Flat index:  k = ix * Ny*Nz + iy * Nz + iz
#
# Interior row contributions:
#   diag  += -(c11[k]*2/hx^2 + c22[k]*2/hy^2 + c33[k]*2/hz^2)  [+ c[k]]
#   (k, k ± Ny*Nz) +=  c11[k] / hx^2    (x neighbours, stride Ny*Nz)
#   (k, k ±    Nz) +=  c22[k] / hy^2    (y neighbours, stride Nz)
#   (k, k ±     1) +=  c33[k] / hz^2    (z neighbours, stride 1)
# ---------------------------------------------------------------------------

def _constructPDO3D_csr(op, xpts, ypts, zpts, XX,
                        Ji, Jx, Ni, Nx_bnd):
    """
    Build Aii (Ni×Ni) and Aix (Ni×Nx_bnd) directly in CSR format.

    Parameters
    ----------
    op              : PDO object with fields c11, c22, c33
    xpts,ypts,zpts  : 1-D coordinate arrays  (lengths Nx, Ny, Nz)
    XX              : (Nx*Ny*Nz, 3) array of all grid points
    Ji, Jx          : interior / boundary flat indices
    Ni, Nx_bnd      : len(Ji), len(Jx)
    """
    Nx, Ny, Nz = len(xpts), len(ypts), len(zpts)
    hx = xpts[1] - xpts[0]
    hy = ypts[1] - ypts[0]
    hz = zpts[1] - zpts[0]
    ax = 1.0 / hx**2
    ay = 1.0 / hy**2
    az = 1.0 / hz**2

    c11 = op.c11(XX)
    c22 = op.c22(XX)
    c33 = op.c33(XX)
    # Zeroth-order term  c(x)*u  -- ESSENTIAL for Helmholtz, where c = -k^2.
    # Omitting it silently reduces the operator to Poisson (-lap u = f) and the
    # solve converges to the wrong PDE (the residual stalls under refinement).
    c0  = op.c(XX)  if op.c  else None

    # Compressed index maps
    row_of     = np.full(Nx * Ny * Nz, -1, dtype=np.intp)
    col_of_bnd = np.full(Nx * Ny * Nz, -1, dtype=np.intp)
    row_of[Ji]  = np.arange(Ni,      dtype=np.intp)
    col_of_bnd[Jx] = np.arange(Nx_bnd, dtype=np.intp)

    # Upper bound: 7 entries per interior row
    max_nnz = 7 * Ni
    ii_row = np.empty(max_nnz, dtype=np.intp)
    ii_col = np.empty(max_nnz, dtype=np.intp)
    ii_val = np.empty(max_nnz, dtype=np.float64)
    ix_row = np.empty(max_nnz, dtype=np.intp)
    ix_col = np.empty(max_nnz, dtype=np.intp)
    ix_val = np.empty(max_nnz, dtype=np.float64)
    nii = 0
    nix = 0

    sy = Nz        # stride in y-direction
    sx = Ny * Nz   # stride in x-direction

    def _add(flat_row, flat_col, value):
        nonlocal nii, nix
        r  = row_of[flat_row]
        ci = row_of[flat_col]
        cx = col_of_bnd[flat_col]
        if ci >= 0:
            ii_row[nii] = r;  ii_col[nii] = ci;  ii_val[nii] = value;  nii += 1
        elif cx >= 0:
            ix_row[nix] = r;  ix_col[nix] = cx;  ix_val[nix] = value;  nix += 1

    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                k = ix * sx + iy * sy + iz
                if row_of[k] < 0:
                    continue

                # Coefficients used exactly as supplied (no sign flip):
                # assembles +c11 u_xx + c22 u_yy + c33 u_zz [+ c u] (Delta).
                # Standard 2nd difference: diag = -2/h^2, off-diag = +1/h^2.
                diag_val = -(c11[k] * 2.0 * ax
                           + c22[k] * 2.0 * ay
                           + c33[k] * 2.0 * az)
                if c0 is not None:
                    diag_val += c0[k]          # Helmholtz: c0 = -k^2 (as given)
                _add(k, k, diag_val)

                if ix > 0:        _add(k, k - sx, c11[k] * ax)
                if ix < Nx - 1:   _add(k, k + sx, c11[k] * ax)
                if iy > 0:        _add(k, k - sy, c22[k] * ay)
                if iy < Ny - 1:   _add(k, k + sy, c22[k] * ay)
                if iz > 0:        _add(k, k - 1,  c33[k] * az)
                if iz < Nz - 1:   _add(k, k + 1,  c33[k] * az)

    Aii = sparse.csr_matrix(
        (ii_val[:nii], (ii_row[:nii], ii_col[:nii])), shape=(Ni, Ni))
    Aix = sparse.csr_matrix(
        (ix_val[:nix], (ix_row[:nix], ix_col[:nix])), shape=(Ni, Nx_bnd))
    return Aii, Aix


# ---------------------------------------------------------------------------
# Stencil solver
# ---------------------------------------------------------------------------

class stencilSolver(AbstractPDESolver):

    def __init__(self, pdo, geom, ord):
        """
        Initializes the stencil solver with domain
        information and discretization parameters.

        Parameters
        ----------
        pdo  : partial differential operator object
        geom : computational domain (must expose .bounds)
        ord  : list[int] – number of grid points in each direction
        """
        self._box_geom = geom.bounds
        ndim           = self._box_geom.shape[-1]
        self._geom     = geom

        if ndim == 2:
            Nx, Ny = ord[0], ord[1]
            xpts   = np.linspace(self._box_geom[0][0], self._box_geom[1][0], Nx)
            ypts   = np.linspace(self._box_geom[0][1], self._box_geom[1][1], Ny)

            self._XX = np.empty((Nx * Ny, 2))
            self._XX[:, 0] = np.kron(xpts, np.ones(Ny))
            self._XX[:, 1] = np.kron(np.ones(Nx), ypts)

            self._Ji = np.where(
                (self._XX[:, 0] > self._box_geom[0][0]) &
                (self._XX[:, 0] < self._box_geom[1][0]) &
                (self._XX[:, 1] > self._box_geom[0][1]) &
                (self._XX[:, 1] < self._box_geom[1][1])
            )[0]
            self._Jx = np.where(
                (self._XX[:, 0] == self._box_geom[0][0]) |
                (self._XX[:, 0] == self._box_geom[1][0]) |
                (self._XX[:, 1] == self._box_geom[0][1]) |
                (self._XX[:, 1] == self._box_geom[1][1])
            )[0]

            Ni, Nx_bnd = len(self._Ji), len(self._Jx)
            self._Aii, self._Aix = _constructPDO2D_csr(
                pdo, xpts, ypts, self._XX,
                self._Ji, self._Jx, Ni, Nx_bnd)

            # Axi / Axx: boundary rows from the full operator are identity,
            # but callers may need the off-diagonal coupling. Build lazily
            # from the full matrix only when needed; set to None for now.
            # (Uncomment the two lines below if Axi/Axx are required.)
            # self._Axi, self._Axx = _constructPDO2D_csr(
            #     pdo, xpts, ypts, self._XX, self._Jx, self._Ji, ...)
            self._Axi = None
            self._Axx = None

        elif ndim == 3:
            Nx, Ny, Nz = ord[0], ord[1], ord[2]
            xpts = np.linspace(self._box_geom[0][0], self._box_geom[1][0], Nx)
            ypts = np.linspace(self._box_geom[0][1], self._box_geom[1][1], Ny)
            zpts = np.linspace(self._box_geom[0][2], self._box_geom[1][2], Nz)

            self._XX = np.empty((Nx * Ny * Nz, 3))
            self._XX[:, 0] = np.kron(np.kron(xpts, np.ones(Ny)), np.ones(Nz))
            self._XX[:, 1] = np.kron(np.kron(np.ones(Nx), ypts), np.ones(Nz))
            self._XX[:, 2] = np.kron(np.kron(np.ones(Nx), np.ones(Ny)), zpts)

            self._Ji = np.where(
                (self._XX[:, 0] > self._box_geom[0][0]) &
                (self._XX[:, 0] < self._box_geom[1][0]) &
                (self._XX[:, 1] > self._box_geom[0][1]) &
                (self._XX[:, 1] < self._box_geom[1][1]) &
                (self._XX[:, 2] > self._box_geom[0][2]) &
                (self._XX[:, 2] < self._box_geom[1][2])
            )[0]
            self._Jx = np.where(
                (self._XX[:, 0] == self._box_geom[0][0]) |
                (self._XX[:, 0] == self._box_geom[1][0]) |
                (self._XX[:, 1] == self._box_geom[0][1]) |
                (self._XX[:, 1] == self._box_geom[1][1]) |
                (self._XX[:, 2] == self._box_geom[0][2]) |
                (self._XX[:, 2] == self._box_geom[1][2])
            )[0]

            Ni, Nx_bnd = len(self._Ji), len(self._Jx)
            self._Aii, self._Aix = _constructPDO3D_csr(
                pdo, xpts, ypts, zpts, self._XX,
                self._Ji, self._Jx, Ni, Nx_bnd)
            self._Axi = None
            self._Axx = None

        else:
            raise ValueError(f"Unsupported spatial dimension: {ndim}")

        self._XXi = self._XX[self._Ji, :]
        self._XXb = self._XX[self._Jx, :]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def npoints_dim(self):
        return self.npan_dim * self.p

    @property
    def geom(self):
        return self._geom

    @property
    def XX(self):
        return self._XX

    @property
    def XXi(self):
        return self._XXi

    @property
    def XXb(self):
        return self._XXb

    @property
    def Ji(self):
        return self._Ji

    @property
    def Jx(self):
        return self._Jx

    @property
    def Aii(self):
        return self._Aii

    @property
    def Aix(self):
        return self._Aix

    @property
    def Axi(self):
        return self._Axi

    @property
    def Axx(self):
        return self._Axx

    @property
    def p(self):
        return self._p