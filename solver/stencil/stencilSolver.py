import numpy as np
from scipy.sparse.linalg import LinearOperator, splu
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

    # Interior-row COO buffers  (Aii, Aix): <= 5 entries per interior row
    max_nnz = 5 * Ni
    ii_row  = np.empty(max_nnz, dtype=np.intp)
    ii_col  = np.empty(max_nnz, dtype=np.intp)
    ii_val  = np.empty(max_nnz, dtype=np.float64)
    ix_row  = np.empty(max_nnz, dtype=np.intp)
    ix_col  = np.empty(max_nnz, dtype=np.intp)
    ix_val  = np.empty(max_nnz, dtype=np.float64)
    nii = 0
    nix = 0

    # Boundary-row COO buffers  (Axi, Axx): conormal-derivative rows.
    xi_max = 3 * max(Nx_bnd, 1)
    xx_max = 20 * max(Nx_bnd, 1)
    xi_row = np.empty(xi_max, dtype=np.intp)
    xi_col = np.empty(xi_max, dtype=np.intp)
    xi_val = np.empty(xi_max, dtype=np.float64)
    xx_row = np.empty(xx_max, dtype=np.intp)
    xx_col = np.empty(xx_max, dtype=np.intp)
    xx_val = np.empty(xx_max, dtype=np.float64)
    nxi = 0
    nxx = 0

    def _add(flat_row, flat_col, value):
        """Route an interior-row entry to Aii (col interior) or Aix (col bnd)."""
        nonlocal nii, nix
        r = row_of[flat_row]       # compressed row (always valid: flat_row is interior)
        ci = row_of[flat_col]      # is flat_col interior?
        cx = col_of_bnd[flat_col]  # is flat_col boundary?
        if ci >= 0:
            ii_row[nii] = r;  ii_col[nii] = ci;  ii_val[nii] = value;  nii += 1
        elif cx >= 0:
            ix_row[nix] = r;  ix_col[nix] = cx;  ix_val[nix] = value;  nix += 1
        # neighbours that fall completely outside the grid are ignored

    def _add_bnd(flat_brow, flat_col, value):
        """Route a boundary (conormal) row entry to Axi (col interior) or Axx (col bnd)."""
        nonlocal nxi, nxx
        br = col_of_bnd[flat_brow]  # compressed bnd row (always valid: flat_brow is bnd)
        ci = row_of[flat_col]       # is flat_col interior?
        cx = col_of_bnd[flat_col]   # is flat_col boundary?
        if ci >= 0:
            xi_row[nxi] = br;  xi_col[nxi] = ci;  xi_val[nxi] = value;  nxi += 1
        elif cx >= 0:
            xx_row[nxx] = br;  xx_col[nxx] = cx;  xx_val[nxx] = value;  nxx += 1
        # neighbours that fall completely outside the grid are ignored

    # Per-axis data:  (stride, h, coeff array, length, index-getter)
    #   axis 0 = x (stride Ny),  axis 1 = y (stride 1)
    for ix in range(Nx):
        for iy in range(Ny):
            k = ix * Ny + iy

            if row_of[k] >= 0:
                # ---------- interior row: full stencil (Aii / Aix) ----------
                diag_val  = -c11[k] * 2.0 * ax - c22[k] * 2.0 * ay
                if c0 is not None: diag_val += c0[k]
                _add(k, k, diag_val)

                if ix > 0:        _add(k, k - Ny, c11[k] * ax)
                if ix < Nx - 1:   _add(k, k + Ny, c11[k] * ax)
                if iy > 0:        _add(k, k - 1,  c22[k] * ay)
                if iy < Ny - 1:   _add(k, k + 1,  c22[k] * ay)

                # first-order y-derivative (c1 u_y) – backward difference
                if c1 is not None:
                    if iy > 0:    _add(k, k - 1, -c1[k] / hy)
                    _add(k, k,                    c1[k] / hy)
                continue

            # ---------- boundary row: O(h^2) conormal derivative ----------
            # For each face this node lies on (axis a, outward normal), use a
            # one-sided normal flux plus a half-cell (h_a/2) correction in the
            # tangential second differences and the zeroth-order term.  The
            # ghost outside the box is eliminated through the node's own PDE row.
            # (A first-order term c1, if present, is kept in the interior solve
            # but its O(h) contribution to the boundary flux is not added here;
            # for c1 = None the conormal rows are exact to O(h^2).)
            axes = ((Ny, hx, c11, Nx, ix),
                    (1,  hy, c22, Ny, iy))
            for a in range(2):
                s_a, h_a, c_a, N_a, i_a = axes[a]
                on_low  = (i_a == 0)
                on_high = (i_a == N_a - 1)
                if not (on_low or on_high):
                    continue
                k_in = k - s_a if on_high else k + s_a   # interior normal neighbour
                # one-sided normal flux  (c_a/h_a)(u_b - u_in)
                _add_bnd(k, k,    c_a[k] / h_a)
                _add_bnd(k, k_in, -c_a[k] / h_a)
                # half-cell tangential second differences
                for t in range(2):
                    if t == a:
                        continue
                    s_t, h_t, c_t, N_t, i_t = axes[t]
                    coef = 0.5 * h_a * c_t[k] / h_t**2
                    _add_bnd(k, k, 2.0 * coef)                       # +h_a c_t/h_t^2
                    if i_t > 0:        _add_bnd(k, k - s_t, -coef)
                    if i_t < N_t - 1:  _add_bnd(k, k + s_t, -coef)
                # half-cell zeroth-order term
                if c0 is not None:
                    _add_bnd(k, k, -0.5 * h_a * c0[k])

    Aii = sparse.csr_matrix(
        (ii_val[:nii], (ii_row[:nii], ii_col[:nii])), shape=(Ni, Ni))
    Aix = sparse.csr_matrix(
        (ix_val[:nix], (ix_row[:nix], ix_col[:nix])), shape=(Ni, Nx_bnd))
    Axi = sparse.csr_matrix(
        (xi_val[:nxi], (xi_row[:nxi], xi_col[:nxi])), shape=(Nx_bnd, Ni))
    Axx = sparse.csr_matrix(
        (xx_val[:nxx], (xx_row[:nxx], xx_col[:nxx])), shape=(Nx_bnd, Nx_bnd))
    return Aii, Aix, Axi, Axx


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

    # Interior-row COO buffers (Aii, Aix): <= 7 entries per interior row
    max_nnz = 7 * Ni
    ii_row = np.empty(max_nnz, dtype=np.intp)
    ii_col = np.empty(max_nnz, dtype=np.intp)
    ii_val = np.empty(max_nnz, dtype=np.float64)
    ix_row = np.empty(max_nnz, dtype=np.intp)
    ix_col = np.empty(max_nnz, dtype=np.intp)
    ix_val = np.empty(max_nnz, dtype=np.float64)
    nii = 0
    nix = 0

    # Boundary-row COO buffers (Axi, Axx): conormal-derivative rows.
    xi_max = 4 * max(Nx_bnd, 1)
    xx_max = 30 * max(Nx_bnd, 1)
    xi_row = np.empty(xi_max, dtype=np.intp)
    xi_col = np.empty(xi_max, dtype=np.intp)
    xi_val = np.empty(xi_max, dtype=np.float64)
    xx_row = np.empty(xx_max, dtype=np.intp)
    xx_col = np.empty(xx_max, dtype=np.intp)
    xx_val = np.empty(xx_max, dtype=np.float64)
    nxi = 0
    nxx = 0

    sy = Nz        # stride in y-direction
    sx = Ny * Nz   # stride in x-direction

    def _add(flat_row, flat_col, value):
        """Interior-row entry -> Aii (col interior) or Aix (col boundary)."""
        nonlocal nii, nix
        r  = row_of[flat_row]
        ci = row_of[flat_col]
        cx = col_of_bnd[flat_col]
        if ci >= 0:
            ii_row[nii] = r;  ii_col[nii] = ci;  ii_val[nii] = value;  nii += 1
        elif cx >= 0:
            ix_row[nix] = r;  ix_col[nix] = cx;  ix_val[nix] = value;  nix += 1

    def _add_bnd(flat_brow, flat_col, value):
        """Boundary (conormal) row entry -> Axi (col interior) or Axx (col bnd)."""
        nonlocal nxi, nxx
        br = col_of_bnd[flat_brow]
        ci = row_of[flat_col]
        cx = col_of_bnd[flat_col]
        if ci >= 0:
            xi_row[nxi] = br;  xi_col[nxi] = ci;  xi_val[nxi] = value;  nxi += 1
        elif cx >= 0:
            xx_row[nxx] = br;  xx_col[nxx] = cx;  xx_val[nxx] = value;  nxx += 1

    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                k = ix * sx + iy * sy + iz

                if row_of[k] >= 0:
                    # ---------- interior row: full 7-point stencil ----------
                    diag_val = -(c11[k] * 2.0 * ax
                               + c22[k] * 2.0 * ay
                               + c33[k] * 2.0 * az)
                    if c0 is not None:
                        diag_val += c0[k]      # Helmholtz: c0 = -k^2 (as given)
                    _add(k, k, diag_val)

                    if ix > 0:        _add(k, k - sx, c11[k] * ax)
                    if ix < Nx - 1:   _add(k, k + sx, c11[k] * ax)
                    if iy > 0:        _add(k, k - sy, c22[k] * ay)
                    if iy < Ny - 1:   _add(k, k + sy, c22[k] * ay)
                    if iz > 0:        _add(k, k - 1,  c33[k] * az)
                    if iz < Nz - 1:   _add(k, k + 1,  c33[k] * az)
                    continue

                # ---------- boundary row: O(h^2) conormal derivative ----------
                # For every face the node lies on (axis a, outward normal): a
                # one-sided normal flux  (c_a/h_a)(u_b - u_in)  plus a half-cell
                # (h_a/2) correction in the two tangential second differences and
                # the zeroth-order term.  The exterior ghost is eliminated via the
                # node's own PDE row -> second-order conormal derivative.  Nodes on
                # an edge/corner accumulate one such contribution per incident face.
                axes = ((sx, hx, c11, Nx, ix),
                        (sy, hy, c22, Ny, iy),
                        (1,  hz, c33, Nz, iz))
                for a in range(3):
                    s_a, h_a, c_a, N_a, i_a = axes[a]
                    on_low  = (i_a == 0)
                    on_high = (i_a == N_a - 1)
                    if not (on_low or on_high):
                        continue
                    k_in = k - s_a if on_high else k + s_a   # interior normal nbr
                    _add_bnd(k, k,    c_a[k] / h_a)
                    _add_bnd(k, k_in, -c_a[k] / h_a)
                    for t in range(3):
                        if t == a:
                            continue
                        s_t, h_t, c_t, N_t, i_t = axes[t]
                        coef = 0.5 * h_a * c_t[k] / h_t**2
                        _add_bnd(k, k, 2.0 * coef)              # +h_a c_t/h_t^2
                        if i_t > 0:        _add_bnd(k, k - s_t, -coef)
                        if i_t < N_t - 1:  _add_bnd(k, k + s_t, -coef)
                    if c0 is not None:
                        _add_bnd(k, k, -0.5 * h_a * c0[k])

    Aii = sparse.csr_matrix(
        (ii_val[:nii], (ii_row[:nii], ii_col[:nii])), shape=(Ni, Ni))
    Aix = sparse.csr_matrix(
        (ix_val[:nix], (ix_row[:nix], ix_col[:nix])), shape=(Ni, Nx_bnd))
    Axi = sparse.csr_matrix(
        (xi_val[:nxi], (xi_row[:nxi], xi_col[:nxi])), shape=(Nx_bnd, Ni))
    Axx = sparse.csr_matrix(
        (xx_val[:nxx], (xx_row[:nxx], xx_col[:nxx])), shape=(Nx_bnd, Nx_bnd))
    return Aii, Aix, Axi, Axx


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
            self._Aii, self._Aix, self._Axi, self._Axx = _constructPDO2D_csr(
                pdo, xpts, ypts, self._XX,
                self._Ji, self._Jx, Ni, Nx_bnd)

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
            self._Aii, self._Aix, self._Axi, self._Axx = _constructPDO3D_csr(
                pdo, xpts, ypts, zpts, self._XX,
                self._Ji, self._Jx, Ni, Nx_bnd)

        else:
            raise ValueError(f"Unsupported spatial dimension: {ndim}")

        self._XXi = self._XX[self._Ji, :]
        self._XXb = self._XX[self._Jx, :]
        self._solver_Aii = None        # lazily-built Aii^{-1} operator

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
    def solver_Aii(self):
        """Aii^{-1} as a LinearOperator (lazy LU).  Supports @, .T and 2-D rhs."""
        if self._solver_Aii is None:
            lu = splu(self._Aii.tocsc())
            N  = self._Aii.shape[0]
            self._solver_Aii = LinearOperator(
                (N, N), dtype=np.float64,
                matvec  = lu.solve,
                matmat  = lu.solve,
                rmatvec = lambda b: lu.solve(b, trans='T'),
                rmatmat = lambda b: lu.solve(b, trans='T'))
        return self._solver_Aii

    def schur_complement(self):
        """
        Local Schur complement  S = Axx - Axi @ Aii^{-1} @ Aix  as a matrix-free
        LinearOperator (Nx_bnd x Nx_bnd).  Applied to a boundary trace it returns
        the (homogeneous-interior) O(h^2) conormal derivative on the boundary.
        Volume-source contributions to the flux are handled separately on the rhs.
        """
        Aii_inv = self.solver_Aii
        Aix, Axi, Axx = self._Aix, self._Axi, self._Axx
        n = Axx.shape[0]

        def _mv(x):
            return Axx @ x - Axi @ (Aii_inv @ (Aix @ x))

        def _rmv(x):
            return Axx.T @ x - Aix.T @ (Aii_inv.T @ (Axi.T @ x))

        return LinearOperator((n, n), matvec=_mv, matmat=_mv,
                              rmatvec=_rmv, rmatmat=_rmv, dtype=np.float64)

    @property
    def p(self):
        return self._p