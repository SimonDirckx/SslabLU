"""
SOMS3D_legendre_v2.py

Option B implementation: every face of the merged box-pair is covered by a
single physical Legendre face (one per face of the merged-pair surface).
Lifting maps p Legendre nodes -> p Cheb-with-endpoints nodes per single
sub-box axis, and 2p Legendre nodes -> (p_joined+1) Cheb-with-endpoints
nodes per joined axis.

Pipeline per box-pair, direction `dir`:
  1. Leg face data (p x p per single, 2p x p on side faces of merged pair)
  2. Lift Leg -> Cheb-with-endpoints on each of 6 surrounding faces; each
     boundary node of the merged-box-pair is owned by exactly one Legendre
     face by deterministic partition (no averaging).
  3. Cheb merged-box solve via static condensation.
  4. Extract the FULL centerline-face trace (p x p Cheb-with-endpoints,
     including its edges/corners which sit on Ib_dir). Project to p x p
     Legendre on the centerline face.

Convention: `p` = number of points (Legendre OR Chebyshev). Polynomial
order is p-1, basis dim is p.
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import numpy.polynomial.chebyshev as chebpoly


# ---------------------------------------------------------------------------
# Cheb differentiation matrix
# ---------------------------------------------------------------------------

def _cheb_diff(p_minus_1):
    """Chebyshev differentiation matrix on `p_minus_1 + 1` Cheb nodes on [-1, 1]."""
    n = p_minus_1
    if n == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1); c[0] = 2.0; c[-1] = 2.0
    c *= (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(np.sum(D, axis=1))
    return D, x


def _cheb_1d(p, scl):
    """
    p Cheb-with-endpoints nodes on [0, scl] (in increasing order) and
    associated differentiation matrix.
    """
    D, pts = _cheb_diff(p - 1)
    D = -2 * D / scl
    pts = ((pts[::-1] + 1) / 2) * scl
    return D, pts


def _joined_cheb_1d(p, scl):
    """
    Joined-direction Cheb-with-endpoints grid spanning [0, 2*scl].

    The number of joined points is odd (so the centerline scl is a node).
    Returns (D, pts, pj) where pj = number of points (odd, ~3p/2).
    """
    # Want pj = 3p/2 made odd. Equivalently pj-1 = 3p/2 - 1 made even,
    # so use the existing formula pj-1 = (3p//2) made even, pj = that+1.
    pjm1 = (3 * p) // 2
    pjm1 -= pjm1 % 2                # ensure pjm1 is even -> pj is odd
    pj = pjm1 + 1
    D, pts = _cheb_diff(pjm1)
    pts = (1 + pts[::-1]) * scl     # maps to [0, 2*scl]
    D = -D / scl
    return D, pts, pj


def _legendre_1d(p, scl):
    """p open Gauss-Legendre nodes and weights on (0, scl), increasing order."""
    pts, wts = np.polynomial.legendre.leggauss(p)
    pts = (pts + 1) / 2 * scl
    wts = wts * scl / 2
    return pts, wts


def _svd_interp(E_src, E_tgt):
    """E_tgt @ pinv(E_src) via truncated SVD."""
    U, s, V = np.linalg.svd(E_src, full_matrices=False)
    k = np.sum(s > 1e-15 * s[0])
    Uk, Vk, sk = U[:, :k], V[:k, :].T, s[:k]
    return (E_tgt @ Vk) @ np.diag(sk ** -1) @ Uk.T


# ---------------------------------------------------------------------------
# 1D Leg <-> Cheb-full interpolators (basis dim = p)
# ---------------------------------------------------------------------------

def _leg_to_chebfull_1d(p, scl):
    """
    Map p Legendre on (0, scl) -> p Cheb-with-endpoints on [0, scl].
    Shape: (p, p). Exact for polynomials of degree p-1.
    """
    leg_pts = _legendre_1d(p, scl)[0]    # length p
    cheb_pts = _cheb_1d(p, scl)[1]       # length p
    nbasis = p
    leg_ref = 2 * leg_pts / scl - 1
    cheb_ref = 2 * cheb_pts / scl - 1
    V_src = chebpoly.chebvander(leg_ref, nbasis - 1)    # (p, p)
    V_tgt = chebpoly.chebvander(cheb_ref, nbasis - 1)   # (p, p)
    return _svd_interp(V_src, V_tgt)


def _chebfull_to_leg_1d(p, scl):
    """
    Map p Cheb-with-endpoints on [0, scl] -> p Legendre on (0, scl).
    Shape: (p, p). Exact for polynomials of degree p-1.
    """
    cheb_pts = _cheb_1d(p, scl)[1]
    leg_pts = _legendre_1d(p, scl)[0]
    nbasis = p
    cheb_ref = 2 * cheb_pts / scl - 1
    leg_ref = 2 * leg_pts / scl - 1
    V_src = chebpoly.chebvander(cheb_ref, nbasis - 1)
    V_tgt = chebpoly.chebvander(leg_ref, nbasis - 1)
    return _svd_interp(V_src, V_tgt)


def _legstack_to_chebjoinedfull_1d(p, scl):
    """
    Map stacked Legendre (length 2p, on (0, 2*scl)) -> joined Cheb-with-endpoints
    (length pj, on [0, 2*scl]). Shape: (pj, 2p). Exact for polynomials of degree 2p-1.
    """
    leg = _legendre_1d(p, scl)[0]
    leg2 = np.concatenate([leg, scl + leg])                # length 2p
    _, cj_full, pj = _joined_cheb_1d(p, scl)               # length pj
    nbasis = 2 * p
    leg2_ref = leg2 / scl - 1                              # in [-1, 1] on [0, 2*scl]
    cj_ref = cj_full / scl - 1
    V_src = chebpoly.chebvander(leg2_ref, nbasis - 1)
    V_tgt = chebpoly.chebvander(cj_ref, nbasis - 1)
    return _svd_interp(V_src, V_tgt)


def interp_ops(px, py, pz, scl_x, scl_y, scl_z):
    """All 1D interpolation operators packaged."""
    ops = {}
    for axis, p, scl in [('x', px, scl_x), ('y', py, scl_y), ('z', pz, scl_z)]:
        ops[f'L2C_{axis}'] = _leg_to_chebfull_1d(p, scl)
        ops[f'C2L_{axis}'] = _chebfull_to_leg_1d(p, scl)
        ops[f'LS2CJ_{axis}'] = _legstack_to_chebjoinedfull_1d(p, scl)
    return ops


# ---------------------------------------------------------------------------
# Merged-box operator and disjoint face partition
# ---------------------------------------------------------------------------

def L_op(dir, px, py, pz, scl_x, scl_y, scl_z, coeffs):
    """
    Build the merged-box operator and a disjoint partition of its boundary
    Ib_dir into 6 face-blocks.

    The PDE operator is the general 2nd-order non-divergence form

        L = sum_{i,j} c_{ij}(x) d_i d_j  +  sum_i c_i(x) d_i  +  c(x) I

    with axis indices i,j in {x, y, z}.

    `coeffs` is a dict with keys
        'c11','c12','c13','c21','c22','c23','c31','c32','c33',
        'c1','c2','c3', 'c'
    Each entry is either a Python scalar (constant coefficient) or a 1D numpy
    array of length njx*njy*njz giving the coefficient at every merged-box
    collocation node (in the natural XY ordering produced inside this function).

    For backward compatibility, Helmholtz (-Delta - k^2) u = 0 corresponds to
        c11 = c22 = c33 = -1,   c = -k^2,   all other coefficients zero.

    Disjoint face-partition convention (each Ib_dir node assigned to exactly
    one face):
      Left  (x = 0)     gets ALL nodes with x = 0
      Right (x = xlim)  gets ALL nodes with x = xlim (excluding x=0 by def)
      Front (y = 0)     gets nodes with y = 0   AND  0 < x < xlim
      Back  (y = ylim)  gets nodes with y = ylim AND  0 < x < xlim
      Down  (z = 0)     gets nodes with z = 0   AND  0 < x < xlim AND 0 < y < ylim
      Up    (z = zlim)  gets nodes with z = zlim AND  0 < x < xlim AND 0 < y < ylim
    """
    Dx, xpts = _cheb_1d(px, scl_x)
    Dy, ypts = _cheb_1d(py, scl_y)
    Dz, zpts = _cheb_1d(pz, scl_z)

    Dx2, xpts2, pj_x = _joined_cheb_1d(px, scl_x)
    Dy2, ypts2, pj_y = _joined_cheb_1d(py, scl_y)
    Dz2, zpts2, pj_z = _joined_cheb_1d(pz, scl_z)

    if dir == 0:
        Dx = Dx2;   xj = xpts2;   xlim, ylim, zlim = 2 * scl_x, scl_y, scl_z
        yj, zj = ypts, zpts
        xc = scl_x
    elif dir == 1:
        Dy = Dy2;   yj = ypts2;   xlim, ylim, zlim = scl_x, 2 * scl_y, scl_z
        xj, zj = xpts, zpts
        xc = scl_y
    else:
        Dz = Dz2;   zj = zpts2;   xlim, ylim, zlim = scl_x, scl_y, 2 * scl_z
        xj, yj = xpts, ypts
        xc = scl_z

    njx, njy, njz = len(xj), len(yj), len(zj)
    ones_x, ones_y, ones_z = np.ones(njx), np.ones(njy), np.ones(njz)

    XY = np.column_stack([
        np.kron(np.kron(xj, ones_y), ones_z),
        np.kron(np.kron(ones_x, yj), ones_z),
        np.kron(np.kron(ones_x, ones_y), zj),
    ])
    x, y, z = XY[:, 0], XY[:, 1], XY[:, 2]

    on_xlo = np.abs(x) < 1e-10
    on_xhi = np.abs(x - xlim) < 1e-10
    on_ylo = np.abs(y) < 1e-10
    on_yhi = np.abs(y - ylim) < 1e-10
    on_zlo = np.abs(z) < 1e-10
    on_zhi = np.abs(z - zlim) < 1e-10

    Ii_dir = np.where(~(on_xlo | on_xhi | on_ylo | on_yhi | on_zlo | on_zhi))[0]
    Ib_dir = np.where(on_xlo | on_xhi | on_ylo | on_yhi | on_zlo | on_zhi)[0]

    is_left_full  = on_xlo
    is_right_full = on_xhi & ~is_left_full
    is_front_full = on_ylo & ~is_left_full & ~is_right_full
    is_back_full  = on_yhi & ~is_left_full & ~is_right_full
    is_down_full  = on_zlo & ~is_left_full & ~is_right_full & ~is_front_full & ~is_back_full
    is_up_full    = on_zhi & ~is_left_full & ~is_right_full & ~is_front_full & ~is_back_full

    in_Ib = lambda flag: np.where(flag[Ib_dir])[0]

    Il_dir = in_Ib(is_left_full)
    Ir_dir = in_Ib(is_right_full)
    If_dir = in_Ib(is_front_full)
    Ibk_dir = in_Ib(is_back_full)
    Id_dir = in_Ib(is_down_full)
    Iu_dir = in_Ib(is_up_full)

    total = len(Il_dir) + len(Ir_dir) + len(If_dir) + len(Ibk_dir) + len(Id_dir) + len(Iu_dir)
    assert total == len(Ib_dir), \
        f"L_op({dir}): partition leaks ({total} != {len(Ib_dir)})"

    Ic_dir = np.where(np.abs(XY[Ii_dir, dir] - xc) < 1e-10)[0]

    on_center = np.abs(XY[:, dir] - xc) < 1e-10
    Ic_edge = np.where(on_center[Ib_dir])[0]

    Ibox = np.concatenate([Il_dir, If_dir, Id_dir, Iu_dir, Ibk_dir, Ir_dir])
    face_lengths = [len(Il_dir), len(If_dir), len(Id_dir), len(Iu_dir), len(Ibk_dir), len(Ir_dir)]
    cumlen = np.concatenate([[0], np.cumsum(face_lengths)])
    Ijl, Ijf, Ijd, Iju, Ijb, Ijr = [np.arange(cumlen[i], cumlen[i + 1]) for i in range(6)]

    # ----- General-coefficient operator assembly -----
    # 3D first-derivative operators (full N x N where N = njx*njy*njz)
    Ix, Iy, Iz = np.eye(njx), np.eye(njy), np.eye(njz)
    Dx_3d = np.kron(np.kron(Dx, Iy), Iz)
    Dy_3d = np.kron(np.kron(Ix, Dy), Iz)
    Dz_3d = np.kron(np.kron(Ix, Iy), Dz)
    N = njx * njy * njz

    def _broadcast(coef_value):
        """Return a length-N array regardless of whether coef_value is a scalar or array."""
        if np.isscalar(coef_value):
            return np.full(N, float(coef_value))
        arr = np.asarray(coef_value, dtype=float)
        assert arr.shape == (N,), \
            f"coefficient array shape {arr.shape} != expected ({N},)"
        return arr

    # Accumulate L in place: for each nonzero coefficient, add diag(c) @ D
    L = np.zeros((N, N))

    # First-order terms
    for key, D_3d in [('c1', Dx_3d), ('c2', Dy_3d), ('c3', Dz_3d)]:
        if key in coeffs:
            c = _broadcast(coeffs[key])
            if not np.all(c == 0.0):
                L += c[:, None] * D_3d   # diag(c) @ D == elementwise scale of rows

    # Second-order terms (9 pairs).
    # We build mixed derivatives Dij = Di @ Dj as needed.
    second_pairs = [
        ('c11', Dx_3d, Dx_3d),
        ('c12', Dx_3d, Dy_3d),
        ('c13', Dx_3d, Dz_3d),
        ('c21', Dy_3d, Dx_3d),
        ('c22', Dy_3d, Dy_3d),
        ('c23', Dy_3d, Dz_3d),
        ('c31', Dz_3d, Dx_3d),
        ('c32', Dz_3d, Dy_3d),
        ('c33', Dz_3d, Dz_3d),
    ]
    for key, Di, Dj in second_pairs:
        if key in coeffs:
            c = _broadcast(coeffs[key])
            if not np.all(c == 0.0):
                Dij = Di @ Dj
                L += c[:, None] * Dij

    # Zero-order term
    if 'c' in coeffs:
        c0 = _broadcast(coeffs['c'])
        if not np.all(c0 == 0.0):
            # diag(c0) added to L
            idx = np.arange(N)
            L[idx, idx] += c0

    return {
        'L': L, 'XY': XY, 'Ii_dir': Ii_dir, 'Ib_dir': Ib_dir,
        'Ibox': Ibox, 'Ic_dir': Ic_dir, 'Ic_edge': Ic_edge,
        'Ijl': Ijl, 'Ijf': Ijf, 'Ijd': Ijd, 'Iju': Iju, 'Ijb': Ijb, 'Ijr': Ijr,
        'Il_dir': Il_dir, 'Ir_dir': Ir_dir, 'If_dir': If_dir, 'Ibk_dir': Ibk_dir,
        'Id_dir': Id_dir, 'Iu_dir': Iu_dir,
        'pj_x': pj_x, 'pj_y': pj_y, 'pj_z': pj_z,
        'njx': njx, 'njy': njy, 'njz': njz,
        'xlim': xlim, 'ylim': ylim, 'zlim': zlim, 'xc': xc,
    }


# ---------------------------------------------------------------------------
# Per-direction precomputation: geometry, differential operators, lift / project.
# These depend ONLY on dir, p*, scl* — NOT on PDE coefficients. Compute once
# per direction, reuse for every interior centerline face of that direction.
# ---------------------------------------------------------------------------

def precompute_geometry(dir, px, py, pz, scl_x, scl_y, scl_z):
    """
    Build the merged-box collocation grid in local coords and the disjoint
    boundary face partition. Returns a tuple of pure-geometry arrays.
    """
    # Call L_op with all-zero coeffs just to recover all the index sets and XY.
    # (Cheaper than duplicating the partition logic.)
    info = L_op(dir, px, py, pz, scl_x, scl_y, scl_z, {})
    XY = info['XY']
    Ii_dir = info['Ii_dir']
    Ib_dir = info['Ib_dir']
    Ibox = info['Ibox']
    Ic_dir = info['Ic_dir']
    Ic_edge = info['Ic_edge']

    Ibox_inv = np.empty(len(Ibox), dtype=np.int64)
    Ibox_inv[Ibox] = np.arange(len(Ibox))

    cl_xy_idx, cl_source_flag, cl_source_pos = _build_centerline_map(
        dir, XY, info, Ic_dir, Ic_edge
    )

    return (XY, Ii_dir, Ib_dir, Ibox, Ibox_inv,
            Ic_dir, Ic_edge,
            cl_xy_idx, cl_source_flag, cl_source_pos,
            info['Il_dir'], info['If_dir'], info['Id_dir'],
            info['Iu_dir'], info['Ibk_dir'], info['Ir_dir'],
            info['Ijl'], info['Ijf'], info['Ijd'],
            info['Iju'], info['Ijb'], info['Ijr'],
            info['pj_x'], info['pj_y'], info['pj_z'])


def precompute_diffops(dir, px, py, pz, scl_x, scl_y, scl_z, coeffs=None):
    """
    Build the 3D first-derivative operators Dx_3d, Dy_3d, Dz_3d and only the
    mixed second-derivative operators Dij[i][j] = D_i @ D_j that the user's
    `coeffs` dict requires (lazy construction).

    If `coeffs` is None, all 9 Dij entries are built (legacy behaviour).
    Otherwise, only entries (i, j) corresponding to keys 'c{i+1}{j+1}' that
    appear in coeffs (with a non-zero value or a callable) are constructed;
    other slots remain None.

    Returns (Dx_3d, Dy_3d, Dz_3d, Dij) where Dij is a 3x3 list (entries may be None).
    """
    Dx, _ = _cheb_1d(px, scl_x)
    Dy, _ = _cheb_1d(py, scl_y)
    Dz, _ = _cheb_1d(pz, scl_z)
    Dx2, _, _ = _joined_cheb_1d(px, scl_x)
    Dy2, _, _ = _joined_cheb_1d(py, scl_y)
    Dz2, _, _ = _joined_cheb_1d(pz, scl_z)

    if dir == 0:
        Dx = Dx2
    elif dir == 1:
        Dy = Dy2
    else:
        Dz = Dz2

    njx = Dx.shape[0]; njy = Dy.shape[0]; njz = Dz.shape[0]
    Ix = np.eye(njx); Iy = np.eye(njy); Iz = np.eye(njz)
    Dx_3d = np.kron(np.kron(Dx, Iy), Iz)
    Dy_3d = np.kron(np.kron(Ix, Dy), Iz)
    Dz_3d = np.kron(np.kron(Ix, Iy), Dz)

    D = [Dx_3d, Dy_3d, Dz_3d]

    # Determine which (i, j) slots are needed
    if coeffs is None:
        needed = [(i, j) for i in range(3) for j in range(3)]
    else:
        needed = []
        for i in range(3):
            for j in range(3):
                key = f'c{i+1}{j+1}'
                if key in coeffs:
                    v = coeffs[key]
                    # Build unless it's a scalar 0; callables and non-zero scalars need it.
                    if callable(v) or v != 0.0:
                        needed.append((i, j))

    Dij = [[None, None, None], [None, None, None], [None, None, None]]
    for (i, j) in needed:
        Dij[i][j] = D[i] @ D[j]

    # Preallocated scratch buffers for local_S_fast:
    #   L_buf: per-face L matrix (zeroed at start of each local_S_fast call)
    #   tmp_buf: per-term `c[:, None] * D_term` accumulator
    N = njx * njy * njz
    L_buf = np.empty((N, N))
    tmp_buf = np.empty((N, N))

    return Dx_3d, Dy_3d, Dz_3d, Dij, L_buf, tmp_buf


def precompute_local_S_ops(dir, geom, px, py, pz, scl_x, scl_y, scl_z):
    """
    Build the universal lifting matrix C_dir (Legendre face -> Cheb merged-box
    boundary) and the centerline Cheb -> Legendre projection P_out for this
    direction. Also returns the surrounding-face Legendre coord array XYu.
    """
    (XY, Ii_dir, Ib_dir, Ibox, Ibox_inv,
     Ic_dir, Ic_edge,
     cl_xy_idx, cl_source_flag, cl_source_pos,
     Il_dir, If_dir, Id_dir, Iu_dir, Ibk_dir, Ir_dir,
     Ijl, Ijf, Ijd, Iju, Ijb, Ijr,
     pj_x, pj_y, pj_z) = geom

    ops = interp_ops(px, py, pz, scl_x, scl_y, scl_z)

    face_names = ['L', 'F', 'D', 'U', 'B', 'R']
    face_idxs = [Ijl, Ijf, Ijd, Iju, Ijb, Ijr]
    face_dir_idxs = [Il_dir, If_dir, Id_dir, Iu_dir, Ibk_dir, Ir_dir]

    XYu, Iul, Iuf, Iud, Iuu, Iub, Iur = XYU(dir, px, py, pz, scl_x, scl_y, scl_z)
    col_idxs = [Iul, Iuf, Iud, Iuu, Iub, Iur]

    n_box = len(Ibox)
    n_leg_total = XYu.shape[0]
    C_dir = np.zeros((n_box, n_leg_total))

    for fname, Ij, Iface_dir, Icol in zip(face_names, face_idxs, face_dir_idxs, col_idxs):
        lift_full = _face_lift(dir, fname, ops, px, py, pz, pj_x, pj_y, pj_z)
        rows_full = _face_full_row_order(dir, fname, XY, pj_x, pj_y, pj_z, px, py, pz)
        XY_idx_of_face = Ib_dir[Iface_dir]
        rows_full_to_pos = {int(rf): i for i, rf in enumerate(rows_full)}
        lift_row_pick = np.array([rows_full_to_pos[int(g)] for g in XY_idx_of_face])
        C_dir[np.ix_(Ij, Icol)] = lift_full[lift_row_pick, :]

    C2Lx, C2Ly, C2Lz = ops['C2L_x'], ops['C2L_y'], ops['C2L_z']
    if dir == 0:
        P_out = np.kron(C2Ly, C2Lz)
    elif dir == 1:
        P_out = np.kron(C2Lx, C2Lz)
    else:
        P_out = np.kron(C2Lx, C2Ly)

    return C_dir, P_out, XYu


# ---------------------------------------------------------------------------
# Fast local_S: fused L assembly + cached geometry/ops
# ---------------------------------------------------------------------------

# Coefficient key -> (kind, i, j). kind 0 = zero-order, 1 = first-order, 2 = second-order
_COEFF_DECODE = {
    'c': (0, None, None),
    'c1': (1, 0, None), 'c2': (1, 1, None), 'c3': (1, 2, None),
    'c11': (2, 0, 0), 'c12': (2, 0, 1), 'c13': (2, 0, 2),
    'c21': (2, 1, 0), 'c22': (2, 1, 1), 'c23': (2, 1, 2),
    'c31': (2, 2, 0), 'c32': (2, 2, 1), 'c33': (2, 2, 2),
}


def local_S_fast(dir, geom, diffops, ops_pkg, coeffs, origin, forcing):
    """
    Per-face local S and forcing trace b_dir, using precomputed per-direction
    geometry and operators.

    Parameters
    ----------
    dir : int
    geom : tuple from precompute_geometry(dir, ...)
    diffops : tuple (Dx_3d, Dy_3d, Dz_3d, Dij) from precompute_diffops(dir, ...)
    ops_pkg : tuple (C_dir, P_out, XYu) from precompute_local_S_ops(dir, ...)
    coeffs : dict, scalar or callable per entry
    origin : (x0, y0, z0)
    forcing : callable / scalar / None
    """
    (XY, Ii_dir, Ib_dir, Ibox, Ibox_inv,
     Ic_dir, Ic_edge,
     cl_xy_idx, cl_source_flag, cl_source_pos,
     *_) = geom
    Dx_3d, Dy_3d, Dz_3d, Dij, L_buf, tmp_buf = diffops
    C_dir, P_out, XYu = ops_pkg

    N = XY.shape[0]
    x0, y0, z0 = origin

    # Only allocate shifted-coordinate arrays if any user coefficient (or forcing)
    # actually needs them (i.e., is a callable). For all-scalar constant-coefficient
    # cases these are unused.
    need_coords = any(callable(v) for v in coeffs.values()) or callable(forcing)
    if need_coords:
        xg = XY[:, 0] + x0
        yg = XY[:, 1] + y0
        zg = XY[:, 2] + z0
    else:
        xg = yg = zg = None  # unused

    # Fused L assembly: one pass over user-supplied coeffs only.
    # Reuse the preallocated L_buf (zero-fill in place).
    L = L_buf
    L.fill(0.0)
    D1 = [Dx_3d, Dy_3d, Dz_3d]
    diag_idx = np.arange(N)
    for key, v in coeffs.items():
        if key not in _COEFF_DECODE:
            raise KeyError(f"Unknown coefficient key {key!r}")
        # Sample coefficient on the merged box.
        if callable(v):
            c = np.asarray(v(xg, yg, zg), dtype=float)
        elif v == 0.0:
            continue
        else:
            c = v   # scalar

        kind, i, j = _COEFF_DECODE[key]
        if kind == 0:
            L[diag_idx, diag_idx] += c
        elif kind == 1:
            D_term = D1[i]
            if np.isscalar(c):
                # L += c * D_term (in place, using tmp_buf to avoid a temporary)
                np.multiply(D_term, c, out=tmp_buf)
                L += tmp_buf
            else:
                np.multiply(c[:, None], D_term, out=tmp_buf)
                L += tmp_buf
        else:
            D_term = Dij[i][j]
            if np.isscalar(c):
                np.multiply(D_term, c, out=tmp_buf)
                L += tmp_buf
            else:
                np.multiply(c[:, None], D_term, out=tmp_buf)
                L += tmp_buf

    # Static-condensation pipeline using cached ops
    Lii = L[np.ix_(Ii_dir, Ii_dir)]
    Lii_lu = sla.lu_factor(Lii)
    Lib_box = L[np.ix_(Ii_dir, Ib_dir)][:, Ibox]

    u_int = -sla.lu_solve(Lii_lu, Lib_box @ C_dir)
    centerline_int = u_int[Ic_dir, :]
    centerline_edge = C_dir[Ibox_inv[Ic_edge], :]

    n_center = len(cl_xy_idx)
    centerline_trace = np.empty((n_center, C_dir.shape[1]))
    int_mask = (cl_source_flag == 0)
    centerline_trace[int_mask, :] = centerline_int[cl_source_pos[int_mask], :]
    centerline_trace[~int_mask, :] = centerline_edge[cl_source_pos[~int_mask], :]

    S_dir = P_out @ centerline_trace

    # Forcing
    if forcing is None:
        b_dir = np.zeros(S_dir.shape[0])
    else:
        if callable(forcing):
            # need_coords above already ensured xg/yg/zg exist in this branch
            fi = np.asarray(forcing(xg[Ii_dir], yg[Ii_dir], zg[Ii_dir]), dtype=float)
            if fi.ndim == 0:
                fi = np.full(len(Ii_dir), float(fi))
        else:
            if forcing == 0.0:
                b_dir = np.zeros(S_dir.shape[0])
                return S_dir, b_dir, Lii_lu
            fi = np.full(len(Ii_dir), float(forcing))
        if np.all(fi == 0.0):
            b_dir = np.zeros(S_dir.shape[0])
        else:
            u_part = sla.lu_solve(Lii_lu, fi)
            cl_vec = np.zeros(n_center)
            cl_vec[int_mask] = u_part[Ic_dir[cl_source_pos[int_mask]]]
            b_dir = P_out @ cl_vec

    return S_dir, b_dir, Lii_lu


def compute_forcing_trace(dir, geom, ops_pkg, Lii_lu, origin, forcing):
    """
    Compute the per-face forcing trace b_dir for a constant-coefficient problem,
    reusing the cached LU factorisation of Lii (translation-invariant).

    Skips L assembly and the homogeneous solve entirely.
    """
    (XY, Ii_dir, Ib_dir, Ibox, Ibox_inv,
     Ic_dir, Ic_edge,
     cl_xy_idx, cl_source_flag, cl_source_pos,
     *_) = geom
    _, P_out, _ = ops_pkg

    x0, y0, z0 = origin
    n_out = P_out.shape[0]
    n_center = len(cl_xy_idx)
    int_mask = (cl_source_flag == 0)

    if callable(forcing):
        xg = XY[Ii_dir, 0] + x0
        yg = XY[Ii_dir, 1] + y0
        zg = XY[Ii_dir, 2] + z0
        fi = np.asarray(forcing(xg, yg, zg), dtype=float)
        if fi.ndim == 0:
            fi = np.full(len(Ii_dir), float(fi))
    else:
        if forcing == 0.0:
            return np.zeros(n_out)
        fi = np.full(len(Ii_dir), float(forcing))

    if np.all(fi == 0.0):
        return np.zeros(n_out)

    u_part = sla.lu_solve(Lii_lu, fi)
    cl_vec = np.zeros(n_center)
    cl_vec[int_mask] = u_part[Ic_dir[cl_source_pos[int_mask]]]
    return P_out @ cl_vec



def XYU(dir, px, py, pz, scl_x, scl_y, scl_z):
    """
    Legendre DOF coordinates on the 6 surrounding faces of the merged box-pair.

    Each face has its own p x p (or 2p x p) Legendre tensor grid. The row
    ordering for doubled faces is (sub-box-outer, in-face-a, in-face-b-inner)
    to match the contiguous-by-sub-box memory layout that construct_SOMS_sparse
    expects.

    Face block order: [Left, Front, Down, Up, Back, Right].
    """
    xleg = _legendre_1d(px, scl_x)[0]
    yleg = _legendre_1d(py, scl_y)[0]
    zleg = _legendre_1d(pz, scl_z)[0]

    ones_x = np.ones(px); ones_y = np.ones(py); ones_z = np.ones(pz)

    def _face(a_pts, b_pts, fixed_axis, fixed_val, a_axis, b_axis,
              double_axis_flag, scl_d, n_sb):
        """
        Build face DOFs with (a outer, b inner) layout per sub-box, and
        sub-box as the outermost index when n_sb > 1.
        double_axis_flag = 'a' or 'b' if doubled in that axis (n_sb=2).
        """
        pa, pb = len(a_pts), len(b_pts)
        rows = pa * pb
        out = np.zeros((n_sb * rows, 3))
        for sb in range(n_sb):
            sl = slice(sb * rows, (sb + 1) * rows)
            a_disp = (sb * scl_d) if double_axis_flag == 'a' else 0.0
            b_disp = (sb * scl_d) if double_axis_flag == 'b' else 0.0
            out[sl, a_axis] = np.kron(a_pts + a_disp, np.ones(pb))
            out[sl, b_axis] = np.kron(np.ones(pa), b_pts + b_disp)
            out[sl, fixed_axis] = fixed_val
        return out

    if dir == 0:
        # Joined in x. End faces L/R (yz, no double); side faces F/B (xz, double in x),
        # D/U (xy, double in x). For F/B and D/U, x is the a-axis (outer in template).
        XYul = _face(yleg, zleg, 0, 0.0, 1, 2, None, 0.0, 1)
        XYuf = _face(xleg, zleg, 1, 0.0, 0, 2, 'a', scl_x, 2)
        XYud = _face(xleg, yleg, 2, 0.0, 0, 1, 'a', scl_x, 2)
        XYur = XYul.copy(); XYur[:, 0] = 2 * scl_x
        XYub = XYuf.copy(); XYub[:, 1] = scl_y
        XYuu = XYud.copy(); XYuu[:, 2] = scl_z
    elif dir == 1:
        # Joined in y. End faces F/B (xz, no double); side faces L/R (yz, double in y),
        # D/U (xy, double in y). For L/R, y is a-axis. For D/U, y is b-axis (inner).
        XYul = _face(yleg, zleg, 0, 0.0, 1, 2, 'a', scl_y, 2)
        XYuf = _face(xleg, zleg, 1, 0.0, 0, 2, None, 0.0, 1)
        XYud = _face(xleg, yleg, 2, 0.0, 0, 1, 'b', scl_y, 2)
        XYur = XYul.copy(); XYur[:, 0] = scl_x
        XYub = XYuf.copy(); XYub[:, 1] = 2 * scl_y
        XYuu = XYud.copy(); XYuu[:, 2] = scl_z
    else:  # dir == 2
        # Joined in z. End faces D/U (xy, no double); side faces L/R (yz, double in z),
        # F/B (xz, double in z). For L/R: z is b-axis. For F/B: z is b-axis.
        XYul = _face(yleg, zleg, 0, 0.0, 1, 2, 'b', scl_z, 2)
        XYuf = _face(xleg, zleg, 1, 0.0, 0, 2, 'b', scl_z, 2)
        XYud = _face(xleg, yleg, 2, 0.0, 0, 1, None, 0.0, 1)
        XYur = XYul.copy(); XYur[:, 0] = scl_x
        XYub = XYuf.copy(); XYub[:, 1] = scl_y
        XYuu = XYud.copy(); XYuu[:, 2] = 2 * scl_z

    XYu = np.concatenate([XYul, XYuf, XYud, XYuu, XYub, XYur], axis=0)
    face_sizes = [XYul.shape[0], XYuf.shape[0], XYud.shape[0],
                  XYuu.shape[0], XYub.shape[0], XYur.shape[0]]
    cumlen = np.concatenate([[0], np.cumsum(face_sizes)])
    Iul, Iuf, Iud, Iuu, Iub, Iur = [np.arange(cumlen[i], cumlen[i + 1]) for i in range(6)]

    return XYu, Iul, Iuf, Iud, Iuu, Iub, Iur


# ---------------------------------------------------------------------------
# Helper: build the lift operator for a single face
# ---------------------------------------------------------------------------

def _face_lift(dir, face_name, ops, px, py, pz, pj_x, pj_y, pj_z):
    """
    Build the 2D interpolation matrix mapping Legendre face DOFs to
    Cheb-with-endpoints values on the FULL face of the merged box (including
    edges and corners).

    Returns a matrix of shape (n_cheb_face_full, n_leg_face).

    The Cheb face has size n_cheb_face_full = pa_cheb * pb_cheb, where
    pa_cheb / pb_cheb are the full Cheb counts on each axis of that face
    (which is pj for the joined axis of the merged box, p for the others).

    Row ordering of the Cheb face matches the row ordering in Ib_dir for that
    face (which is governed by how the merged-box XY was built: x outer, y mid,
    z inner — when restricted to a face plane this becomes the two non-fixed
    axes in their natural x->y->z order).
    """
    L2Cx, L2Cy, L2Cz = ops['L2C_x'], ops['L2C_y'], ops['L2C_z']
    LS2CJx, LS2CJy, LS2CJz = ops['LS2CJ_x'], ops['LS2CJ_y'], ops['LS2CJ_z']

    # Identify which axis is joined and what op to use along each direction.
    # For a face perpendicular to axis `f_axis`, the in-face axes are the
    # remaining two, in the order (smaller_axis_index, larger_axis_index).
    # We need: along each in-face axis, the appropriate 1D Leg->Cheb op:
    #   - If that axis is the joined axis of the merged box: LS2CJ (input is 2p stacked)
    #   - Otherwise: L2C (input is p)
    # The 2D operator is a kron, with row ordering matching the XY natural order
    # (x outer, then y, then z restricted to the face).
    f_axis = {'L': 0, 'R': 0, 'F': 1, 'B': 1, 'D': 2, 'U': 2}[face_name]
    in_face_axes = [a for a in (0, 1, 2) if a != f_axis]   # already in increasing order
    a1, a2 = in_face_axes                                   # a1 < a2

    def axis_op(a):
        is_joined = (a == dir)
        if a == 0: return LS2CJx if is_joined else L2Cx
        if a == 1: return LS2CJy if is_joined else L2Cy
        if a == 2: return LS2CJz if is_joined else L2Cz

    op_a1 = axis_op(a1)
    op_a2 = axis_op(a2)

    # Column ordering: the Legendre face row ordering produced by XYU has
    # axis-a outer, axis-b inner where (a, b) = (a1, a2) in our face template,
    # BUT for doubled faces the sub-box outer convention applies. We need the
    # lift operator's columns to match XYU's row order on the Leg side.
    #
    # If neither axis is joined: cols are (a1_leg, a2_leg) outer/inner.
    #   Op = kron(op_a1, op_a2).
    # If a1 is joined (so op_a1 is LS2CJ with cols (sb_outer, leg_inner)):
    #   XYU produces (sb_outer, a1_leg, a2_leg). Op cols are (sb, a1_leg, a2_leg).
    #   kron(LS2CJ, L2C) has cols (LS2CJ_col, L2C_col) = (sb, a1_leg, a2_leg).  ✓
    # If a2 is joined (op_a2 is LS2CJ):
    #   XYU produces (sb_outer, a1_leg, a2_leg) [because doubled axis is inner of template].
    #   kron(L2C, LS2CJ) cols are (a1_leg, sb, a2_leg)  -- sb middle, WRONG.
    #   Instead use the split-concat trick:
    #     concat([kron(L2C, LS2CJ[:, :p]), kron(L2C, LS2CJ[:, p:])], axis=1)
    #     -> cols (sb_outer, a1_leg, a2_leg)  ✓
    if a1 == dir:
        # a1 is joined
        return np.kron(op_a1, op_a2)
    elif a2 == dir:
        # a2 is joined: split LS2CJ by sub-box
        p_joined_leg = op_a2.shape[1] // 2
        return np.concatenate([
            np.kron(op_a1, op_a2[:, :p_joined_leg]),
            np.kron(op_a1, op_a2[:, p_joined_leg:]),
        ], axis=1)
    else:
        # Neither axis is joined (end face)
        return np.kron(op_a1, op_a2)


# ---------------------------------------------------------------------------
# Local S matrix
# ---------------------------------------------------------------------------

def _resolve_coeffs_on_merged_box(coeffs, dir, px, py, pz,
                                  scl_x, scl_y, scl_z, origin):
    """
    Given user-supplied coefficient dict (scalars or vectorized callables),
    sample any callable on the merged-box collocation grid (translated by
    origin), and return a dict where every entry is either a scalar (passed
    through) or a 1D numpy array of length N = njx*njy*njz.

    The merged-box grid used here MUST match the XY ordering inside L_op.
    """
    # Reproduce L_op's joined-direction logic to get the local grid in
    # local coords [0, xlim] x [0, ylim] x [0, zlim].
    _, xpts = _cheb_1d(px, scl_x)
    _, ypts = _cheb_1d(py, scl_y)
    _, zpts = _cheb_1d(pz, scl_z)
    _, xpts2, _ = _joined_cheb_1d(px, scl_x)
    _, ypts2, _ = _joined_cheb_1d(py, scl_y)
    _, zpts2, _ = _joined_cheb_1d(pz, scl_z)

    if dir == 0:
        xj, yj, zj = xpts2, ypts, zpts
    elif dir == 1:
        xj, yj, zj = xpts, ypts2, zpts
    else:
        xj, yj, zj = xpts, ypts, zpts2

    njx, njy, njz = len(xj), len(yj), len(zj)
    ones_x, ones_y, ones_z = np.ones(njx), np.ones(njy), np.ones(njz)
    x_local = np.kron(np.kron(xj, ones_y), ones_z)
    y_local = np.kron(np.kron(ones_x, yj), ones_z)
    z_local = np.kron(np.kron(ones_x, ones_y), zj)

    # Shift into global coordinates if an origin is given. For coefficient
    # evaluation we need the global (x, y, z) physical positions.
    x0, y0, z0 = origin if origin is not None else (0.0, 0.0, 0.0)
    xg = x_local + x0
    yg = y_local + y0
    zg = z_local + z0

    keys = ['c11', 'c12', 'c13', 'c21', 'c22', 'c23', 'c31', 'c32', 'c33',
            'c1', 'c2', 'c3', 'c']
    resolved = {}
    for k in keys:
        if k not in coeffs:
            continue
        v = coeffs[k]
        if callable(v):
            arr = np.asarray(v(xg, yg, zg), dtype=float)
            if arr.ndim == 0:
                arr = np.full(len(xg), float(arr))
            resolved[k] = arr
        else:
            resolved[k] = v   # scalar passes through
    return resolved


def local_S(dir, px, py, pz, scl_x, scl_y, scl_z, coeffs, origin=None,
            forcing=None, check_ordering=False):
    """
    Build S_dir (and optionally the forcing trace b_dir) for one merged
    box-pair.

    S_dir maps Legendre data on the 6 surrounding faces of the merged
    box-pair to the Legendre trace on the centerline interface.

    With forcing, the centerline-trace identity for this merged pair is

        u_centerline + S_dir @ u_surrounding = b_dir

    where b_dir is the projection of L_ii^{-1} f_i onto the centerline
    face Legendre nodes. b_dir is zero when forcing is None or identically 0.

    Parameters
    ----------
    coeffs : dict
        Subset of keys 'c11'...'c33','c1','c2','c3','c'. Scalars or
        vectorized callables f(x, y, z).
    origin : tuple or None
        (x0, y0, z0) global position of the merged-box-pair's lower corner.
    forcing : callable or scalar or None
        f(x, y, z) right-hand side of L u = f. If None or 0, b_dir is zero.

    Returns
    -------
    S_dir : (n_leg_centerline, n_leg_surrounding) ndarray
    b_dir : (n_leg_centerline,) ndarray
    """
    resolved = _resolve_coeffs_on_merged_box(
        coeffs, dir, px, py, pz, scl_x, scl_y, scl_z, origin
    )
    info = L_op(dir, px, py, pz, scl_x, scl_y, scl_z, resolved)
    L = info['L']
    XY = info['XY']
    Ii_dir = info['Ii_dir']; Ib_dir = info['Ib_dir']
    Ibox = info['Ibox']
    Ic_dir = info['Ic_dir']; Ic_edge = info['Ic_edge']
    Ijl, Ijf, Ijd, Iju, Ijb, Ijr = info['Ijl'], info['Ijf'], info['Ijd'], info['Iju'], info['Ijb'], info['Ijr']
    pj_x, pj_y, pj_z = info['pj_x'], info['pj_y'], info['pj_z']

    Lii = L[np.ix_(Ii_dir, Ii_dir)]
    Lib = L[np.ix_(Ii_dir, Ib_dir)]
    Lib_box = Lib[:, Ibox]

    ops = interp_ops(px, py, pz, scl_x, scl_y, scl_z)

    face_names = ['L', 'F', 'D', 'U', 'B', 'R']
    face_idxs  = [Ijl, Ijf, Ijd, Iju, Ijb, Ijr]
    face_dir_idxs = [info['Il_dir'], info['If_dir'], info['Id_dir'],
                     info['Iu_dir'], info['Ibk_dir'], info['Ir_dir']]

    XYu, Iul, Iuf, Iud, Iuu, Iub, Iur = XYU(dir, px, py, pz, scl_x, scl_y, scl_z)
    col_idxs = [Iul, Iuf, Iud, Iuu, Iub, Iur]

    n_box = len(Ibox)
    n_leg_total = XYu.shape[0]
    C_dir = np.zeros((n_box, n_leg_total))

    for fname, Ij, Iface_dir, Icol in zip(face_names, face_idxs, face_dir_idxs, col_idxs):
        lift_full = _face_lift(dir, fname, ops, px, py, pz, pj_x, pj_y, pj_z)
        rows_full = _face_full_row_order(dir, fname, XY, pj_x, pj_y, pj_z, px, py, pz)
        XY_idx_of_face = Ib_dir[Iface_dir]
        rows_full_to_pos = {int(rf): i for i, rf in enumerate(rows_full)}
        lift_row_pick = np.array([rows_full_to_pos[int(g)] for g in XY_idx_of_face])
        C_dir[np.ix_(Ij, Icol)] = lift_full[lift_row_pick, :]

    # Homogeneous part: -L_ii^{-1} L_ib u_b, restricted to centerline interior
    u_int_full = -np.linalg.solve(Lii, Lib_box @ C_dir)
    centerline_int = u_int_full[Ic_dir, :]

    Ibox_inv = np.empty(len(Ibox), dtype=np.int64)
    Ibox_inv[Ibox] = np.arange(len(Ibox))
    centerline_edge = C_dir[Ibox_inv[Ic_edge], :]

    cl_xy_idx, cl_source_flag, cl_source_pos = _build_centerline_map(
        dir, XY, info, Ic_dir, Ic_edge
    )
    n_center = len(cl_xy_idx)

    centerline_trace = np.zeros((n_center, n_leg_total))
    for i in range(n_center):
        if cl_source_flag[i] == 0:
            centerline_trace[i, :] = centerline_int[cl_source_pos[i], :]
        else:
            centerline_trace[i, :] = centerline_edge[cl_source_pos[i], :]

    C2Lx, C2Ly, C2Lz = ops['C2L_x'], ops['C2L_y'], ops['C2L_z']
    if dir == 0:
        P_out = np.kron(C2Ly, C2Lz)
    elif dir == 1:
        P_out = np.kron(C2Lx, C2Lz)
    else:
        P_out = np.kron(C2Lx, C2Ly)

    S_dir = P_out @ centerline_trace

    # ----- Forcing trace b_dir -----
    # Particular solution: u_i_part = L_ii^{-1} f_i, evaluated on the merged-box
    # interior nodes Ii_dir. Centerline trace contribution = P_out @ (vector with
    # u_i_part at Ic_dir positions, 0 at Ic_edge positions).
    if forcing is None:
        b_dir = np.zeros(S_dir.shape[0])
    else:
        # Sample forcing on the merged-box interior nodes (in global coords).
        x0, y0, z0 = origin if origin is not None else (0.0, 0.0, 0.0)
        xi = XY[Ii_dir, 0] + x0
        yi = XY[Ii_dir, 1] + y0
        zi = XY[Ii_dir, 2] + z0
        if callable(forcing):
            fi = np.asarray(forcing(xi, yi, zi), dtype=float)
            if fi.ndim == 0:
                fi = np.full(len(xi), float(fi))
        else:
            fi = np.full(len(xi), float(forcing))
        if np.all(fi == 0.0):
            b_dir = np.zeros(S_dir.shape[0])
        else:
            u_part_int = np.linalg.solve(Lii, fi)
            # Centerline interior values come from u_part_int[Ic_dir];
            # centerline edge values are zero (forcing acts on interior only).
            centerline_vec = np.zeros(n_center)
            for i in range(n_center):
                if cl_source_flag[i] == 0:
                    centerline_vec[i] = u_part_int[Ic_dir[cl_source_pos[i]]]
                # else: edge node, contributes 0
            b_dir = P_out @ centerline_vec

    if check_ordering:
        _check_local_ordering(dir, px, py, pz, scl_x, scl_y, scl_z,
                              S_dir, XY, cl_xy_idx, XYu)

    return S_dir, b_dir


def _face_full_row_order(dir, face_name, XY, pj_x, pj_y, pj_z, px, py, pz):
    """
    Return the array of global XY indices for the full Cheb face of given name,
    in the row order produced by `_face_lift` (which is kron(a_op, b_op) along
    the two in-face axes a < b).

    The full-face row ordering of `_face_lift` is:
      - axis a1 (smaller index) outer
      - axis a2 (larger index) inner
      - if doubled in joined direction with sb-outer convention: sub-box outer
    """
    # Determine the fixed-axis coord value
    f_axis = {'L': 0, 'R': 0, 'F': 1, 'B': 1, 'D': 2, 'U': 2}[face_name]
    # Per face, the fixed value is 0 or the corresponding limit
    # We can look it up from XY by picking any node on the face plane.
    # Easier: identify by the partition convention.
    # The "L" face is x=0; "R" is x=xlim; etc.
    if face_name == 'L':   fixed_val = 0.0
    elif face_name == 'R': fixed_val = (2 * XY[:, 0].max() / 2) if dir != 0 else XY[:, 0].max()
    elif face_name == 'F': fixed_val = 0.0
    elif face_name == 'B': fixed_val = XY[:, 1].max()
    elif face_name == 'D': fixed_val = 0.0
    elif face_name == 'U': fixed_val = XY[:, 2].max()

    # Actually simpler: use the max along that axis or 0
    if face_name in ('L', 'F', 'D'):
        fixed_val = 0.0
    else:
        fixed_val = XY[:, f_axis].max()

    on_face = np.abs(XY[:, f_axis] - fixed_val) < 1e-10
    face_idx = np.where(on_face)[0]
    # XY natural order is x outer, y middle, z inner. Restricting to face_idx
    # preserves this order. So if f_axis == 0 (L or R), remaining order is
    # (y outer, z inner) — already what _face_lift produces (since a1=1, a2=2,
    # and kron(op_y, op_z) has y outer, z inner).
    # If f_axis == 1 (F or B), remaining order in face_idx is (x outer, z inner)
    # — matches kron(op_x, op_z).
    # If f_axis == 2 (D or U), remaining order is (x outer, y inner) — matches
    # kron(op_x, op_y).
    #
    # However, for doubled faces (when the joined direction is one of the in-face
    # axes), _face_lift uses split-concat which puts sb-outer when the joined
    # axis is a2. For sb-outer, the row order is (sb_outer, a1, a2_within_sb).
    # The natural XY order has joined-axis values interleaved (e.g. for dir=2
    # and face_name='L', joined axis is z=a2, full-face is (y outer, z inner)
    # with z covering both sub-boxes interleaved by Cheb-joined-full ordering.
    # Actually for joined-Cheb-full ordering on [0, 2*scl], the nodes are NOT
    # sb-outer — they're a single Cheb grid covering both sub-boxes. So
    # `_face_lift`'s output rows (which produce p_joined values along the
    # joined axis) are NOT sub-box-stacked but rather a single Cheb-joined
    # ordering. So actually there's NO sb-outer convention on the OUTPUT of
    # the lift — only on the INPUT side (which doesn't affect row ordering).
    # So the row ordering is straightforwardly natural XY order on the face.
    return face_idx


def _build_centerline_map(dir, XY, info, Ic_dir, Ic_edge):
    """
    Build a unified ordering of centerline-face nodes (p x p Cheb-full tensor
    on the two in-face axes of the centerline face). For each centerline node,
    record whether it's an interior (Ii_dir) or edge (Ib_dir) node and its
    index within that set.
    """
    xc = info['xc']
    Ii_dir = info['Ii_dir']
    Ib_dir = info['Ib_dir']

    on_center = np.abs(XY[:, dir] - xc) < 1e-10
    cl_xy_idx = np.where(on_center)[0]   # global indices in XY natural order
    # XY natural order: x outer, y mid, z inner. Restricted to a plane, this
    # gives natural order of the two in-face axes (smaller-axis outer).

    # For each centerline node, find whether it's in Ii_dir or Ib_dir, and its position.
    Ii_set = {int(v): i for i, v in enumerate(Ii_dir)}
    Ib_set = {int(v): i for i, v in enumerate(Ib_dir)}

    cl_source_flag = np.zeros(len(cl_xy_idx), dtype=np.int64)  # 0 = int, 1 = edge
    cl_source_pos = np.zeros(len(cl_xy_idx), dtype=np.int64)
    for i, g in enumerate(cl_xy_idx):
        if int(g) in Ii_set:
            cl_source_flag[i] = 0
            ii_pos = Ii_set[int(g)]
            ic_pos = np.where(Ic_dir == ii_pos)[0]
            assert len(ic_pos) == 1, \
                f"centerline node {g} (Ii_dir pos {ii_pos}) not found in Ic_dir"
            cl_source_pos[i] = ic_pos[0]
        else:
            cl_source_flag[i] = 1
            ib_pos = Ib_set[int(g)]
            ic_pos = np.where(Ic_edge == ib_pos)[0]
            assert len(ic_pos) == 1, \
                f"centerline node {g} (Ib_dir pos {ib_pos}) not found in Ic_edge"
            cl_source_pos[i] = ic_pos[0]

    return cl_xy_idx, cl_source_flag, cl_source_pos


def _check_local_ordering(dir, px, py, pz, scl_x, scl_y, scl_z,
                          S_dir, XY, cl_xy_idx, XYu):
    """Sanity checks on shapes and centerline ordering."""
    # Centerline face has p_a * p_b Legendre nodes
    if dir == 0:
        expected_rows = py * pz
    elif dir == 1:
        expected_rows = px * pz
    else:
        expected_rows = px * py
    assert S_dir.shape[0] == expected_rows, \
        f"local_S({dir}): row count {S_dir.shape[0]} != {expected_rows}"

    assert S_dir.shape[1] == XYu.shape[0], \
        f"local_S({dir}): col count {S_dir.shape[1]} != {XYu.shape[0]}"

    # Centerline Cheb-full count
    expected_cl = px * py if dir == 2 else (py * pz if dir == 0 else px * pz)
    assert len(cl_xy_idx) == expected_cl, \
        f"local_S({dir}): centerline full count {len(cl_xy_idx)} != {expected_cl}"


# ---------------------------------------------------------------------------
# Global DOFs (Legendre on every face)
# ---------------------------------------------------------------------------

def global_dofs(tiling, px, py, pz, Lx, Ly, Lz):
    Lx0, Ly0, Lz0 = tiling
    scl_x, scl_y, scl_z = Lx / Lx0, Ly / Ly0, Lz / Lz0


    xleg = _legendre_1d(px, scl_x)[0]
    yleg = _legendre_1d(py, scl_y)[0]
    zleg = _legendre_1d(pz, scl_z)[0]
    ones_x = np.ones(px); ones_y = np.ones(py); ones_z = np.ones(pz)

    xy = np.zeros((px * py, 3))
    xy[:, 0] = np.kron(xleg, ones_y); xy[:, 1] = np.kron(ones_x, yleg)

    yz = np.zeros((py * pz, 3))
    yz[:, 1] = np.kron(yleg, ones_z); yz[:, 2] = np.kron(ones_y, zleg)

    xz = np.zeros((px * pz, 3))
    xz[:, 0] = np.kron(xleg, ones_z); xz[:, 2] = np.kron(ones_x, zleg)

    nxy, nyz, nxz = xy.shape[0], yz.shape[0], xz.shape[0]

    # Closed-form counts
    n_yz_faces = (tiling[0] + 1) * tiling[1] * tiling[2]
    n_xz_faces = tiling[0] * (tiling[1] + 1) * tiling[2]
    n_xy_faces = tiling[0] * tiling[1] * (tiling[2] + 1)
    n_faces = n_yz_faces + n_xz_faces + n_xy_faces
    n_dofs = n_yz_faces * nyz + n_xz_faces * nxz + n_xy_faces * nxy

    # Preallocate
    XYtot = np.empty((n_dofs, 3))
    md_vec = np.empty(n_faces, dtype=np.int8)
    b_vec = np.empty(n_faces, dtype=bool)
    indx_vec = np.empty(n_faces, dtype=np.int64)
    indy_vec = np.empty(n_faces, dtype=np.int64)
    indz_vec = np.empty(n_faces, dtype=np.int64)

    dof_off = 0   # running DOF offset
    face_off = 0  # running face index

    for indx in range(tiling[0] + 1):
        x_off = indx * scl_x
        for indy in range(tiling[1]):
            y_off = indy * scl_y
            for indz in range(tiling[2]):
                z_off = indz * scl_z
                XYtot[dof_off:dof_off + nyz, 0] = yz[:, 0] + x_off
                XYtot[dof_off:dof_off + nyz, 1] = yz[:, 1] + y_off
                XYtot[dof_off:dof_off + nyz, 2] = yz[:, 2] + z_off
                dof_off += nyz
                md_vec[face_off] = 0
                b_vec[face_off] = (indx == 0 or indx == tiling[0])
                indx_vec[face_off] = indx
                indy_vec[face_off] = indy
                indz_vec[face_off] = indz
                face_off += 1

        if indx < tiling[0]:
            for indy in range(tiling[1] + 1):
                y_off = indy * scl_y
                for indz in range(tiling[2]):
                    z_off = indz * scl_z
                    XYtot[dof_off:dof_off + nxz, 0] = xz[:, 0] + x_off
                    XYtot[dof_off:dof_off + nxz, 1] = xz[:, 1] + y_off
                    XYtot[dof_off:dof_off + nxz, 2] = xz[:, 2] + z_off
                    dof_off += nxz
                    md_vec[face_off] = 1
                    b_vec[face_off] = (indy == 0 or indy == tiling[1])
                    indx_vec[face_off] = indx
                    indy_vec[face_off] = indy
                    indz_vec[face_off] = indz
                    face_off += 1

                if indy < tiling[1]:
                    for indz in range(tiling[2] + 1):
                        z_off = indz * scl_z
                        XYtot[dof_off:dof_off + nxy, 0] = xy[:, 0] + x_off
                        XYtot[dof_off:dof_off + nxy, 1] = xy[:, 1] + y_off
                        XYtot[dof_off:dof_off + nxy, 2] = xy[:, 2] + z_off
                        dof_off += nxy
                        md_vec[face_off] = 2
                        b_vec[face_off] = (indz == 0 or indz == tiling[2])
                        indx_vec[face_off] = indx
                        indy_vec[face_off] = indy
                        indz_vec[face_off] = indz
                        face_off += 1

    assert dof_off == n_dofs, f"DOF count mismatch: {dof_off} != {n_dofs}"
    assert face_off == n_faces, f"face count mismatch: {face_off} != {n_faces}"

    return XYtot, md_vec, b_vec, nxy, nyz, nxz, indx_vec, indy_vec, indz_vec


# ---------------------------------------------------------------------------
# Sparse assembly
# ---------------------------------------------------------------------------

def construct_SOMS_sparse(nxy, nyz, nxz, md_vec, b_vec, n_dofs, tiling,
                          indx_vec, indy_vec, indz_vec,
                          Sx, Sy, Sz,
                          ct_pde=True,
                          coeffs=None,
                          forcing=None,
                          bx=None, by=None, bz=None,
                          px=None, py=None, pz=None,
                          scl_x=None, scl_y=None, scl_z=None,
                          geom=None, diffops=None, ops_pkg=None,
                          Ii=None, Ib=None,
                          Lii_lu_x=None, Lii_lu_y=None, Lii_lu_z=None):
    """
    Build Sii and Sib directly in CSR format, and the global forcing vector
    ftild.  Avoids forming the full n_dofs x n_dofs Stot matrix.

    ``Ii`` and ``Ib`` are the interior / boundary DOF index arrays produced by
    ``global_dofs`` + the boundary-coordinate test in ``SOMS_solver_sparse``.
    They must be provided; ``SOMS_solver_sparse`` always supplies them.

    With forcing (any non-None argument), the system to solve is

        Sii @ u_i  =  -Sib @ u_b  +  ftild[Ii]

    If ct_pde is True and forcing is None, Sx/Sy/Sz are precomputed S-matrices
    used at every interior centerline face.  Otherwise local_S_fast is called
    per face using the precomputed per-direction contexts geom[dir], diffops[dir],
    ops_pkg[dir].
    """
    # ------------------------------------------------------------------
    # Partition information
    # ------------------------------------------------------------------
    # global_to_ii[g]  = local row in Sii  (or -1 if g is in Ib)
    # global_to_ib[g]  = local col in Sib  (or -1 if g is in Ii)
    # Every global DOF belongs to exactly one of Ii, Ib.
    n_ii = len(Ii)
    n_ib = len(Ib)

    global_to_ii = np.full(n_dofs, -1, dtype=np.int32)
    global_to_ib = np.full(n_dofs, -1, dtype=np.int32)
    global_to_ii[Ii] = np.arange(n_ii, dtype=np.int32)
    global_to_ib[Ib] = np.arange(n_ib, dtype=np.int32)

    # ------------------------------------------------------------------
    # Exact NNZ pre-pass for Sii and Sib
    # ------------------------------------------------------------------
    # Every source block is exactly one face's DOF range; b_vec[that face]
    # tells us whether all its DOFs are Ib (True) or Ii (False).
    # We run the same loop as assembly but only do b_vec lookups, giving
    # exact nnz counts before any allocation.
    nFYZ = tiling[1] * tiling[2] * nyz
    nFXZ = tiling[2] * nxz
    nFXY = (tiling[2] + 1) * nxy

    # Map DOF-offset -> face index (only face-boundary offsets are keys).
    _face_start = np.empty(len(md_vec), dtype=np.int64)
    _off = 0
    for _i in range(len(md_vec)):
        _face_start[_i] = _off
        _sz = nyz if md_vec[_i] == 0 else (nxz if md_vec[_i] == 1 else nxy)
        _off += _sz
    _start_to_face = {int(s): i for i, s in enumerate(_face_start)}

    sii_nnz_max = 0
    sib_nnz_max = 0
    _ctr = 0
    for _f in range(len(md_vec)):
        _md = md_vec[_f]
        _ix = indx_vec[_f]; _iy = indy_vec[_f]; _iz = indz_vec[_f]
        if _md == 2:
            _step_bk    = (tiling[2] + 1 - _iz) * nxy + (_iz - 1) * nxz
            _step_front = _iz * nxy + (tiling[2] + 1 - _iz) * nxz
            _step_right = ((tiling[1] - _iy) * tiling[2] * nxz
                           + (tiling[2] + 1) * (tiling[1] - _iy - 1) * nxy
                           + nxy * (tiling[2] + 1 - _iz)
                           + (_iz - 1) * nyz + _iy * tiling[2] * nyz)
            _start_left = (_ctr + _step_right
                           - tiling[2] * tiling[1] * nyz
                           - (tiling[2] + 1) * tiling[1] * nxy
                           - (tiling[1] + 1) * tiling[2] * nxz)
            _src_starts = [_start_left, _start_left + nyz,
                           _ctr - _step_front, _ctr - _step_front + nxz,
                           _ctr - nxy, _ctr + nxy,
                           _ctr + _step_bk, _ctr + _step_bk + nxz,
                           _ctr + _step_right, _ctr + _step_right + nyz]
            _src_sizes  = [nyz, nyz, nxz, nxz, nxy, nxy, nxz, nxz, nyz, nyz]
            _nT = nxy; _ctr += nxy
        elif _md == 1:
            _step_front  = nxz * tiling[2] + nxy * (tiling[2] + 1)
            _block_stride = nFYZ + (tiling[1] + 1) * nFXZ + tiling[1] * nFXY
            _sl1 = _ix * _block_stride + tiling[2] * nyz * (_iy - 1) + nyz * _iz
            _sl2 = _ix * _block_stride + tiling[2] * nyz * _iy       + nyz * _iz
            _sr1 = _sl1 + _block_stride; _sr2 = _sl2 + _block_stride
            _sd1 = _ix * _block_stride + nFYZ + _iy * nFXZ + (_iy - 1) * nFXY + _iz * nxy
            _sd2 = _sd1 + nFXY + nFXZ; _su1 = _sd1 + nxy; _su2 = _su1 + nFXY + nFXZ
            _src_starts = [_sl1, _sl2, _ctr - _step_front,
                           _sd1, _sd2, _su1, _su2,
                           _ctr + _step_front, _sr1, _sr2]
            _src_sizes  = [nyz, nyz, nxz, nxy, nxy, nxy, nxy, nxz, nyz, nyz]
            _nT = nxz; _ctr += nxz
        else:
            _step_right  = (nyz * tiling[2] * tiling[1]
                            + nxy * (tiling[2] + 1) * tiling[1]
                            + nxz * (tiling[1] + 1) * tiling[2])
            _block_stride = nFYZ + (tiling[1] + 1) * nFXZ + tiling[1] * nFXY
            _prev         = (_ix - 1) * _block_stride
            _sf1 = _prev + nFYZ + _iy * nFXZ + _iy * nFXY + _iz * nxz
            _sf2 = _sf1 + nFYZ + tiling[1] * nFXY + (tiling[1] + 1) * nFXZ
            _sb1 = _sf1 + nFXY + nFXZ; _sb2 = _sf2 + nFXY + nFXZ
            _sd1 = _prev + nFYZ + (_iy + 1) * nFXZ + _iy * nFXY + _iz * nxy
            _sd2 = _sd1 + tiling[1] * nFXY + (tiling[1] + 1) * nFXZ + nFYZ
            _su1 = _sd1 + nxy; _su2 = _sd2 + nxy
            _src_starts = [_ctr - _step_right,
                           _sf1, _sf2, _sd1, _sd2, _su1, _su2,
                           _sb1, _sb2, _ctr + _step_right]
            _src_sizes  = [nyz, nxz, nxz, nxy, nxy, nxy, nxy, nxz, nxz, nyz]
            _nT = nyz; _ctr += nyz
        if not b_vec[_f]:
            _n_ib = sum(_src_sizes[_k] for _k, _s in enumerate(_src_starts)
                        if b_vec[_start_to_face[_s]])
            _n_ii = sum(_src_sizes) - _n_ib
            sii_nnz_max += _nT * (_n_ii + 1)   # +1 for diagonal
            sib_nnz_max += _nT * _n_ib

    sii_data    = np.empty(sii_nnz_max, dtype=np.float64)
    sii_indices = np.empty(sii_nnz_max, dtype=np.int32)
    sii_indptr  = np.zeros(n_ii + 1, dtype=np.int32)

    sib_data    = np.empty(sib_nnz_max, dtype=np.float64)
    sib_indices = np.empty(sib_nnz_max, dtype=np.int32)
    sib_indptr  = np.zeros(n_ii + 1, dtype=np.int32)

    sii_nnz = 0
    sib_nnz = 0

    ftild = np.zeros(n_dofs)
    ctr = 0

    def _write_interior_face(target, source, S_local):
        """
        For each row in `target` (all in Ii), scatter the off-diagonal entries
        in `source` into Sii (sources in Ii) or Sib (sources in Ib), and write
        the +1 diagonal into Sii.  All per-row column lists are kept sorted.
        """
        nonlocal sii_nnz, sib_nnz
        nT = len(target)
        nS = len(source)

        # Classify source columns once per face.
        src_ii_local = global_to_ii[source]   # -1 where source is in Ib
        src_ib_local = global_to_ib[source]   # -1 where source is in Ii
        in_ii = src_ii_local >= 0             # boolean mask, length nS
        in_ib = ~in_ii

        # Sorted local column indices for each sub-matrix (needed for CSR).
        src_ii_cols = src_ii_local[in_ii]     # already in global order -> sort
        src_ib_cols = src_ib_local[in_ib]

        # Argsort within each sub-group (global source order is not necessarily
        # sorted in local Ii / Ib index space).
        perm_ii = np.argsort(src_ii_cols, kind='stable')
        perm_ib = np.argsort(src_ib_cols, kind='stable')
        src_ii_cols_sorted = src_ii_cols[perm_ii]
        src_ib_cols_sorted = src_ib_cols[perm_ib]

        # S_local columns that map to Ii / Ib (indices into the nS source axis).
        orig_ii = np.where(in_ii)[0][perm_ii]   # column positions in S_local -> Sii
        orig_ib = np.where(in_ib)[0][perm_ib]   # column positions in S_local -> Sib

        for k in range(nT):
            r_global = int(target[k])
            r_ii     = int(global_to_ii[r_global])   # local Sii row

            # --- Sii row: off-diagonal (Ii sources) + diagonal ---
            diag_col = r_ii
            n_ii_src = len(src_ii_cols_sorted)
            ins = int(np.searchsorted(src_ii_cols_sorted, diag_col))

            # left of diag
            sii_indices[sii_nnz : sii_nnz + ins] = src_ii_cols_sorted[:ins]
            sii_data   [sii_nnz : sii_nnz + ins] = -S_local[k, orig_ii[:ins]]
            sii_nnz += ins
            # diagonal
            sii_indices[sii_nnz] = diag_col
            sii_data   [sii_nnz] = 1.0
            sii_nnz += 1
            # right of diag
            sii_indices[sii_nnz : sii_nnz + n_ii_src - ins] = src_ii_cols_sorted[ins:]
            sii_data   [sii_nnz : sii_nnz + n_ii_src - ins] = -S_local[k, orig_ii[ins:]]
            sii_nnz += n_ii_src - ins

            sii_indptr[r_ii + 1] = sii_nnz

            # --- Sib row: Ib sources only (already sorted) ---
            n_ib_src = len(src_ib_cols_sorted)
            sib_indices[sib_nnz : sib_nnz + n_ib_src] = src_ib_cols_sorted
            sib_data   [sib_nnz : sib_nnz + n_ib_src] = -S_local[k, orig_ib]
            sib_nnz += n_ib_src
            sib_indptr[r_ii + 1] = sib_nnz

    def _write_boundary_face(target):
        """Boundary DOFs: identity rows in Sii (they are in Ii? No — boundary
        faces have all their DOFs in Ib, so they produce NO rows in Sii/Sib).
        We simply skip them; Sii/Sib only have rows for Ii DOFs."""
        pass  # boundary face DOFs are in Ib; they have no rows in Sii or Sib

    for indxyz in range(len(md_vec)):
        match md_vec[indxyz]:
            case 2:
                target = np.arange(ctr, ctr + nxy)
                step_up = nxy
                step_down = nxy
                step_bk = (tiling[2] + 1 - indz_vec[indxyz]) * nxy + (indz_vec[indxyz] - 1) * nxz
                step_front = indz_vec[indxyz] * nxy + (tiling[2] + 1 - indz_vec[indxyz]) * nxz
                step_right = (
                    (tiling[1] - indy_vec[indxyz]) * tiling[2] * nxz
                    + (tiling[2] + 1) * (tiling[1] - indy_vec[indxyz] - 1) * nxy
                    + nxy * (tiling[2] + 1 - indz_vec[indxyz])
                    + (indz_vec[indxyz] - 1) * nyz
                    + indy_vec[indxyz] * tiling[2] * nyz
                )
                start_left = (ctr + step_right
                              - tiling[2] * tiling[1] * nyz
                              - (tiling[2] + 1) * tiling[1] * nxy
                              - (tiling[1] + 1) * tiling[2] * nxz)
                source = np.concatenate([
                    np.arange(start_left,       start_left + 2 * nyz),
                    np.arange(ctr - step_front, ctr - step_front + 2 * nxz),
                    np.arange(ctr - step_down,  ctr - step_down + nxy),
                    np.arange(ctr + step_up,    ctr + step_up + nxy),
                    np.arange(ctr + step_bk,    ctr + step_bk + 2 * nxz),
                    np.arange(ctr + step_right, ctr + step_right + 2 * nyz),
                ])
                ctr += nxy
                if not b_vec[indxyz]:
                    origin = (indx_vec[indxyz] * scl_x,
                              indy_vec[indxyz] * scl_y,
                              (indz_vec[indxyz] - 1) * scl_z)
                    if ct_pde:
                        S_local = Sz
                        if forcing is not None:
                            b_local = compute_forcing_trace(
                                2, geom[2], ops_pkg[2], Lii_lu_z, origin, forcing)
                        else:
                            b_local = None
                    else:
                        S_local, b_local, _ = local_S_fast(
                            2, geom[2], diffops[2], ops_pkg[2],
                            coeffs, origin, forcing,
                        )
                    _write_interior_face(target, source, S_local)
                    if b_local is not None:
                        ftild[target] += b_local
                else:
                    _write_boundary_face(target)
            case 1:
                target = np.arange(ctr, ctr + nxz)
                step_front = nxz * tiling[2] + nxy * (tiling[2] + 1)
                step_back = step_front
                block_stride = nFYZ + (tiling[1] + 1) * nFXZ + tiling[1] * nFXY
                start_left1 = indx_vec[indxyz] * block_stride + tiling[2] * nyz * (indy_vec[indxyz] - 1) + nyz * indz_vec[indxyz]
                start_left2 = indx_vec[indxyz] * block_stride + tiling[2] * nyz * indy_vec[indxyz]      + nyz * indz_vec[indxyz]
                start_right1 = start_left1 + block_stride
                start_right2 = start_left2 + block_stride
                start_down1 = indx_vec[indxyz] * block_stride + nFYZ + indy_vec[indxyz] * nFXZ + (indy_vec[indxyz] - 1) * nFXY + indz_vec[indxyz] * nxy
                start_down2 = start_down1 + nFXY + nFXZ
                start_up1 = start_down1 + nxy
                start_up2 = start_up1   + nFXY + nFXZ
                source = np.concatenate([
                    np.arange(start_left1,      start_left1  + nyz),
                    np.arange(start_left2,      start_left2  + nyz),
                    np.arange(ctr - step_back,  ctr - step_back  + nxz),
                    np.arange(start_down1,      start_down1  + nxy),
                    np.arange(start_down2,      start_down2  + nxy),
                    np.arange(start_up1,        start_up1    + nxy),
                    np.arange(start_up2,        start_up2    + nxy),
                    np.arange(ctr + step_front, ctr + step_front + nxz),
                    np.arange(start_right1,     start_right1 + nyz),
                    np.arange(start_right2,     start_right2 + nyz),
                ])
                ctr += nxz
                if not b_vec[indxyz]:
                    origin = (indx_vec[indxyz] * scl_x,
                              (indy_vec[indxyz] - 1) * scl_y,
                              indz_vec[indxyz] * scl_z)
                    if ct_pde:
                        S_local = Sy
                        if forcing is not None:
                            b_local = compute_forcing_trace(
                                1, geom[1], ops_pkg[1], Lii_lu_y, origin, forcing)
                        else:
                            b_local = None
                    else:
                        S_local, b_local, _ = local_S_fast(
                            1, geom[1], diffops[1], ops_pkg[1],
                            coeffs, origin, forcing,
                        )
                    _write_interior_face(target, source, S_local)
                    if b_local is not None:
                        ftild[target] += b_local
                else:
                    _write_boundary_face(target)
            case 0:
                target = np.arange(ctr, ctr + nyz)
                step_right = (
                    nyz * tiling[2] * tiling[1]
                    + nxy * (tiling[2] + 1) * tiling[1]
                    + nxz * (tiling[1] + 1) * tiling[2]
                )
                step_left = step_right
                block_stride = nFYZ + (tiling[1] + 1) * nFXZ + tiling[1] * nFXY
                prev_block = (indx_vec[indxyz] - 1) * block_stride
                start_front1 = prev_block + nFYZ + indy_vec[indxyz] * nFXZ + indy_vec[indxyz] * nFXY + indz_vec[indxyz] * nxz
                start_front2 = start_front1 + nFYZ + tiling[1] * nFXY + (tiling[1] + 1) * nFXZ
                start_back1 = start_front1 + nFXY + nFXZ
                start_back2 = start_front2 + nFXY + nFXZ
                start_down1 = prev_block + nFYZ + (indy_vec[indxyz] + 1) * nFXZ + indy_vec[indxyz] * nFXY + indz_vec[indxyz] * nxy
                start_down2 = start_down1 + tiling[1] * nFXY + (tiling[1] + 1) * nFXZ + nFYZ
                start_up1 = start_down1 + nxy
                start_up2 = start_down2 + nxy
                source = np.concatenate([
                    np.arange(ctr - step_left,  ctr - step_left  + nyz),
                    np.arange(start_front1,     start_front1 + nxz),
                    np.arange(start_front2,     start_front2 + nxz),
                    np.arange(start_down1,      start_down1  + nxy),
                    np.arange(start_down2,      start_down2  + nxy),
                    np.arange(start_up1,        start_up1    + nxy),
                    np.arange(start_up2,        start_up2    + nxy),
                    np.arange(start_back1,      start_back1  + nxz),
                    np.arange(start_back2,      start_back2  + nxz),
                    np.arange(ctr + step_right, ctr + step_right + nyz),
                ])
                ctr += nyz
                if not b_vec[indxyz]:
                    origin = ((indx_vec[indxyz] - 1) * scl_x,
                              indy_vec[indxyz] * scl_y,
                              indz_vec[indxyz] * scl_z)
                    if ct_pde:
                        S_local = Sx
                        if forcing is not None:
                            b_local = compute_forcing_trace(
                                0, geom[0], ops_pkg[0], Lii_lu_x, origin, forcing)
                        else:
                            b_local = None
                    else:
                        S_local, b_local, _ = local_S_fast(
                            0, geom[0], diffops[0], ops_pkg[0],
                            coeffs, origin, forcing,
                        )
                    _write_interior_face(target, source, S_local)
                    if b_local is not None:
                        ftild[target] += b_local
                else:
                    _write_boundary_face(target)

    Sii = sp.csr_matrix(
        (sii_data, sii_indices, sii_indptr),
        shape=(n_ii, n_ii),
    )
    Sib = sp.csr_matrix(
        (sib_data, sib_indices, sib_indptr),
        shape=(n_ii, n_ib),
    )
    return Sii, Sib, ftild


def SOMS_solver_sparse(px, py, pz, nbx, nby, nbz, Lx=1., Ly=1., Lz=1.,
                       coeffs=None, ct_pde=True, forcing=None,
                       dbg=0, check_ordering=False):
    """
    Build the SOMS3D system Sii, Sib (and forcing trace ftild) for a general
    2nd-order elliptic PDE on the cuboid [0, Lx] x [0, Ly] x [0, Lz] tiled
    into nbx x nby x nbz blocks, with p x p Legendre face DOFs per face.

    The PDE is

        L u = sum_{ij} c_{ij}(x) d_i d_j u + sum_i c_i(x) d_i u + c(x) u = f(x)

    in non-divergence form, with possible right-hand side f.

    The solve is

        Sii @ u_i = -Sib @ u_b + ftild[Ii]

    where ftild[Ii] is the projection of the local particular solutions to
    each interior centerline face's Legendre nodes. Stot is unchanged by
    the presence of forcing.

    Parameters
    ----------
    coeffs : dict
        Subset of keys 'c11','c12','c13','c21','c22','c23','c31','c32','c33',
        'c1','c2','c3','c'. Each value is a Python scalar or a vectorized
        callable f(x, y, z).
    ct_pde : bool
        If True, coefficients assumed constant; Sx, Sy, Sz computed once
        for matrix assembly. If False, local S is computed per face.
    forcing : callable, scalar, or None
        Right-hand side f(x, y, z) of L u = f. None means f == 0 (the
        homogeneous case), in which case ftild is zero. If forcing is
        non-None, local_S is called per face even when ct_pde=True (since
        the forcing trace depends on the local origin).

    Returns
    -------
    Sii, Sib : sparse CSR matrices
    ftild : ndarray of length n_dofs
        Global forcing trace. Use ftild[Ii] in the RHS of the interior solve.
    XYtot : (n_dofs, 3) array of global DOF coordinates
    Ii, Ib : index arrays for interior and boundary DOFs
    """
    if coeffs is None:
        raise ValueError("coeffs must be provided. For (Delta + k^2) u = f "
                         "use {'c11': 1, 'c22': 1, 'c33': 1, 'c': k**2}.")

    tiling = [nbx, nby, nbz]
    scl_x, scl_y, scl_z = Lx / nbx, Ly / nby, Lz / nbz

    if dbg > 0: print("Precomputing per-direction geometry / diffops / lift-project ops ...")
    geom = [precompute_geometry(d, px, py, pz, scl_x, scl_y, scl_z) for d in (0, 1, 2)]
    diffops = [precompute_diffops(d, px, py, pz, scl_x, scl_y, scl_z, coeffs=coeffs)
               for d in (0, 1, 2)]
    ops_pkg = [precompute_local_S_ops(d, geom[d], px, py, pz, scl_x, scl_y, scl_z)
               for d in (0, 1, 2)]

    if ct_pde:
        if dbg > 0: print("Computing local S matrices (constant-coefficient mode) ...")
        Sx, _, Lii_lu_x = local_S_fast(0, geom[0], diffops[0], ops_pkg[0],
                                        coeffs, (0., 0., 0.), None)
        Sy, _, Lii_lu_y = local_S_fast(1, geom[1], diffops[1], ops_pkg[1],
                                        coeffs, (0., 0., 0.), None)
        Sz, _, Lii_lu_z = local_S_fast(2, geom[2], diffops[2], ops_pkg[2],
                                        coeffs, (0., 0., 0.), None)
        if dbg > 0: print(f"  Sx: {Sx.shape}")
        # diffops is no longer needed: S_local = Sx/Sy/Sz for all faces,
        # and b_local uses the cached Lii_lu factors.
        diffops = None
    else:
        if dbg > 0: print("Variable-coefficient mode: local S computed per face.")
        Sx = Sy = Sz = None
        Lii_lu_x = Lii_lu_y = Lii_lu_z = None

    XYtot, md_vec, b_vec, nxy, nyz, nxz, indx_vec, indy_vec, indz_vec = \
        global_dofs(tiling, px, py, pz, Lx, Ly, Lz)
    n_dofs = XYtot.shape[0]

    if check_ordering:
        unique = np.unique(np.round(XYtot, 12), axis=0)
        assert unique.shape[0] == XYtot.shape[0], "DOF uniqueness violated"

    # Compute Ii / Ib before assembly so construct_SOMS_sparse can build
    # Sii and Sib directly without forming the full n_dofs x n_dofs Stot.
    x, y, z = XYtot[:, 0], XYtot[:, 1], XYtot[:, 2]
    Ib = np.where(
        (np.abs(x) < 1e-10) | (np.abs(x - Lx) < 1e-10) |
        (np.abs(y) < 1e-10) | (np.abs(y - Ly) < 1e-10) |
        (np.abs(z) < 1e-10) | (np.abs(z - Lz) < 1e-10)
    )[0]
    Ii = np.setdiff1d(np.arange(n_dofs), Ib)

    if dbg > 0: print(f"  n_dofs = {n_dofs}, n_ii = {len(Ii)}, n_ib = {len(Ib)}")
    Sii, Sib, ftild = construct_SOMS_sparse(
        nxy, nyz, nxz, md_vec, b_vec, n_dofs,
        tiling, indx_vec, indy_vec, indz_vec, Sx, Sy, Sz,
        ct_pde=ct_pde, coeffs=coeffs, forcing=forcing,
        px=px, py=py, pz=pz,
        scl_x=scl_x, scl_y=scl_y, scl_z=scl_z,
        geom=geom, diffops=diffops, ops_pkg=ops_pkg,
        Ii=Ii, Ib=Ib,
        Lii_lu_x=Lii_lu_x, Lii_lu_y=Lii_lu_y, Lii_lu_z=Lii_lu_z,
    )
    return Sii, Sib, ftild, XYtot, Ii, Ib