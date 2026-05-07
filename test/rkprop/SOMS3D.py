import numpy as np
import solver.spectral.spectralSolver as spectral
import numpy.polynomial.chebyshev as chebpoly
import matplotlib.pyplot as plt
import scipy.linalg as sclinalg
import time


def _cheb_1d(p, scl):
    """
    Return the scaled Chebyshev differentiation matrix and interior-mapped points
    for a single direction with polynomial order p and length scale scl.

    Points are mapped from [-1,1] -> [0, scl].
    """
    D, pts = spectral.cheb(p)
    D = -2 * D / scl
    pts = ((pts[::-1] + 1) / 2) * scl
    return D, pts


def _joined_cheb_1d(p, scl):
    """
    Return the scaled Chebyshev differentiation matrix and interior-mapped points
    for the *joined* (double-wide) direction.

    Polynomial order is rounded up to the nearest even number of (3*p)//2.
    Points are mapped from [-1,1] -> [0, 2*scl].
    """
    p_joined = (3 * p) // 2
    p_joined -= p_joined % 2          # round down to even
    D, pts = spectral.cheb(p_joined)
    pts = (1 + pts[::-1]) * scl       # maps to [0, 2*scl]
    D = -D / scl
    return D, pts, p_joined


def _svd_interp(E_src, E_tgt):
    """
    Build the interpolation matrix that maps from the column-space of E_src to E_tgt
    via a truncated SVD pseudo-inverse:  cc = E_tgt @ pinv(E_src).
    """
    U, s, V = np.linalg.svd(E_src, full_matrices=False)
    k = np.sum(s > 1e-15 * s[0])
    Uk, Vk, sk = U[:, :k], V[:k, :].T, s[:k]
    return (E_tgt @ Vk) @ np.diag(sk ** -1) @ Uk.T


def L_op(dir, px, py, pz, scl_x, scl_y, scl_z, kh):
    """
    Construct differential operator for 2x1 joined block set-up for the Helmholtz equation at kappa = kh

    ----------------------------------------------------------------------------------------
    | 'joined' refers to (attributes of) block with the dimensions of two 1x1 blocks glued |
    | together BUT                                                                         |
    | with the DOFS on the faces along the glued dimension NOT two tiles of 2-d cheb pts,  |
    | but instead one 2-d cheb grid                                                        |
    ----------------------------------------------------------------------------------------

    INPUT:  - dir (in [0,1,2])      :    whether joined block is in x-dir (0), y-dir (1), z-dir (2)
            - (px,py,pz)            :    polynomial orders for each 1x1 block in the 2x1 set-up
            - (scl_x,scl_y,scl_z)   :    length scales of the 1x1 blocks
            - kh                    :    wave number

    OUTPUT: - L_joined_dir                  : diff op for the joined block
            - Ijl,Ijf,Ijd,Iju,Ijb,Ijr       : joined DOFS corresponding to left, front, down, up, back and right face (as subsets of boundary dofs)
            - Ic_dir                        : joined interior DOFs corresponding to the interior central interface
            - Ibox_joined_dir               : union of Ijl,Ijf,Ijd,Iju,Ijb,Ijr, in that order
            - Ii_dir,Ib_dir                 : interior and boundary dofs for the joined block
            - XY_joined_dir                 : spatial locations of the total DOFs
    """
    Dx, xpts = _cheb_1d(px, scl_x)
    Dy, ypts = _cheb_1d(py, scl_y)
    Dz, zpts = _cheb_1d(pz, scl_z)

    Dx2, xpts2, _ = _joined_cheb_1d(px, scl_x)
    Dy2, ypts2, _ = _joined_cheb_1d(py, scl_y)
    Dz2, zpts2, _ = _joined_cheb_1d(pz, scl_z)

    # Replace differentiation matrix and point set in the joined direction
    if dir == 0:
        Dx = Dx2
        xjoined, yjoined, zjoined = xpts2, ypts, zpts
        xlim, ylim, zlim, xc = 2 * scl_x, scl_y, scl_z, scl_x
    elif dir == 1:
        Dy = Dy2
        xjoined, yjoined, zjoined = xpts, ypts2, zpts
        xlim, ylim, zlim, xc = scl_x, 2 * scl_y, scl_z, scl_y
    else:  # dir == 2
        Dz = Dz2
        xjoined, yjoined, zjoined = xpts, ypts, zpts2
        xlim, ylim, zlim, xc = scl_x, scl_y, 2 * scl_z, scl_z

    njx, njy, njz = len(xjoined), len(yjoined), len(zjoined)

    # Build spatial coordinate array via Kronecker products
    ones_x = np.ones(njx)
    ones_y = np.ones(njy)
    ones_z = np.ones(njz)
    XY_joined_dir = np.column_stack([
        np.kron(np.kron(xjoined, ones_y), ones_z),
        np.kron(np.kron(ones_x, yjoined), ones_z),
        np.kron(np.kron(ones_x, ones_y), zjoined),
    ])

    x, y, z = XY_joined_dir[:, 0], XY_joined_dir[:, 1], XY_joined_dir[:, 2]

    # Interior and boundary index sets
    Ii_dir = np.where((x > 0) & (x < xlim) & (y > 0) & (y < ylim) & (z > 0) & (z < zlim))[0]
    Ib_dir = np.where(
        (np.abs(x) < 1e-10) | (np.abs(x - xlim) < 1e-10) |
        (np.abs(y) < 1e-10) | (np.abs(y - ylim) < 1e-10) |
        (np.abs(z) < 1e-10) | (np.abs(z - zlim) < 1e-10)
    )[0]

    xb, yb, zb = XY_joined_dir[Ib_dir, 0], XY_joined_dir[Ib_dir, 1], XY_joined_dir[Ib_dir, 2]

    # Face index sets within boundary DOFs
    Il_dir  = np.where((np.abs(xb) < 1e-10)        & (yb > 0) & (yb < ylim) & (zb > 0) & (zb < zlim))[0]
    Ir_dir  = np.where((np.abs(xb - xlim) < 1e-10) & (yb > 0) & (yb < ylim) & (zb > 0) & (zb < zlim))[0]
    If_dir  = np.where((np.abs(yb) < 1e-10)        & (xb > 0) & (xb < xlim) & (zb > 0) & (zb < zlim))[0]
    Ibk_dir = np.where((np.abs(yb - ylim) < 1e-10) & (xb > 0) & (xb < xlim) & (zb > 0) & (zb < zlim))[0]
    Id_dir  = np.where((np.abs(zb) < 1e-10)        & (xb > 0) & (xb < xlim) & (yb > 0) & (yb < ylim))[0]
    Iu_dir  = np.where((np.abs(zb - zlim) < 1e-10) & (xb > 0) & (xb < xlim) & (yb > 0) & (yb < ylim))[0]

    Ic_dir = np.where(np.abs(XY_joined_dir[Ii_dir, dir] - xc) < 1e-10)[0]

    Ibox_joined_dir = np.concatenate([Il_dir, If_dir, Id_dir, Iu_dir, Ibk_dir, Ir_dir])

    # Build contiguous index ranges for each face within Ibox
    face_lengths = [len(Il_dir), len(If_dir), len(Id_dir), len(Iu_dir), len(Ibk_dir), len(Ir_dir)]
    cumlen = np.concatenate([[0], np.cumsum(face_lengths)])
    Ijl, Ijf, Ijd, Iju, Ijb, Ijr = [np.arange(cumlen[i], cumlen[i + 1]) for i in range(6)]

    # Helmholtz operator: -(Dxx ⊗ I ⊗ I + I ⊗ Dyy ⊗ I + I ⊗ I ⊗ Dzz) - kh^2 * I
    Ix, Iy, Iz = np.eye(njx), np.eye(njy), np.eye(njz)
    Dxx, Dyy, Dzz = Dx @ Dx, Dy @ Dy, Dz @ Dz
    L_joined_dir = (
        -np.kron(np.kron(Dxx, Iy), Iz)
        - np.kron(np.kron(Ix, Dyy), Iz)
        - np.kron(np.kron(Ix, Iy), Dzz)
        - kh * kh * np.kron(np.kron(Ix, Iy), Iz)
    )

    return L_joined_dir, Ijl, Ijf, Ijd, Iju, Ijb, Ijr, Ic_dir, Ibox_joined_dir, Ii_dir, Ib_dir, XY_joined_dir


def interp_ops(px, py, pz, scl_x, scl_y, scl_z):
    """
    Construct interpolation operators from joined dofs to unjoined dofs

    ----------------------------------------------------------------------------------------
    | 'joined' refers to (attributes of) block with the dimensions of two 1x1 blocks glued |
    | together BUT                                                                         |
    | with the DOFS on the faces along the glued dimension NOT two tiles of 2-d cheb pts,  |
    | but instead one 2-d cheb grid                                                        |
    ----------------------------------------------------------------------------------------

    INPUT:  - (px,py,pz)            :    polynomial orders for each 1x1 block in the 2x1 set-up
            - (scl_x,scl_y,scl_z)   :    length scales of the 1x1 blocks

    OUTPUT: - Interp_x,Interp_y,Interp_z    : Interpolation operators associated to (px,scl_x) etc


    DESCRIPTION:

                                          Interp
    |* *  *   *   *  * *|* *  *   *   *  * *| <-------------------------------- |* *  *   *    *    *    *    *   *  * *|
            (scl,p)             (scl,p)                                                         (2*scl,3*p/2)


    Interp is the map that takes ~3*p/2 cheb grid sample on the interval [0,2*scl]
    and interpolates underlying poly to the two pieces of p cheb pts, on [0,scl] and [scl,2*scl] respectively
    """
    # Build single-block interior points (stripped of boundary)
    _, xpts = _cheb_1d(px, scl_x)
    _, ypts = _cheb_1d(py, scl_y)
    _, zpts = _cheb_1d(pz, scl_z)

    nx, ny, nz = len(xpts), len(ypts), len(zpts)

    xpts_int = xpts[1:nx - 1]
    ypts_int = ypts[1:ny - 1]
    zpts_int = zpts[1:nz - 1]

    # Two-tile interior points (concatenate both sub-blocks)
    x2 = np.append(xpts_int, scl_x + xpts_int)
    y2 = np.append(ypts_int, scl_y + ypts_int)
    z2 = np.append(zpts_int, scl_z + zpts_int)

    # Build joined interior points (strip boundary from double-wide grid)
    _, xpts2, _ = _joined_cheb_1d(px, scl_x)
    _, ypts2, _ = _joined_cheb_1d(py, scl_y)
    _, zpts2, _ = _joined_cheb_1d(pz, scl_z)

    xpts2 = xpts2[1:-1]
    ypts2 = ypts2[1:-1]
    zpts2 = zpts2[1:-1]

    # Build Chebyshev Vandermonde matrices and compute interpolation via SVD pseudo-inverse
    # All points are normalised to [-1,1] around the centre of the double-wide interval.
    nxpts2, nypts2, nzpts2 = len(xpts2), len(ypts2), len(zpts2)

    ccx = _svd_interp(
        chebpoly.chebvander((x2    - scl_x) / scl_x, nxpts2 - 1),
        chebpoly.chebvander((xpts2 - scl_x) / scl_x, nxpts2 - 1),
    )
    ccy = _svd_interp(
        chebpoly.chebvander((y2    - scl_y) / scl_y, nypts2 - 1),
        chebpoly.chebvander((ypts2 - scl_y) / scl_y, nypts2 - 1),
    )
    ccz = _svd_interp(
        chebpoly.chebvander((z2    - scl_z) / scl_z, nzpts2 - 1),
        chebpoly.chebvander((zpts2 - scl_z) / scl_z, nzpts2 - 1),
    )

    return ccx, ccy, ccz


def XYU(dir, px, py, pz, scl_x, scl_y, scl_z):
    """
    Construct unjoined DOFs

    ----------------------------------------------------------------------------------------
    | 'unjoined' refers to (attributes of) two 1x1 blocks stuck together along a direction |
    | with their touching face DOFs identified                                             |
    ----------------------------------------------------------------------------------------

    INPUT:  - dir (in [0,1,2])      :    whether two blocks are stacked together in x-dir (0), y-dir (1), z-dir (2)
            - (px,py,pz)            :    polynomial orders for each 1x1 block in the 2x1 set-up
            - (scl_x,scl_y,scl_z)   :    length scales of the 1x1 blocks

    OUTPUT: - XYu                       :   spatial coordinates of the unjoined DOFs
            - Iul,Iuf,Iud,Iuu,Iub,Iur   :   faces (left, front, down, up, back, right) as subsets of the boundary

    NOTE: edges of the faces are removed
    """
    _, xpts = _cheb_1d(px, scl_x)
    _, ypts = _cheb_1d(py, scl_y)
    _, zpts = _cheb_1d(pz, scl_z)

    nx, ny, nz = len(xpts), len(ypts), len(zpts)

    # Strip boundary nodes
    xpts = xpts[1:nx - 1]
    ypts = ypts[1:ny - 1]
    zpts = zpts[1:nz - 1]

    # Extend points in the joined direction to cover both sub-blocks
    if dir == 0:
        xpts = np.append(xpts, scl_x + xpts)
    elif dir == 1:
        ypts = np.append(ypts, scl_y + ypts)
    else:  # dir == 2
        zpts = np.append(zpts, scl_z + zpts)

    nx, ny, nz = len(xpts), len(ypts), len(zpts)
    ones_x, ones_y, ones_z = np.ones(nx), np.ones(ny), np.ones(nz)

    # Left face  (x=0): varies in y,z
    XYul = np.zeros((ny * nz, 3))
    XYul[:, 1] = np.kron(ypts, ones_z)
    XYul[:, 2] = np.kron(ones_y, zpts)

    # Front face (y=0): varies in x,z
    XYuf = np.zeros((nx * nz, 3))
    XYuf[:, 0] = np.kron(xpts, ones_z)
    XYuf[:, 2] = np.kron(ones_x, zpts)

    # Down face  (z=0): varies in x,y
    XYud = np.zeros((nx * ny, 3))
    XYud[:, 0] = np.kron(xpts, ones_y)
    XYud[:, 1] = np.kron(ones_x, ypts)

    # Up, back, right faces are offsets of their counterparts
    up_offset   = scl_z * np.array([0, 0, 1]) * (2 if dir == 2 else 1)
    back_offset = scl_y * np.array([0, 1, 0]) * (2 if dir == 1 else 1)
    right_offset = scl_x * np.array([1, 0, 0]) * (2 if dir == 0 else 1)

    XYuu = XYud + up_offset
    XYub = XYuf + back_offset
    XYur = XYul + right_offset

    XYu = np.concatenate([XYul, XYuf, XYud, XYuu, XYub, XYur], axis=0)

    # Build contiguous index ranges for each face
    face_sizes = [XYul.shape[0], XYuf.shape[0], XYud.shape[0],
                  XYuu.shape[0], XYub.shape[0], XYur.shape[0]]
    cumlen = np.concatenate([[0], np.cumsum(face_sizes)])
    Iul, Iuf, Iud, Iuu, Iub, Iur = [np.arange(cumlen[i], cumlen[i + 1]) for i in range(6)]

    return XYu, Iul, Iuf, Iud, Iuu, Iub, Iur


def global_dofs(tiling, px, py, pz, Lx, Ly, Lz):
    """
    Construct global reduced DOFs

    INPUT:  - tiling (1x3 int array)    :   the tiling defining the underlying non-overlapping domain decomp
            - (px,py,pz)                :    polynomial orders for each 1x1 block
            - (Lx,Ly,Lz)                :    length scales of the global domain

    OUTPUT: - XYtot                     :   spatial coordinates of the global DOFs
            - md_vec                    :   int vec with length the number of cuboid faces
                                            'missing direction vector', equals 2 for an xy face, 1 for an xz face, 0 for a yz face
            - b_vec                     :   bool vec with length the number of cuboid faces
                                            'boundary vec', 1 for a face on the global boundary, zero otherwise
            -nxy,nyz,nxz                :   number of points per face of each cuboid
                                            for example, the bottom and top face have nxy points etc.
            -indx_vec,indy_vec,indz_vec :   int vecs with length the number of cuboid faces
                                            correspond to the loop indexes at which face is added
    """
    Lx0, Ly0, Lz0 = tiling
    scl_x, scl_y, scl_z = Lx / Lx0, Ly / Ly0, Lz / Lz0

    _, xpts = _cheb_1d(px, scl_x)
    _, ypts = _cheb_1d(py, scl_y)
    _, zpts = _cheb_1d(pz, scl_z)

    nx, ny, nz = len(xpts), len(ypts), len(zpts)

    # Strip boundary nodes
    xpts = xpts[1:nx - 1]
    ypts = ypts[1:ny - 1]
    zpts = zpts[1:nz - 1]

    ones_x, ones_y, ones_z = np.ones_like(xpts), np.ones_like(ypts), np.ones_like(zpts)

    # Template face grids at the origin
    xy = np.zeros(((nx - 2) * (ny - 2), 3))
    xy[:, 0] = np.kron(xpts, ones_y)
    xy[:, 1] = np.kron(ones_x, ypts)

    yz = np.zeros(((ny - 2) * (nz - 2), 3))
    yz[:, 1] = np.kron(ypts, ones_z)
    yz[:, 2] = np.kron(ones_y, zpts)

    xz = np.zeros(((nx - 2) * (nz - 2), 3))
    xz[:, 0] = np.kron(xpts, ones_z)
    xz[:, 2] = np.kron(ones_x, zpts)

    nxy, nyz, nxz = xy.shape[0], yz.shape[0], xz.shape[0]

    # Pre-allocate lists; convert to arrays at the end (avoids repeated np.append)
    XYtot_list = []
    md_list, b_list, indx_list, indy_list, indz_list = [], [], [], [], []

    for indx in range(tiling[0] + 1):
        x_off = indx * scl_x
        for indy in range(tiling[1]):
            for indz in range(tiling[2]):
                XYtot_list.append(yz + np.array([x_off, indy * scl_y, indz * scl_z]))
                md_list.append(0)
                b_list.append(indx == 0 or indx == tiling[0])
                indx_list.append(indx)
                indy_list.append(indy)
                indz_list.append(indz)

        if indx < tiling[0]:
            for indy in range(tiling[1] + 1):
                for indz in range(tiling[2]):
                    XYtot_list.append(xz + np.array([x_off, indy * scl_y, indz * scl_z]))
                    md_list.append(1)
                    b_list.append(indy == 0 or indy == tiling[1])
                    indx_list.append(indx)
                    indy_list.append(indy)
                    indz_list.append(indz)

                if indy < tiling[1]:
                    for indz in range(tiling[2] + 1):
                        XYtot_list.append(xy + np.array([x_off, indy * scl_y, indz * scl_z]))
                        md_list.append(2)
                        b_list.append(indz == 0 or indz == tiling[2])
                        indx_list.append(indx)
                        indy_list.append(indy)
                        indz_list.append(indz)

    XYtot     = np.concatenate(XYtot_list, axis=0)
    md_vec    = np.array(md_list,   dtype=np.int8)
    b_vec     = np.array(b_list,    dtype=bool)
    indx_vec  = np.array(indx_list, dtype=np.int64)
    indy_vec  = np.array(indy_list, dtype=np.int64)
    indz_vec  = np.array(indz_list, dtype=np.int64)

    return XYtot, md_vec, b_vec, nxy, nyz, nxz, indx_vec, indy_vec, indz_vec


def construct_SOMS(nxy, nyz, nxz, md_vec, b_vec, XYtot, tiling, indx_vec, indy_vec, indz_vec, Sx, Sy, Sz):
    """
    Construct global SOMS system

    INPUT:  - nxy,nyz,nxz               :   number of points for the xy-faces (down and up) etc.
            - md_vec                    :   vec indication direction of each face (xy=2,xz=1,yz=0)
            - b_vec                     :   vec indicating if corresponding face is on global boundary
            - XYtot                     :   spatial locations of global DOFs
            - tiling                    :   underlying non-overlapping domain decomp tiling
            - indx_vec,indy_vec,indz_vec:   hard to explain, sorry
            - uXY                       :   solution on XYtot, for testing purposes
            - Sx,Sy,Sz                  :   Local solve-and-restrict maps, one for each direction

    OUTPUT: - Stot                      :   total S system
    """
    ctr = 0
    nFYZ = tiling[1] * tiling[2] * nyz
    nFXZ = tiling[2] * nxz
    nFXY = (tiling[2] + 1) * nxy

    Stot = np.identity(XYtot.shape[0])

    for indxyz in range(len(md_vec)):
        source = np.zeros(shape=(0,), dtype=np.int64)
        match md_vec[indxyz]:
            case 2:
                target = np.arange(ctr, ctr + nxy)

                step_up    = nxy
                step_down  = nxy
                step_bk    = (tiling[2] + 1 - indz_vec[indxyz]) * nxy + (indz_vec[indxyz] - 1) * nxz
                step_front = indz_vec[indxyz] * nxy + (tiling[2] + 1 - indz_vec[indxyz]) * nxz
                step_right = (
                    (tiling[1] - indy_vec[indxyz]) * tiling[2] * nxz
                    + (tiling[2] + 1) * (tiling[1] - indy_vec[indxyz] - 1) * nxy
                    + nxy * (tiling[2] + 1 - indz_vec[indxyz])
                    + (indz_vec[indxyz] - 1) * nyz
                    + indy_vec[indxyz] * tiling[2] * nyz
                )
                start_left = ctr + step_right - tiling[2] * tiling[1] * nyz - (tiling[2] + 1) * tiling[1] * nxy - (tiling[1] + 1) * tiling[2] * nxz

                source = np.concatenate([
                    np.arange(start_left,          start_left + 2 * nyz),
                    np.arange(ctr - step_front,    ctr - step_front + 2 * nxz),
                    np.arange(ctr - step_down,     ctr - step_down + nxy),
                    np.arange(ctr + step_up,       ctr + step_up + nxy),
                    np.arange(ctr + step_bk,       ctr + step_bk + 2 * nxz),
                    np.arange(ctr + step_right,    ctr + step_right + 2 * nyz),
                ])
                ctr += nxy

                if not b_vec[indxyz]:
                    Stot[np.ix_(target, source)] = -Sz

            case 1:
                target = np.arange(ctr, ctr + nxz)
                # woops, mixed up front and back
                step_front = nxz * tiling[2] + nxy * (tiling[2] + 1)
                step_back  = step_front

                block_stride = nFYZ + (tiling[1] + 1) * nFXZ + tiling[1] * nFXY
                start_left1 = indx_vec[indxyz] * block_stride + tiling[2] * nyz * (indy_vec[indxyz] - 1) + nyz * indz_vec[indxyz]
                start_left2 = indx_vec[indxyz] * block_stride + tiling[2] * nyz *  indy_vec[indxyz]      + nyz * indz_vec[indxyz]

                start_right1 = start_left1 + block_stride
                start_right2 = start_left2 + block_stride

                start_down1 = indx_vec[indxyz] * block_stride + nFYZ + indy_vec[indxyz] * nFXZ + (indy_vec[indxyz] - 1) * nFXY + indz_vec[indxyz] * nxy
                start_down2 = start_down1 + nFXY + nFXZ
                start_up1   = start_down1 + nxy
                start_up2   = start_up1   + nFXY + nFXZ

                source = np.concatenate([
                    np.arange(start_left1,      start_left1  + nyz),
                    np.arange(start_left2,      start_left2  + nyz),
                    np.arange(ctr - step_back,  ctr - step_back + nxz),
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
                    Stot[np.ix_(target, source)] = -Sy

            case 0:
                target = np.arange(ctr, ctr + nyz)
                step_right = (
                    nyz * tiling[2] * tiling[1]
                    + nxy * (tiling[2] + 1) * tiling[1]
                    + nxz * (tiling[1] + 1) * tiling[2]
                )
                step_left = step_right

                block_stride = nFYZ + (tiling[1] + 1) * nFXZ + tiling[1] * nFXY
                prev_block   = (indx_vec[indxyz] - 1) * block_stride

                start_front1 = prev_block + nFYZ + indy_vec[indxyz] * nFXZ + indy_vec[indxyz] * nFXY + indz_vec[indxyz] * nxz
                start_front2 = start_front1 + nFYZ + tiling[1] * nFXY + (tiling[1] + 1) * nFXZ
                start_back1  = start_front1 + nFXY + nFXZ
                start_back2  = start_front2 + nFXY + nFXZ

                start_down1  = prev_block + nFYZ + (indy_vec[indxyz] + 1) * nFXZ + indy_vec[indxyz] * nFXY + indz_vec[indxyz] * nxy
                start_down2  = start_down1 + tiling[1] * nFXY + (tiling[1] + 1) * nFXZ + nFYZ
                start_up1    = start_down1 + nxy
                start_up2    = start_down2 + nxy

                source = np.concatenate([
                    np.arange(ctr - step_left,  ctr - step_left + nyz),
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
                    Stot[np.ix_(target, source)] = -Sx

    return Stot


def local_S(dir, px, py, pz, scl_x, scl_y, scl_z, kh):
    """
    Construct local SOMS system

    INPUT:  - dir                       :   direction in {0,1,2} in which blocks are stacked/joined
            - (px,py,pz)                :   polynomial orders for each 1x1 block
            - (scl_x,scl_y,scl_z)       :   length scales of 1x1 blocks

    OUTPUT: - S_dir                     :   local S-block in direction dir
    """
    _, xpts = _cheb_1d(px, scl_x)
    _, ypts = _cheb_1d(py, scl_y)
    _, zpts = _cheb_1d(pz, scl_z)
    nx, ny, nz = len(xpts), len(ypts), len(zpts)

    L_joined_dir, Ijl, Ijf, Ijd, Iju, Ijb, Ijr, Ic_dir, Ibox_joined_dir, Ii_dir, Ib_dir, XY_joined_dir = \
        L_op(dir, px, py, pz, scl_x, scl_y, scl_z, kh)

    Lii = L_joined_dir[np.ix_(Ii_dir, Ii_dir)]
    Lib_box = L_joined_dir[np.ix_(Ii_dir, Ib_dir)][:, Ibox_joined_dir]

    # fiddle with interp since kronecker is not respected by global disc
    # inv_inter_xy is interpolation for the two xy-faces etc.
    Interp_x, Interp_y, Interp_z = interp_ops(px, py, pz, scl_x, scl_y, scl_z)

    Iy_int = np.eye(ny - 2)
    Ix_int = np.eye(nx - 2)
    Iz_int = np.eye(nz - 2)

    if dir == 0:
        inv_inter_yz = np.eye((ny - 2) * (nz - 2))
        inv_inter_xz = np.kron(Interp_x, Iz_int)
        inv_inter_xy = np.kron(Interp_x, Iy_int)
    elif dir == 1:
        inv_inter_xy = np.concatenate([
            np.kron(Ix_int, Interp_y[:, :(ny - 2)]),
            np.kron(Ix_int, Interp_y[:, (ny - 2):]),
        ], axis=1)
        inv_inter_yz = np.kron(Interp_y, Iz_int)
        inv_inter_xz = np.eye((nx - 2) * (nz - 2))
    else:  # dir == 2
        inv_inter_xy = np.eye((nx - 2) * (ny - 2))
        inv_inter_yz = np.concatenate([
            np.kron(Iy_int, Interp_z[:, :(nz - 2)]),
            np.kron(Iy_int, Interp_z[:, (nz - 2):]),
        ], axis=1)
        inv_inter_xz = np.concatenate([
            np.kron(Ix_int, Interp_z[:, :(nz - 2)]),
            np.kron(Ix_int, Interp_z[:, (nz - 2):]),
        ], axis=1)

    XYu, Iul, Iuf, Iud, Iuu, Iub, Iur = XYU(dir, px, py, pz, scl_x, scl_y, scl_z)

    # C_dir is total inverse interpolation matrix over the 6 faces
    C_dir = np.zeros((len(Ibox_joined_dir), XYu.shape[0]))
    C_dir[np.ix_(Ijl, Iul)] = inv_inter_yz
    C_dir[np.ix_(Ijf, Iuf)] = inv_inter_xz
    C_dir[np.ix_(Ijd, Iud)] = inv_inter_xy
    C_dir[np.ix_(Iju, Iuu)] = inv_inter_xy
    C_dir[np.ix_(Ijb, Iub)] = inv_inter_xz
    C_dir[np.ix_(Ijr, Iur)] = inv_inter_yz

    S_dir = -(np.linalg.solve(Lii, Lib_box @ C_dir))[Ic_dir, :]
    return S_dir


def SOMS_solver(px, py, pz, nbx, nby, nbz, Lx=1., Ly=1., Lz=1., kh=0, dbg=0):
    tiling = [nbx, nby, nbz]
    scl_x = Lx / nbx
    scl_y = Ly / nby
    scl_z = Lz / nbz

    Sx = local_S(0, px, py, pz, scl_x, scl_y, scl_z, kh)
    Sy = local_S(1, px, py, pz, scl_x, scl_y, scl_z, kh)
    Sz = local_S(2, px, py, pz, scl_x, scl_y, scl_z, kh)

    XYtot, md_vec, b_vec, nxy, nyz, nxz, indx_vec, indy_vec, indz_vec = \
        global_dofs(tiling, px, py, pz, Lx, Ly, Lz)
    Stot = construct_SOMS(nxy, nyz, nxz, md_vec, b_vec, XYtot, tiling, indx_vec, indy_vec, indz_vec, Sx, Sy, Sz)

    x, y, z = XYtot[:, 0], XYtot[:, 1], XYtot[:, 2]
    Ib = np.where(
        (np.abs(x) < 1e-10) | (np.abs(x - Lx) < 1e-10) |
        (np.abs(y) < 1e-10) | (np.abs(y - Ly) < 1e-10) |
        (np.abs(z) < 1e-10) | (np.abs(z - Lz) < 1e-10)
    )[0]
    Ii = np.setdiff1d(np.arange(XYtot.shape[0]), Ib)

    if dbg > 0:
        print("S made, subselecting")

    Sii = Stot[np.ix_(Ii, Ii)]
    Sib = Stot[np.ix_(Ii, Ib)]
    return Sii, Sib, XYtot, Ii, Ib