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

    def coeffs(X):
        X = np.asarray(X, dtype=np.float64)
        if "X" in cache and cache["X"].shape == X.shape and np.array_equal(cache["X"], X):
            return cache["val"]
        N = len(X)
        W = torch.from_numpy(np.ascontiguousarray(X[:, [2, 0, 1]]))
        aw = np.zeros((N, 3, 3))
        for k, f in enumerate((pdo.c11, pdo.c22, pdo.c33)):
            aw[:, k, k] = ev(f, W, N)
        for (k, l), f in (((0, 1), pdo.c12), ((0, 2), pdo.c13), ((1, 2), pdo.c23)):
            aw[:, k, l] = aw[:, l, k] = 0.5 * ev(f, W, N)
        bw = np.stack([ev(f, W, N) for f in (pdo.c1, pdo.c2, pdo.c3)], axis=1)
        c = ev(pdo.c, W, N)
        p = _W_OF_CYL
        val = (aw[:, p][:, :, p], bw[:, p], c)
        cache["X"], cache["val"] = X.copy(), val
        return val

    return PDOPhys(a=lambda X: coeffs(X)[0], b=lambda X: coeffs(X)[1],
                   c=lambda X: coeffs(X)[2], form="nondiv")


class CylinderSlab:
    def __init__(self, geom, pde, p, Rc=1.0, s=0.5, n=4, m=1, nz=2,
                 flux="normal", device=torch.device("cpu"), glue_tol=1e-9):
        geom = np.asarray(geom, dtype=np.float64)
        z0, z1 = float(geom[0][0]), float(geom[1][0])
        if not isinstance(pde, PDOPhys):
            pde = pdo3d_to_cyl(pde)
        if nz % 2:
            raise ValueError("nz must be even: the midplane w0 = xc must be a "
                             "leaf face (OMS takes Ic there)")

        # ---- patches of this double slab: z in [z0, z1] --------------------
        patches = make_cylinder(s=s, L=z1 - z0, n=n, m=m, nz=nz, Rc=Rc)
        for P in patches:
            P.box[2] = [z0, z1]
            P.a = (P.box[:, 1] - P.box[:, 0]) / (2 * P.nleaf)

        built = [build_patch(P, p, pde, device=device, flux=flux) for P in patches]
        gl = glue(built, tol=glue_tol * max(1.0, Rc, abs(z0), abs(z1)))
        del built                                  # drop the hps objects

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

    @staticmethod
    def to_cyl(w):
        """OMS coordinates (z, x, y) -> cylinder coordinates (x, y, z)."""
        w = w.detach().cpu().numpy() if torch.is_tensor(w) else np.asarray(w)
        return w[:, [1, 2, 0]]
