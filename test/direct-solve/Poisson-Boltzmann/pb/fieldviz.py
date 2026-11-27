"""
Field visualization data for a converged (or prescribed) potential on the
stellarator: everything is evaluated from the solver's own leaf polynomials
(no re-interpolation onto a plotting grid), through the same patch and torus
maps the solver uses.

  sections  phi and psi on poloidal cross-sections at given toroidal angles:
            per patch a tensor grid in PATCH reference coordinates (core (x, y),
            arch (xi, eta)), mapped patch -> cylinder (x, y) -> torus (R, Z).
            The arch edge xi = 1 is the wall, so patch seams and the wall are
            grid lines of these grids.
  wall      the wall field E_n = -d phi / d n (n the outward unit normal) on a
            (toroidal angle, poloidal angle) grid over the full torus:
                grad_q phi = J^{-T} grad_p phi,      p = (t, x, y) torus reference,
                n          = J^{-T} grad_p rho / |J^{-T} grad_p rho|,   grad_p rho ~ (0, x, y),
            J = dq/dp from the torus map (geometry/geom_3D/stellerator.py) and
            grad_p phi = (patch metric)^T grad_(patch ref) phi from the leaf
            polynomial derivative (one-sided at the wall face, spectral).

Field sources (FieldSampler):
  from_ops(slabs, geom)        list of (CylinderSlabOps, phi): the fine OMS slabs
                               of a run (each point is taken from the slab whose
                               midplane is nearest, as for the probes), or one
                               periodic slab over [0, L) (the global coarse problem);
  from_leafgrid(lg, phi, geom) a pb.prolong.LeafGrid over [0, L) with nodal values
                               (tests: a prescribed phi sampled at the nodes).

collect(...) returns a dict of plain arrays; save(path, data) writes it as .npz
for plot_fields.py (numpy + matplotlib only).
"""
import numpy as np

from pb.prolong import lagrange_matrix


def _diffmat(u):
    """Differentiation matrix of the Lagrange interpolant on nodes u."""
    u = np.asarray(u, float)
    n = len(u)
    w = 1.0 / np.prod(u[:, None] - u[None, :] + np.eye(n), axis=1)
    D = np.zeros((n, n))
    off = ~np.eye(n, dtype=bool)
    dx = u[:, None] - u[None, :]
    D[off] = ((w[None, :] / w[:, None]) / np.where(off, dx, 1.0))[off]
    D[np.diag_indices(n)] = -D.sum(axis=1)
    return D


def leaf_eval(P, grid, off, phi, X, grad=False):
    """Leaf polynomial of patch P (leaf grid `grid` (nbox, npl, 3) in patch
    reference coordinates, field phi[off:]) at patch-reference points X (N, 3).
    Returns values (N,), and with grad also d/d(patch ref) (N, 3)."""
    X = np.asarray(X, float)
    box = np.asarray(P.box, float)
    nleaf = np.asarray(P.nleaf, int)
    ia = [np.clip(np.floor((X[:, a] - box[a, 0]) / (box[a, 1] - box[a, 0]) * nleaf[a]).astype(int),
                  0, nleaf[a] - 1) for a in range(3)]
    b = np.ravel_multi_index(tuple(ia), nleaf)            # box = (i n1 + j) n2 + k
    npl = grid.shape[1]
    lo, hi = grid.min(axis=1), grid.max(axis=1)           # leaf boxes (nodes include faces)
    rel0 = (grid[0] - lo[0]) / (hi[0] - lo[0])
    r = (X - lo[b]) / (hi[b] - lo[b])
    V = np.asarray(phi)[off:off + grid.shape[0] * npl].reshape(-1, npl)[b]
    Ls, Ds = [], []
    for a in range(3):
        nodes, idx = np.unique(np.round(rel0[:, a], 12), return_inverse=True)
        La = lagrange_matrix(nodes, r[:, a])
        Ls.append(La[:, idx])
        if grad:
            Ds.append((La @ _diffmat(nodes))[:, idx] / (hi[b, a] - lo[b, a])[:, None])
    val = np.einsum("kn,kn->k", Ls[0] * Ls[1] * Ls[2], V)
    if not grad:
        return val
    g = np.stack([np.einsum("kn,kn->k", Ds[0] * Ls[1] * Ls[2], V),
                  np.einsum("kn,kn->k", Ls[0] * Ds[1] * Ls[2], V),
                  np.einsum("kn,kn->k", Ls[0] * Ls[1] * Ds[2], V)], axis=1)
    return val, g


def torus_jacobian(geom, p):
    """q = (X, Y, Z) and J[n, a, b] = dq_a / dp_b at torus reference points p = (t, x, y)."""
    R, Z = geom.fields(p, order=1)
    g = 2 * np.pi / geom.L                                 # d (toroidal angle) / dt
    c, s = np.cos(g * p[:, 0]), np.sin(g * p[:, 0])
    J = np.empty((len(p), 3, 3))
    J[:, 0, 0] = R["t"] * c - g * R[""] * s
    J[:, 1, 0] = R["t"] * s + g * R[""] * c
    J[:, 2, 0] = Z["t"]
    for b, k in ((1, "x"), (2, "y")):
        J[:, 0, b] = R[k] * c
        J[:, 1, b] = R[k] * s
        J[:, 2, b] = Z[k]
    q = np.c_[R[""] * c, R[""] * s, Z[""]]
    return q, J


class FieldSampler:
    """entries: list of (leaf_grids, (z0, z1), phi); leaf_grids = [(P, grid, off)]
    for the 5 patches (same order in every entry)."""

    def __init__(self, entries, geom):
        self.entries, self.geom = entries, geom
        self.L = float(geom.L)
        self.centers = np.array([0.5 * (z0 + z1) for _, (z0, z1), _ in entries])

    @classmethod
    def from_ops(cls, slabs, geom):
        return cls([(ops.leaf_grids(), tuple(float(z) for z in ops.zbounds), phi) for ops, phi in slabs], geom)

    @classmethod
    def from_leafgrid(cls, lg, phi, geom):
        lgs = [(d["P"], d["grid"], lg.offsets[i]) for i, d in enumerate(lg.patches)]
        z = [float(v) for v in lgs[0][0].box[2]]
        return cls([(lgs, (z[0], z[1]), phi)], geom)

    @property
    def patches(self):
        return [P for P, _, _ in self.entries[0][0]]

    # ------------------------------------------------------------ evaluation
    def _owner(self, t):
        """Slab per global t (nearest midplane, periodic) and its local axial coordinate."""
        d = np.mod(t[:, None] - self.centers[None, :] + 0.5 * self.L, self.L) - 0.5 * self.L
        i = np.argmin(np.abs(d), axis=1)
        tloc = self.centers[i] + d[np.arange(len(t)), i]
        return i, tloc

    def eval(self, k, uv, t, grad=False):
        """phi in patch k at patch-reference cross-section coordinates uv (N, 2)
        and global axial coordinate t (N,).  With grad: also the gradient in
        torus reference coordinates p = (t, x, y) (N, 3)."""
        uv, t = np.atleast_2d(np.asarray(uv, float)), np.asarray(t, float) * np.ones(len(uv))
        owner, tloc = self._owner(np.mod(t, self.L))
        val = np.full(len(t), np.nan)
        gp = np.full((len(t), 3), np.nan)
        for i in np.unique(owner):
            sel = owner == i
            lgs, (z0, z1), phi = self.entries[i]
            P, grid, off = lgs[k]
            X = np.c_[uv[sel], tloc[sel]]
            assert np.all((X[:, 2] >= z0 - 1e-12) & (X[:, 2] <= z1 + 1e-12)), "point outside its slab"
            if grad:
                v, g = leaf_eval(P, grid, off, phi, X, grad=True)
                Gr = P.grad_ref(X)                        # rows: cylinder gradients of the patch coords
                gc = np.einsum("nki,nk->ni", Gr, g)       # (d/dx, d/dy, d/dz) in cylinder coords
                gp[sel] = gc[:, [2, 0, 1]]                # torus reference order (t, x, y)
            else:
                v = leaf_eval(P, grid, off, phi, X)
            val[sel] = v
        return (val, gp) if grad else val

    def cyl_xy(self, k, uv):
        """Cylinder (x, y) of patch-reference points uv of patch k."""
        P = self.patches[k]
        return P.to_phys(np.c_[uv, np.zeros(len(uv))])[:, :2]

    # ------------------------------------------------------------ data sets
    def section(self, ang, ns=41):
        """Cross-section at toroidal angle ang: per patch an ns x ns grid.
        Returns arrays of shape (5, ns, ns): R, Z, phi, psi, and W (5, ns, ns, 3)
        the torus reference points (t, x, y)."""
        t = self.L * ang / (2 * np.pi)
        Rc = float(self.geom.Rc)
        out = {k: [] for k in ("R", "Z", "phi", "psi", "W")}
        for k, P in enumerate(self.patches):
            a = np.linspace(P.box[0, 0], P.box[0, 1], ns)
            b = np.linspace(P.box[1, 0], P.box[1, 1], ns)
            A, B = np.meshgrid(a, b, indexing="ij")
            uv = np.c_[A.ravel(), B.ravel()]
            xy = self.cyl_xy(k, uv)
            W = np.c_[np.full(len(uv), t), xy]
            R, Z = self.geom.fields(W, order=0)
            sh = (ns, ns)
            out["R"].append(R[""].reshape(sh))
            out["Z"].append(Z[""].reshape(sh))
            out["phi"].append(self.eval(k, uv, W[:, 0]).reshape(sh))
            out["psi"].append(((xy[:, 0] ** 2 + xy[:, 1] ** 2) / Rc ** 2).reshape(sh))
            out["W"].append(W.reshape(ns, ns, 3))
        return {k: np.stack(v) for k, v in out.items()}

    def wall(self, nt=256, neta=32):
        """Wall grid (nt toroidal x 4 neta poloidal; arch k covers reference
        angles theta in [-pi/4, pi/4) + k pi/2).  Returns q (nt, nth, 3), the
        outward unit normal n, E_n = -d phi/dn, phi (wall values), the toroidal
        angle (nt,), the reference poloidal angle theta (nth,), W (nt, nth, 3)."""
        t = np.linspace(0.0, self.L, nt, endpoint=False)
        eta = np.linspace(-1.0, 1.0, neta, endpoint=False)
        cols = {k: [] for k in ("W", "phi", "gp")}
        theta = []
        for k in range(1, 5):                              # the four arches, in rot order
            P = self.patches[k]
            T, E = np.meshgrid(t, eta, indexing="ij")
            uv = np.c_[np.full(T.size, P.box[0, 1]), E.ravel()]   # xi = 1: the wall
            xy = self.cyl_xy(k, uv)
            v, g = self.eval(k, uv, T.ravel(), grad=True)
            cols["W"].append(np.c_[T.ravel(), xy].reshape(nt, neta, 3))
            cols["phi"].append(v.reshape(nt, neta))
            cols["gp"].append(g.reshape(nt, neta, 3))
            theta.append(np.pi * eta / 4 + (k - 1) * np.pi / 2)
        W = np.concatenate(cols["W"], axis=1)
        phi = np.concatenate(cols["phi"], axis=1)
        gp = np.concatenate(cols["gp"], axis=1).reshape(-1, 3)
        q, J = torus_jacobian(self.geom, W.reshape(-1, 3))
        JT = np.transpose(J, (0, 2, 1))
        gq = np.linalg.solve(JT, gp[..., None])[..., 0]
        grho = np.c_[np.zeros(len(gp)), W.reshape(-1, 3)[:, 1:]]
        nq = np.linalg.solve(JT, grho[..., None])[..., 0]
        nq /= np.linalg.norm(nq, axis=1, keepdims=True)
        En = -np.einsum("ni,ni->n", gq, nq)
        sh = W.shape[:2]
        return dict(q=q.reshape(*sh, 3), n=nq.reshape(*sh, 3), En=En.reshape(sh), phi=phi,
                    tor=2 * np.pi * t / self.L, theta=np.concatenate(theta), W=W)

    def collect(self, ncut=6, ns=41, nt=256, neta=32, label="", aref=1.0):
        """Sections at ncut toroidal angles over one field period + the wall."""
        nfp = int(self.geom.nfp)
        angs = np.linspace(0, 2 * np.pi / nfp, ncut, endpoint=False)
        secs = [self.section(a, ns) for a in angs]
        data = {"sec_" + k: np.stack([s[k] for s in secs]) for k in secs[0]}
        data.update({"wall_" + k: v for k, v in self.wall(nt, neta).items()})
        data.update(angs=angs, nfp=nfp, aref=float(aref), label=str(label))
        return data


def save(path, data):
    np.savez_compressed(path, **data)
