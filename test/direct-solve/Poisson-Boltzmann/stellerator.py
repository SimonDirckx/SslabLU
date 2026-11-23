"""
Gluing hpsmultidomain patches into a 5-patch cylinder (square core + 4 arched
patches) for general second-order PDOs.  Leaf fluxes are converted from
hpsmultidomain's reference-coordinate normal derivatives to a PHYSICAL flux, so
that the Jacobian jump across patch interfaces is handled correctly.

Cylinder: radius Rc (default 1), z in [0, L].  Patch reference boxes:
    core : (x, y, z)    in [-s Rc, s Rc]^2 x [0, L]    (identity map)
    arch : (xi, eta, z) in [-1, 1]^2 x [0, L]          (blending map on the unit
           disk, scaled by Rc and rotated by k*pi/2)

Physical PDE (PDOPhys), either
    form="nondiv":  -sum_ij a_ij d_i d_j u          + b.grad u + c u = f
    form="div"   :  -div(a grad u)                  + b.grad u + c u = f
with a symmetric positive definite.  The divergence form is expanded with
b_eff = b - div(a),  (div a)_j = sum_i d_i a_ij.

In reference coordinates y = y(x), with rows grad(y_k) of J^{-1}:
    A~_kl = grad(y_k)^T a grad(y_l)
    b~_k  = sum_i b_eff_i d_i y_k - sum_ij a_ij d_i d_j y_k
hpsmultidomain convention:
    -(c11 u_11 + c22 u_22 + c33 u_33 + c12 u_12 + c13 u_13 + c23 u_23)
    + c1 u_1 + c2 u_2 + c3 u_3 + c u
so c_kk = A~_kk, c_kl = 2 A~_kl (k<l), c_k = b~_k.

Interface flux options (build_patch(..., flux=...)):
    "normal"    : physical outward normal derivative  n . grad u
    "conormal"  : physical conormal flux              n . a grad u
    "reference" : hpsmultidomain's reference normal derivative, unconverted
                  (only correct inside one patch; kept as a negative control)
"""
import sys
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import sympy as sym
import torch

import functools
import types
from numpy.polynomial import legendre as _leg
from arch_stretch import blend_sym, xi_of_blend
import solver.hpsmultidomain.hpsmultidomain.hps_leaf_disc as leaf
from solver.hpsmultidomain.hpsmultidomain.hps_multidomain_disc import HPS_Multidomain   # noqa: E402
from solver.hpsmultidomain.hpsmultidomain.pdo import PDO_3d                              # noqa: E402

FACE_AXIS = [0, 0, 1, 1, 2, 2]          # L R D U B F
FACE_SIGN = [-1, +1, -1, +1, -1, +1]


# ----------------------------------------------------------------- PDE
@dataclass
class PDOPhys:
    """General second-order PDO in physical (x, y, z) coordinates.

    All callables take physical points X of shape (N, 3) (numpy) and return
        a: (N, 3, 3) symmetric positive definite,  b: (N, 3),  c: (N,),  f: (N,),
        div_a: (N, 3) with (div a)_j = sum_i d_i a_ij   (needed for form="div").
    """
    a: Callable
    b: Optional[Callable] = None
    c: Optional[Callable] = None
    f: Optional[Callable] = None
    div_a: Optional[Callable] = None
    form: str = "nondiv"

    def __post_init__(self):
        if self.form not in ("nondiv", "div"):
            raise ValueError("form must be 'nondiv' or 'div'")
        if self.form == "div" and self.div_a is None:
            raise ValueError("form='div' needs div_a = (sum_i d_i a_ij)_j")

    @staticmethod
    def laplace_plus_c(cfun, f=None):
        """-Lap u + c u = f  (backward compatible with the old cfun interface)."""
        eye = lambda X: np.broadcast_to(np.eye(3), (len(X), 3, 3)).copy()
        return PDOPhys(a=eye, c=cfun, f=f)

    def b_eff(self, X):
        b = np.zeros((len(X), 3)) if self.b is None else np.asarray(self.b(X), float)
        if self.form == "div":
            b = b - np.asarray(self.div_a(X), float)
        return b


def as_pdophys(pde):
    return pde if isinstance(pde, PDOPhys) else PDOPhys.laplace_plus_c(pde)


# ----------------------------------------------------------------- geometry
@functools.lru_cache(maxsize=None)
def _arch_symbolic(s, stretch=0.0):
    """Arch map and metric.  stretch: radial clustering parameter beta
    (arch_stretch.py); the blend parameter is x = g(xi), x = xi for 0."""
    xi, eta = sym.symbols("xi eta", real=True)
    th = sym.pi * eta / 4
    x = blend_sym(xi, stretch)
    u = (1 - x) / 2 * s + (1 + x) / 2 * sym.cos(th)
    v = (1 - x) / 2 * s * eta + (1 + x) / 2 * sym.sin(th)
    J = sym.Matrix([[sym.diff(u, xi), sym.diff(u, eta)],
                    [sym.diff(v, xi), sym.diff(v, eta)]])
    detJ = J.det()
    Jinv = J.inv()                     # rows = grad(xi), grad(eta) in (x', y')
    gi = Jinv * Jinv.T                 # inverse metric g^{ij}
    # Voss-Weyl: Lap(y_i) = (1/detJ) d_j (detJ g^{ij})
    lap = [sum(sym.diff(detJ * gi[i, j], (xi, eta)[j]) for j in range(2)) / detJ
           for i in range(2)]
    # Hessians of the inverse map: d_{x'_i} d_{x'_j} y_k = sum_m d_{y_m}(Jinv[k,i]) Jinv[m,j]
    ref = (xi, eta)
    hess = {}
    for k in range(2):
        for i in range(2):
            for j in range(i, 2):
                hess[f"H{k}{i}{j}"] = sum(sym.diff(Jinv[k, i], ref[m]) * Jinv[m, j]
                                          for m in range(2))
    f = lambda e: sym.lambdify((xi, eta), e, "numpy")   # no simplify: it creates 0/0 forms
    out = dict(g11=f(gi[0, 0]), g22=f(gi[1, 1]), g12=f(gi[0, 1]),
               Ji11=f(Jinv[0, 0]), Ji12=f(Jinv[0, 1]),
               Ji21=f(Jinv[1, 0]), Ji22=f(Jinv[1, 1]),
               lap1=f(lap[0]), lap2=f(lap[1]), detJ=f(detJ),
               u=f(u), v=f(v))
    out.update({key: f(e) for key, e in hess.items()})
    return out


class Patch:
    """One patch: reference box, map to physical space, metric, PDO."""

    def __init__(self, kind, s, L, rot=0, nleaf=(2, 2, 2), sym_arch=None, Rc=1.0, stretch=0.0):
        self.kind, self.s, self.L, self.rot, self.Rc = kind, s, L, rot, Rc
        # radial clustering of the arch (arch_stretch.py); 0 for the core
        self.stretch = 0.0 if kind == "core" else float(stretch)
        if kind == "core":
            self.box = np.array([[-s * Rc, s * Rc], [-s * Rc, s * Rc], [0.0, L]])
        else:
            self.box = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, L]])
            self.S = sym_arch
        self.nleaf = np.array(nleaf)
        self.a = (self.box[:, 1] - self.box[:, 0]) / (2 * self.nleaf)

    def ref_from_blend(self, X):
        """Copy of reference points X (N,3) whose arch axis 0 is given as the
        UNSTRETCHED blend parameter x, converted to xi = g^{-1}(x).  Lets
        resolution-independent point sets (probes) refer to the same physical
        points for any stretch.  Identity for the core / stretch 0."""
        X = np.array(X, float)
        if self.kind != "core" and self.stretch != 0:
            X[:, 0] = xi_of_blend(X[:, 0], self.stretch)
        return X

    def _R(self):
        c, sn = np.cos(self.rot * np.pi / 2), np.sin(self.rot * np.pi / 2)
        return np.array([[c, -sn], [sn, c]])

    def _ev(self, key, X):
        return self.S[key](X[:, 0], X[:, 1]) * np.ones(len(X))

    # reference (N,3) numpy -> physical (N,3)
    def to_phys(self, X):
        X = np.asarray(X)
        if self.kind == "core":
            return X.copy()
        uv = self.Rc * np.c_[self._ev("u", X), self._ev("v", X)] @ self._R().T
        return np.c_[uv, X[:, 2]]

    # inverse metric G (N,3,3) at reference points
    def ginv(self, X):
        Gr = self.grad_ref(X)
        return Gr @ Gr.transpose(0, 2, 1)

    # rows = physical gradients of the reference coordinates, grad(y_k), (N,3,3)
    def grad_ref(self, X):
        X = np.asarray(X); N = len(X)
        Gr = np.zeros((N, 3, 3)); Gr[:, 2, 2] = 1.0
        if self.kind == "core":
            Gr[:, 0, 0] = Gr[:, 1, 1] = 1.0
            return Gr
        Ji = np.empty((N, 2, 2))
        for a, b, key in [(0, 0, "Ji11"), (0, 1, "Ji12"), (1, 0, "Ji21"), (1, 1, "Ji22")]:
            Ji[:, a, b] = self._ev(key, X)
        # x = Rc * Rot x'  =>  grad_x y = Rot grad_x' y / Rc
        Gr[:, :2, :2] = (Ji @ self._R().T) / self.Rc
        return Gr

    # physical Hessians of the reference coordinates, H[n,k,i,j] = d_i d_j y_k
    def hess_ref(self, X):
        X = np.asarray(X); N = len(X)
        H = np.zeros((N, 3, 3, 3))
        if self.kind == "core":
            return H
        R = self._R()
        for k in range(2):
            Hp = np.empty((N, 2, 2))
            Hp[:, 0, 0] = self._ev(f"H{k}00", X)
            Hp[:, 1, 1] = self._ev(f"H{k}11", X)
            Hp[:, 0, 1] = Hp[:, 1, 0] = self._ev(f"H{k}01", X)
            H[:, k, :2, :2] = R @ Hp @ R.T / self.Rc ** 2
        return H

    def ref_a(self, X, pde):
        """Reference-coordinate second-order coefficient A~ (N,3,3) only (no b,
        c, f evaluation: used for the flux conversion at face slots, where the
        zero-order term may not be defined, e.g. a nodal c)."""
        X = np.asarray(X)
        Gr = self.grad_ref(X)
        return Gr @ np.asarray(pde.a(self.to_phys(X)), float) @ Gr.transpose(0, 2, 1)

    def ref_coeffs(self, X, pde):
        """Reference-coordinate coefficients at reference points X:
        A~ (N,3,3), b~ (N,3), c (N,), f (N,)."""
        X = np.asarray(X); N = len(X)
        x = self.to_phys(X)
        a = np.asarray(pde.a(x), float)
        Gr = self.grad_ref(X)
        At = Gr @ a @ Gr.transpose(0, 2, 1)
        bt = np.einsum("nki,ni->nk", Gr, pde.b_eff(x)) \
            - np.einsum("nij,nkij->nk", a, self.hess_ref(X))
        c = np.zeros(N) if pde.c is None else np.asarray(pde.c(x), float) * np.ones(N)
        f = np.zeros(N) if pde.f is None else np.asarray(pde.f(x), float) * np.ones(N)
        return At, bt, c, f

    def pdo(self, pde):
        """hpsmultidomain PDO_3d in this patch's reference coordinates.
        pde: PDOPhys, or (backward compatible) a callable c(x) for -Lap u + c u."""
        pde = as_pdophys(pde)
        cache = {}

        def coeffs(xx):
            # hpsmultidomain calls every coefficient on the same points in turn
            if "xx" in cache and cache["xx"].shape == xx.shape and torch.equal(cache["xx"], xx):
                return cache["val"]
            val = self.ref_coeffs(xx.detach().cpu().numpy(), pde)
            cache["xx"], cache["val"] = xx.detach().clone(), val
            return val

        # coefficients go back to the device hpsmultidomain evaluates on (GPU
        # for reduced_gpu); the metric itself is evaluated in numpy on the host
        T = lambda arr, xx: torch.from_numpy(np.ascontiguousarray(arr, dtype=float)).to(xx.device)
        A_ = lambda k, l, fac=1.0: (lambda xx: T(fac * coeffs(xx)[0][:, k, l], xx))
        b_ = lambda k: (lambda xx: T(coeffs(xx)[1][:, k], xx))
        c_ = lambda xx: T(coeffs(xx)[2], xx)
        # c12/c13/c23 always given (possibly zero): keeps the Gauss-node path
        return PDO_3d(c11=A_(0, 0), c22=A_(1, 1), c33=A_(2, 2),
                      c12=A_(0, 1, 2.0), c13=A_(0, 2, 2.0), c23=A_(1, 2, 2.0),
                      c1=b_(0), c2=b_(1), c3=b_(2), c=c_)

    def body(self, pde):
        """Body-load callable for hpsmultidomain (reference points -> f), or None."""
        pde = as_pdophys(pde)
        if pde.f is None:
            return None
        return lambda xx: torch.from_numpy(np.asarray(
            pde.f(self.to_phys(xx.detach().cpu().numpy())), float) * np.ones(xx.shape[0])).to(xx.device)


def make_cylinder(s=0.5, L=1.0, n=2, m=2, nz=2, Rc=1.0, stretch=0.0):
    """Cylinder of radius Rc and length L.  Core n x n x nz leaves; arches: m
    leaves radially (xi), n along eta, nz in z.  stretch: radial clustering of
    the arch leaves toward the wall (arch_stretch.py; uniform in xi, graded
    physically with ratio exp(-stretch/m))."""
    S = _arch_symbolic(float(s), float(stretch))
    core = Patch("core", s, L, nleaf=(n, n, nz), Rc=Rc)
    arches = [Patch("arch", s, L, rot=k, nleaf=(m, n, nz), sym_arch=S, Rc=Rc, stretch=stretch)
              for k in range(4)]
    return [core] + arches


# ----------------------------------------------------------------- helpers
def diffmat_1d(x):
    """Lagrange differentiation matrix on arbitrary distinct nodes x."""
    x = np.asarray(x); n = len(x)
    w = np.array([1.0 / np.prod([x[j] - x[k] for k in range(n) if k != j]) for j in range(n)])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
        D[i, i] = -D[i].sum()
    return D


def face_layout(q):
    sizes = [q[1] * q[2]] * 2 + [q[0] * q[2]] * 2 + [q[0] * q[1]] * 2
    offs = np.concatenate([[0], np.cumsum(sizes)])
    return sizes, offs


def _tangential_patterns(Xleaf, offs):
    """For one leaf's face nodes (identical layout in every leaf of a patch):
    sparse pattern (rows, cols, vals) of d/dy_j on each face, for both tangential j."""
    pat = {}
    for f in range(6):
        k = FACE_AXIS[f]
        loc = np.arange(offs[f], offs[f + 1])
        tang = [j for j in range(3) if j != k]
        for j in tang:
            other = [t for t in tang if t != j][0]
            uj, ij = np.unique(np.round(Xleaf[loc, j], 12), return_inverse=True)
            _, io = np.unique(np.round(Xleaf[loc, other], 12), return_inverse=True)
            D1 = diffmat_1d(uj)
            same = io[:, None] == io[None, :]
            r, c = np.nonzero(same)
            pat[(f, j)] = (loc[r], loc[c], D1[ij[r], ij[c]])
    return pat


# ----------------------------------------------------------------- patch build
def flux_conversion(P, X, q, nb, pde, flux):
    """Row conversion of a patch's leaf DtN maps (reference normal derivative)
    to a physical flux:  Aphys = diag(w) A + B.  Independent of the zero-order
    coefficient.  X: reference slot coordinates (nb * size_ext, 3), q: face
    node counts per axis.  flux: "normal" or "conormal".

    flux = sigma * (M_kk d_k u + sum_{j!=k} M_kj d_j u) / sqrt(g^kk)."""
    _, offs = face_layout(q)
    size_ext = offs[-1]
    g = P.ginv(X)
    if flux == "normal":
        M = g
    elif flux == "conormal":
        M = P.ref_a(X, pde)
    else:
        raise ValueError(flux)
    loc = np.arange(len(X)) % size_ext
    face = np.searchsorted(offs, loc, side="right") - 1
    kax = np.array(FACE_AXIS)[face]
    n_ = np.arange(len(X))
    sq = np.sqrt(g[n_, kax, kax])
    w = M[n_, kax, kax] / sq                     # multiplies the DtN row (= sigma d_k u)

    pat = _tangential_patterns(X[:size_ext], offs)
    rows, cols, vals = [], [], []
    base = np.arange(nb) * size_ext
    for (f, j), (r, c, d) in pat.items():
        k = FACE_AXIS[f]
        R = (base[:, None] + r[None, :]).ravel()
        C = (base[:, None] + c[None, :]).ravel()
        coef = FACE_SIGN[f] * M[R, k, j] / sq[R]
        rows.append(R); cols.append(C); vals.append(coef * np.tile(d, nb))
    B = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(X), len(X)))
    B.eliminate_zeros()
    return w, B


def build_patch(P, p, pde, device=torch.device("cpu"), flux="normal", convert=None):
    """Leaf DtN maps of one patch with rows converted to a physical flux.

    flux: "normal" (n.grad u), "conormal" (n.a grad u) or "reference" (unconverted).
    convert: deprecated alias; convert=False means flux="reference".
    Returns dict with the converted matrix A, per-slot body-load term h,
    reference slot coordinates, the hps object and the patch.
    """
    if convert is not None:
        flux = "normal" if convert else "reference"
    pde = as_pdophys(pde)
    # p: one order for all three reference axes, or (p0, p1, p2) per axis
    pv = np.array([p, p, p]) if np.ndim(p) == 0 else np.asarray(p, dtype=int).reshape(3)
    hps = HPS_Multidomain(P.pdo(pde), torch.tensor(P.box), P.a, pv, 3)
    assert hps.interpolate, "Gauss face nodes required for gluing"
    A, _ = hps.sparse_mat(device)
    A = sp.csr_matrix(A)
    X = hps.xx_ext.numpy()                         # reference slot coords
    nb = int(hps.nboxes); q = np.array(hps.q)
    _, offs = face_layout(q)
    size_ext = offs[-1]

    # per-slot body-load term: -(reference flux of the particular solution)
    ffun = P.body(pde)
    if ffun is not None:
        h = hps.get_DtNs(device, mode="reduce_body", ff_body_func=ffun)
        h = h.detach().cpu().numpy().reshape(-1)
        # hpsmultidomain returns body loads as complex by default; keep real if exact
        if np.iscomplexobj(h) and not np.any(h.imag):
            h = h.real.copy()
    else:
        h = np.zeros(len(X))

    if flux == "reference":
        return dict(A=A, h=h, Xref=X, hps=hps, P=P, ffun=ffun, flux=flux, w=np.ones(len(X)))

    w, B = flux_conversion(P, X, q, nb, pde, flux)
    Aphys = sp.diags(w) @ A + B
    # particular solution has zero face data: only the normal part survives
    # w: per-slot weight converting a reference body-load reduction to the
    # physical flux (used by CylinderSlab(keep_leaf=True).reduce_body)
    return dict(A=Aphys.tocsr(), h=w * h, Xref=X, hps=hps, P=P, ffun=ffun, flux=flux, w=w)


# ----------------------------------------------------------------- gluing
def glue(built, tol=1e-9):
    Xphys = np.vstack([b["P"].to_phys(b["Xref"]) for b in built])
    offsets = np.concatenate([[0], np.cumsum([len(b["Xref"]) for b in built])])
    tree = cKDTree(Xphys)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    # union-find on pairs
    parent = np.arange(len(Xphys))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(Xphys))])
    _, labels = np.unique(roots, return_inverse=True)
    ndof = labels.max() + 1
    Gm = sp.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)),
                       shape=(len(labels), ndof))
    Aall = sp.block_diag([b["A"] for b in built], format="csr")
    Aglob = (Gm.T @ Aall @ Gm).tocsr()
    hglob = Gm.T @ np.concatenate([b["h"] for b in built])
    counts = np.bincount(labels)
    Xdof = np.zeros((ndof, 3)); Xdof[labels] = Xphys
    return dict(A=Aglob, h=hglob, G=Gm, labels=labels, counts=counts, Xdof=Xdof,
                Xslot=Xphys, offsets=offsets)


def solve_dirichlet(gl, built, ubc, device=torch.device("cpu")):
    """Solve with Dirichlet data ubc(xyz) on single-slot (boundary) unknowns,
    then recover leaf interiors (with body load).  Returns per-patch (xyz, u)
    on all leaf nodes, plus the interface solution."""
    A, counts = gl["A"], gl["counts"]
    Ib = np.flatnonzero(counts == 1); Ic = np.flatnonzero(counts == 2)
    ub = np.asarray(ubc(gl["Xdof"][Ib]))
    u = np.zeros(A.shape[0], dtype=np.result_type(A.dtype, gl["h"].dtype, ub.dtype))
    u[Ib] = ub
    rhs = gl["h"][Ic] - A[Ic][:, Ib] @ u[Ib]
    u[Ic] = spla.spsolve(A[Ic][:, Ic].tocsc(), rhs)
    uslots = gl["G"] @ u
    out = []
    for i, b in enumerate(built):
        hps = b["hps"]
        us = uslots[gl["offsets"][i]:gl["offsets"][i + 1]]
        uu = torch.from_numpy(us[hps.I_unique.numpy()]).unsqueeze(-1)
        sol, _ = hps.solve(device, uu, ff_body_func=b["ffun"])
        xyz = b["P"].to_phys(hps.xx_tot.numpy())
        out.append((xyz, sol[:, 0].detach().cpu().numpy()))
    return out, u


def physical_normals(P, Xref):
    """Rows = grad(y_k) in physical coordinates at reference points (analytic,
    from the symbolic inverse Jacobian).  Used only by the test's sanity check;
    the solver itself uses the analytic metric P.ginv."""
    return P.grad_ref(np.asarray(Xref))
