"""
Twisted elliptic torus (full revolution), parametrized over a straight
cylinder whose axial coordinate is periodic.

Reference (OMS) coordinates  p = (t, x, y):
    t in [0, L), periodic  axial (slabs are stacked along p[:,0]),  L = 2 pi R0
    (x, y) in disk(Rc)     cross-section,                            Rc = sqrt(a b)
The cylinder is discretized by the 5-patch HPS cylinder (square core + 4
arches); bnds is only its bounding box over one period.

Forward map z = Phi(p)  (reference -> physical):
    zeta  = t / R0                    toroidal angle in [0, 2 pi) (arc length kept)
    X, Y  = (a/Rc) x, (b/Rc) y        disk -> ellipse (area kept exactly)
    alpha = twist * zeta              planar twist of the ellipse
    R     = R0 + X cos(alpha) - Y sin(alpha)
    z     = (R cos zeta, R sin zeta, X sin(alpha) + Y cos(alpha))

Inverse map y = Phi^{-1}(q)  (physical -> reference):
    zeta = atan2(q2, q1) in [0, 2 pi),  rho = sqrt(q1^2 + q2^2),  u = rho - R0
    P    =  u cos(alpha) + q3 sin(alpha)        (= X)
    Q    = -u sin(alpha) + q3 cos(alpha)        (= Y)
    y    = (R0 zeta, (Rc/a) P, (Rc/b) Q)
with derivatives (j = 1, 2, 3; zeta_3 = u_3 = 0, q3_j = delta_j3)
    P_j  =  u_j c + q3_j s + tw zeta_j Q
    Q_j  = -u_j s + q3_j c - tw zeta_j P
    P_jj =  u_jj c - 2 tw zeta_j u_j s + tw zeta_jj Q - tw^2 zeta_j^2 P
    Q_jj = -u_jj s - 2 tw zeta_j u_j c - tw zeta_jj P - tw^2 zeta_j^2 Q
Only the pure second derivatives are needed: transform_helmholtz_pdo uses
c_k = -Lap(y_k).

Periodicity
-----------
Phi(t + L, x, y) = Phi(t, x, y) needs the cross-section to come back
unrotated after one revolution, i.e. 2 pi twist must be a multiple of 2 pi:
twist is an integer (full turns of the ellipse per revolution).  A
half-integer twist also closes up geometrically (an ellipse is symmetric
under a half turn), but then (L, x, y) is identified with (0, -x, -y), and
the periodic coupling would need that permutation of the interface dofs; it
is not supported here.

With an integer twist the map is smooth across t = 0 == L, and every
derivative of the inverse map is independent of the branch of atan2 (zeta
enters only through its derivatives and through cos/sin(twist zeta)).  Only
the value y1 = R0 zeta is defined modulo L; it is returned in [0, L).

There are no end caps: the whole physical boundary is the tube wall.  The
periodic double slabs are [nH, (n+2)H], n = 0..N-1, so the last one,
[L-H, L+H], reaches past t = L.  That is fine: the map, and with it the
transformed PDO and the boundary data, are L-periodic in t.

All functions accept numpy, torch or jax arrays of shape (..., 3) and return
the same type; the (jax_avail, torch_avail) flags are kept for interface
compatibility with the other geometry modules.
"""
import numpy as np

from solver.hpsmultidomain.hpsmultidomain.geom import ParametrizedGeometry3D as ParametrizedGeometry3Dalt

R0, a, b, twist = 3.0, 1.2, 0.75, 2.0
Rc = np.sqrt(a * b)          # cylinder radius: equal cross-sectional area
L = 2 * np.pi * R0           # cylinder length (one period): equal centreline length
s = 0.5                      # core half-width / Rc (5-patch cross-section)
assert R0 > max(a, b), "torus must not self-intersect"
assert float(twist).is_integer(), (
    "the full torus is periodic in t only for an integer twist; a half-integer "
    "twist identifies (L, x, y) with (0, -x, -y), which is not supported")

bnds = [[0., -Rc, -Rc], [L, Rc, Rc]]


def _xp(p):
    """Array namespace of p (numpy, torch or jax.numpy)."""
    mod = type(p).__module__
    if mod.startswith("torch"):
        import torch
        return torch
    if mod.startswith("jax"):
        import jax.numpy as jnp
        return jnp
    return np


####################################
#      FORWARD MAP  z = Phi(p)
####################################

def _forward(p):
    xp = _xp(p)
    zeta = p[..., 0] / R0
    al = twist * zeta
    X, Y = (a / Rc) * p[..., 1], (b / Rc) * p[..., 2]
    R = R0 + X * xp.cos(al) - Y * xp.sin(al)
    return R * xp.cos(zeta), R * xp.sin(zeta), X * xp.sin(al) + Y * xp.cos(al)


def z1(p, jax_avail=True, torch_avail=False):
    return _forward(p)[0]


def z2(p, jax_avail=True, torch_avail=False):
    return _forward(p)[1]


def z3(p, jax_avail=True, torch_avail=False):
    return _forward(p)[2]


####################################
#   INVERSE MAP  y = Phi^{-1}(q)
####################################

def _inverse(q):
    """Inverse map and its derivatives at physical points q.  Returns dict:
    'y': (y1, y2, y3),  'd': d[k][j] = d y_k / d q_j,  'dd': dd[k][j] = d^2 y_k / d q_j^2."""
    xp = _xp(q)
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    zero = 0. * q1
    rho2 = q1 * q1 + q2 * q2
    rho = xp.sqrt(rho2)
    zeta = xp.arctan2(q2, q1)
    zeta = xp.where(zeta < 0, zeta + 2 * np.pi, zeta)            # zeta in [0, 2pi)
    u = rho - R0
    al = twist * zeta
    c, sn = xp.cos(al), xp.sin(al)
    P = u * c + q3 * sn
    Q = -u * sn + q3 * c

    z_d = (-q2 / rho2, q1 / rho2, zero)
    z_dd = (2 * q1 * q2 / rho2 ** 2, -2 * q1 * q2 / rho2 ** 2, zero)
    u_d = (q1 / rho, q2 / rho, zero)
    u_dd = (q2 * q2 / rho ** 3, q1 * q1 / rho ** 3, zero)
    w_d = (zero, zero, zero + 1.)

    P_d = [u_d[j] * c + w_d[j] * sn + twist * z_d[j] * Q for j in range(3)]
    Q_d = [-u_d[j] * sn + w_d[j] * c - twist * z_d[j] * P for j in range(3)]
    P_dd = [u_dd[j] * c - 2 * twist * z_d[j] * u_d[j] * sn + twist * z_dd[j] * Q
            - twist ** 2 * z_d[j] ** 2 * P for j in range(3)]
    Q_dd = [-u_dd[j] * sn - 2 * twist * z_d[j] * u_d[j] * c - twist * z_dd[j] * P
            - twist ** 2 * z_d[j] ** 2 * Q for j in range(3)]

    ka, kb = Rc / a, Rc / b
    return {
        "y": (R0 * zeta, ka * P, kb * Q),
        "d": ([R0 * v for v in z_d], [ka * v for v in P_d], [kb * v for v in Q_d]),
        "dd": ([R0 * v for v in z_dd], [ka * v for v in P_dd], [kb * v for v in Q_dd]),
    }


def y1(p, jax_avail=True, torch_avail=False):
    return _inverse(p)["y"][0]


def y2(p, jax_avail=True, torch_avail=False):
    return _inverse(p)["y"][1]


def y3(p, jax_avail=True, torch_avail=False):
    return _inverse(p)["y"][2]


def _d(k, j):
    def f(p, jax_avail=True, torch_avail=False):
        return _inverse(p)["d"][k][j]
    f.__name__ = "y%d_d%d" % (k + 1, j + 1)
    return f


def _dd(k, j):
    def f(p, jax_avail=True, torch_avail=False):
        return _inverse(p)["dd"][k][j]
    f.__name__ = "y%d_d%dd%d" % (k + 1, j + 1, j + 1)
    return f


# first derivatives (y1_d3 = 0 identically)
y1_d1, y1_d2 = _d(0, 0), _d(0, 1)
y2_d1, y2_d2, y2_d3 = _d(1, 0), _d(1, 1), _d(1, 2)
y3_d1, y3_d2, y3_d3 = _d(2, 0), _d(2, 1), _d(2, 2)
# pure second derivatives (all d3d3 = 0 identically)
y1_d1d1, y1_d2d2 = _dd(0, 0), _dd(0, 1)
y2_d1d1, y2_d2d2 = _dd(1, 0), _dd(1, 1)
y3_d1d1, y3_d2d2 = _dd(2, 0), _dd(2, 1)


def contains(q):
    """True for physical points q (n, 3) strictly inside the torus."""
    q = np.asarray(q, dtype=np.float64)
    y = _inverse(q)["y"]
    return y[1] ** 2 + y[2] ** 2 < Rc ** 2


####################################
#   BOUNDARY (reference coordinates)
####################################

def gb_np(p):
    # periodic in t: no end caps, the tube wall is the only physical boundary
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    return p[:, 1] ** 2 + p[:, 2] ** 2 >= Rc ** 2 * (1 - 1e-8)


def gb(p, jax_avail=True, torch_avail=False):
    return gb_np(p)


def box_geom(jax_avail=True, torch_avail=False):
    if jax_avail:
        import jax.numpy as jnp
        return jnp.array(bnds)
    return np.array(bnds)


def bounds():
    return bnds


def param_geom(jax_avail=True, torch_avail=False, hpsalt=False):
    if not hpsalt:
        raise NotImplementedError("only the hpsmultidomain (hpsalt) geometry is provided")
    f = lambda g: (lambda p: g(p, jax_avail, torch_avail))
    return ParametrizedGeometry3Dalt(
        box_geom(jax_avail, torch_avail),
        z1=f(z1), z2=f(z2), z3=f(z3),
        y1=f(y1), y2=f(y2), y3=f(y3),
        y1_d1=f(y1_d1), y1_d2=f(y1_d2),
        y2_d1=f(y2_d1), y2_d2=f(y2_d2), y2_d3=f(y2_d3),
        y3_d1=f(y3_d1), y3_d2=f(y3_d2), y3_d3=f(y3_d3),
        y1_d1d1=f(y1_d1d1), y1_d2d2=f(y1_d2d2),
        y2_d1d1=f(y2_d1d1), y2_d2d2=f(y2_d2d2),
        y3_d1d1=f(y3_d1d1), y3_d2d2=f(y3_d2d2),
    )


####################################
#       slab functions
####################################

def dSlabs(N):
    """Periodic decomposition of [0, L) into N single slabs of width H = L/N.

    Returns N double slabs [nH, (n+2)H] with interface n at t = (n+1)H (the
    last interface sits at t = L == 0), cyclic connectivity
    [(n-1) % N, (n+1) % N], and H.  Slab N-1, [L-H, L+H], extends past
    t = L; its right face at t = L+H is slab 0's interface at t = H.
    """
    if N < 3:
        raise ValueError("a periodic decomposition needs N >= 3 slabs, got %d" % N)
    H = (bnds[1][0] - bnds[0][0]) / N
    slabs = [[[bnds[0][0] + n * H, bnds[0][1], bnds[0][2]],
              [bnds[0][0] + (n + 2) * H, bnds[1][1], bnds[1][2]]] for n in range(N)]
    connectivity = [[(n - 1) % N, (n + 1) % N] for n in range(N)]
    return slabs, connectivity, H


####################################
#       validate
####################################

def _wrap(d):
    """Axial differences modulo the period, in [-L/2, L/2)."""
    return d - L * np.floor(d / L + 0.5)


def check_param(n=200, seed=0):
    """y(z(p)) = p (y1 mod L), Phi(t + L) = Phi(t), and finite-difference
    checks of every derivative, including points across the seam t = 0 == L."""
    rng = np.random.default_rng(seed)
    r = Rc * np.sqrt(rng.uniform(size=n)); th = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(0, L, n)
    t[: n // 4] = rng.uniform(-1e-3, 1e-3, n // 4) % L          # at the seam
    p = np.c_[t, r * np.cos(th), r * np.sin(th)]
    q = np.stack(_forward(p), axis=-1)
    inv = _inverse(q)
    y = np.stack(inv["y"], -1)
    err_y = max(np.max(np.abs(_wrap(y[:, 0] - p[:, 0]))), np.max(np.abs(y[:, 1:] - p[:, 1:])))
    err_per = np.max(np.abs(np.stack(_forward(p + [L, 0., 0.]), -1) - q))
    h = 1e-4
    err_d = err_dd = 0.
    for j in range(3):
        e = np.zeros(3); e[j] = h
        yp, ym = _inverse(q + e)["y"], _inverse(q - e)["y"]
        for k in range(3):
            dp, dm = yp[k] - inv["y"][k], ym[k] - inv["y"][k]
            if k == 0:
                dp, dm = _wrap(dp), _wrap(dm)
            err_d = max(err_d, np.max(np.abs((dp - dm) / (2 * h) - inv["d"][k][j])))
            err_dd = max(err_dd, np.max(np.abs((dp + dm) / h ** 2 - inv["dd"][k][j])))
    print("err y(z(p)) - p = %.2e,  err periodicity = %.2e,  err dy = %.2e,  err ddy = %.2e"
          % (err_y, err_per, err_d, err_dd))
    return err_y, err_per, err_d, err_dd
