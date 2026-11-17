"""
Half twisted elliptic torus, parametrized over a straight cylinder.

Reference (OMS) coordinates  p = (t, x, y):
    t in [0, L]           axial (slabs are stacked along p[:,0]),  L = pi R0
    (x, y) in disk(Rc)    cross-section,                           Rc = sqrt(a b)
The cylinder is discretized by the 5-patch HPS cylinder (square core + 4
arches); bnds is only its bounding box.

Forward map z = Phi(p)  (reference -> physical):
    zeta  = t / R0                    toroidal angle in [0, pi] (arc length kept)
    X, Y  = (a/Rc) x, (b/Rc) y        disk -> ellipse (area kept exactly)
    alpha = twist * zeta              planar twist of the ellipse
    R     = R0 + X cos(alpha) - Y sin(alpha)
    z     = (R cos zeta, R sin zeta, X sin(alpha) + Y cos(alpha))

Inverse map y = Phi^{-1}(q)  (physical -> reference):
    zeta = atan2(q2, q1),  rho = sqrt(q1^2 + q2^2),  u = rho - R0
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

All functions accept numpy, torch or jax arrays of shape (..., 3) and return
the same type; the (jax_avail, torch_avail) flags are kept for interface
compatibility with the other geometry modules.
"""
import numpy as np

from solver.hpsmultidomain.hpsmultidomain.geom import ParametrizedGeometry3D as ParametrizedGeometry3Dalt

R0, a, b, twist = 3.0, 1.2, 0.75, 2.0
Rc = np.sqrt(a * b)          # cylinder radius: equal cross-sectional area
L = np.pi * R0               # cylinder length: equal centreline length
s = 0.5                      # core half-width / Rc (5-patch cross-section)
assert R0 > max(a, b), "torus must not self-intersect"

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
    zeta = xp.where(zeta < -np.pi / 2, zeta + 2 * np.pi, zeta)   # zeta in [-pi/2, 3pi/2)
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


####################################
#   BOUNDARY (reference coordinates)
####################################

def gb_np(p):
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    cap = (np.abs(p[:, 0] - bnds[0][0]) < 1e-10 * L) | (np.abs(p[:, 0] - bnds[1][0]) < 1e-10 * L)
    wall = p[:, 1] ** 2 + p[:, 2] ** 2 >= Rc ** 2 * (1 - 1e-8)
    return cap | wall


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

# N is number of single slabs; not periodic (end caps at t = 0 and t = L)
def dSlabs(N):
    dSlabs = []
    H = (bnds[1][0] - bnds[0][0]) / N
    connectivity = []
    for n in range(N - 1):
        c = bnds[0][0] + (n + 1) * H
        bnds_n = [[c - H, bnds[0][1], bnds[0][2]], [c + H, bnds[1][1], bnds[1][2]]]
        if n == 0:
            connectivity += [[-1, 1]]
        elif n == N - 2:
            connectivity += [[n - 1, -1]]
        else:
            connectivity += [[n - 1, n + 1]]
        dSlabs += [bnds_n]
    return dSlabs, connectivity, H


####################################
#       validate
####################################

def check_param(n=200, seed=0):
    """y(z(p)) = p, and finite-difference checks of every derivative."""
    rng = np.random.default_rng(seed)
    r = Rc * np.sqrt(rng.uniform(size=n)); th = rng.uniform(0, 2 * np.pi, n)
    p = np.c_[rng.uniform(0, L, n), r * np.cos(th), r * np.sin(th)]
    q = np.stack(_forward(p), axis=-1)
    inv = _inverse(q)
    err_y = np.max(np.abs(np.stack(inv["y"], -1) - p))
    h = 1e-4
    err_d = err_dd = 0.
    for j in range(3):
        e = np.zeros(3); e[j] = h
        yp, ym = _inverse(q + e)["y"], _inverse(q - e)["y"]
        for k in range(3):
            err_d = max(err_d, np.max(np.abs((yp[k] - ym[k]) / (2 * h) - inv["d"][k][j])))
            err_dd = max(err_dd, np.max(np.abs((yp[k] - 2 * inv["y"][k] + ym[k]) / h ** 2
                                               - inv["dd"][k][j])))
    print("err y(z(p)) - p = %.2e,  err dy = %.2e,  err ddy = %.2e" % (err_y, err_d, err_dd))
    return err_y, err_d, err_dd
