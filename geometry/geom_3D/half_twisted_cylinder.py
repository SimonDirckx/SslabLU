"""
Straight cylinder = computational domain of the half twisted elliptic torus
(see half_torus.HalfTwistedTorus), in OMS coordinates

        w = (z, x, y):   w0 in [0, L] axial,   (w1, w2) in the disk of radius Rc.

Same interface as cube.py (bnds, gb, dSlabs).  Slabs are stacked along w0.
"""
import numpy as np

# half twisted elliptic torus parameters (see half_torus.py)
R0, a, b, twist = 3.0, 1.2, 0.75, 2.0
Rc = np.sqrt(a * b)          # cylinder radius: equal cross-sectional area
L = np.pi * R0               # cylinder length: equal centreline length
s = 0.5                      # core half-width / Rc (5-patch cross-section)

bnds = [[0, -Rc, -Rc], [L, Rc, Rc]]   # bounding box (slab x-extents come from here)

_tol = 1e-10


def gb_np(p):
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    cap = (np.abs(p[:, 0] - bnds[0][0]) < _tol * L) | (np.abs(p[:, 0] - bnds[1][0]) < _tol * L)
    wall = p[:, 1] ** 2 + p[:, 2] ** 2 >= Rc ** 2 * (1 - 1e-8)
    return cap | wall


def gb(p, jax_avail=True, torch_avail=False):
    return gb_np(p)


def box_geom(jax_avail=True, torch_avail=True):
    return np.array(bnds)


def bounds():
    return bnds


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
