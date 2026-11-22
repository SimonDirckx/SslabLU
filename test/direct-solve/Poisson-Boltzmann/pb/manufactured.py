"""
Manufactured solution and nonlinearity for normalized PB, model (A):

    -Lap phi + G[phi] = f*  in Omega,   phi = phi* on dOmega,
    g(x, s) = Lam^2 N(psi) exp(s / tau(psi)),   d_s g = g / tau,
    psi(chi(t, x, y)) = (x^2 + y^2) / Rc^2,  N = 1 - psi/2,  tau = 1 - 0.4 psi,
    phi*(q) = A sin(k1 q1 + th) cos(k2 q2) exp(k3 q3) + B,
    -Lap phi* = (k1^2 + k2^2 - k3^2) (phi* - B).

All field functions take REFERENCE points W in OMS coordinates (t, x, y),
shape (N, 3) numpy, and map to physical space with the geometry module's
forward map.  psi lives in reference coordinates.
"""
import numpy as np
import torch


class PBManufactured:
    def __init__(self, geom, Lam=1.0, A=1.0, B=0.0, kappa=(0.7, 0.5, 0.4), theta=0.3):
        self.geom = geom                          # e.g. geometry.geom_3D.full_twisted_torus
        self.Lam, self.A, self.B, self.theta = float(Lam), float(A), float(B), float(theta)
        self.kappa = tuple(float(k) for k in kappa)

    # ---- geometry
    def phys(self, W):
        W = np.asarray(W, dtype=np.float64)
        g = self.geom
        return np.stack([g.z1(W, False), g.z2(W, False), g.z3(W, False)], axis=-1)

    def psi(self, W):
        W = np.asarray(W, dtype=np.float64)
        return (W[:, 1] ** 2 + W[:, 2] ** 2) / self.geom.Rc ** 2

    @staticmethod
    def N(psi):
        return 1.0 - 0.5 * psi

    @staticmethod
    def tau(psi):
        return 1.0 - 0.4 * psi

    # ---- manufactured solution, physical points q
    def phistar_q(self, q):
        k1, k2, k3 = self.kappa
        return self.A * np.sin(k1 * q[:, 0] + self.theta) * np.cos(k2 * q[:, 1]) \
            * np.exp(k3 * q[:, 2]) + self.B

    def lap_phistar_q(self, q):
        k1, k2, k3 = self.kappa
        return -(k1 ** 2 + k2 ** 2 - k3 ** 2) * (self.phistar_q(q) - self.B)

    # ---- fields at reference points
    def phistar(self, W):
        return self.phistar_q(self.phys(W))

    def lap_phistar(self, W):
        return self.lap_phistar_q(self.phys(W))

    def g(self, W, s, psi=None):
        psi = self.psi(W) if psi is None else psi
        return self.Lam ** 2 * self.N(psi) * np.exp(s / self.tau(psi))

    def dg(self, W, s, psi=None):
        psi = self.psi(W) if psi is None else psi
        return self.g(W, s, psi) / self.tau(psi)

    def fstar(self, W):
        return -self.lap_phistar(W) + self.g(W, self.phistar(W))

    # ---- calibration / checks
    def calibrate_A(self, W, spread=3.0):
        """Set A so that phi*/tau varies by `spread` over the points W (B kept)."""
        A0, B0 = self.A, self.B
        self.A, self.B = 1.0, 0.0
        v = self.phistar(W) / self.tau(self.psi(W))
        self.A, self.B = A0, B0
        self.A = spread / (v.max() - v.min())
        return self.A


# ---------------------------------------------------------------- PDO
def transform_pdo(param_geom, c_fun):
    """PDO_3d (OMS reference coordinates) for  -Lap u + c u  on the mapped
    domain: the second/first-order terms of transform_helmholtz_pdo, and the
    zero-order term c_fun(W) evaluated at the REFERENCE points W (numpy (N,3)
    -> (N,)).  transform_pdo(pg, lambda W: -kh**2 + 0*W[:,0]) reproduces
    pg.transform_helmholtz_pdo(-kh^2 bfield, kh); see check_transform_pdo."""
    pdo = param_geom.transform_helmholtz_pdo(lambda q, kh: 0.0 * q[..., 0], 0.0)
    if c_fun is None:              # geometry only (CondensedSlab(fast=True) adds c)
        pdo.c = None
        return pdo

    def c(xx):
        W = xx.detach().cpu().numpy() if torch.is_tensor(xx) else np.asarray(xx)
        v = np.asarray(c_fun(W), dtype=np.float64) * np.ones(len(W))
        return torch.from_numpy(v).to(xx.device) if torch.is_tensor(xx) else v

    pdo.c = c
    return pdo


def check_transform_pdo(param_geom, kh=3.0, npts=50, seed=0):
    """max |difference| over all coefficients between transform_pdo(-kh^2) and
    transform_helmholtz_pdo(-kh^2) at random reference points (expect 0)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(param_geom.bounds, dtype=np.float64)
    W = torch.from_numpy(lo + (hi - lo) * rng.uniform(size=(npts, 3)) * [1, 0.5, 0.5])
    ref = param_geom.transform_helmholtz_pdo(lambda q, k: -k * k * torch.ones_like(q[..., 0]), kh)
    new = transform_pdo(param_geom, lambda X: -kh * kh + 0 * X[:, 0])
    err = 0.0
    for name in ("c11", "c22", "c33", "c12", "c13", "c23", "c1", "c2", "c3", "c"):
        f, g = getattr(ref, name), getattr(new, name)
        if f is None or g is None:
            assert f is None and g is None, name
            continue
        err = max(err, float(torch.max(torch.abs(torch.as_tensor(f(W)) - torch.as_tensor(g(W))))))
    return err
