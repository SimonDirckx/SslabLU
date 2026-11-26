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


# ---------------------------------------------------------------- limited-regularity solution
class PBPedestal:
    """Manufactured solution with controlled regularity for the PB model of
    pb.profiles.PBProfiles (same N, tau, g, d_s g):

        -Lap phi + Lam^2 N(psi) exp(phi / tau(psi)) = f*  in Omega,   phi = phi* on dOmega,

    defined in REFERENCE coordinates W = (t, x, y):
        phi*  = F(psi) + eps (1 - psi) Re[w^m e^{-i n zeta}],
        F     = amp ( -(1 - psi) + kink H(psi) ),
        H     = (1 + tanh((psi - psi0) / delta)) / 2           (profile 'tanh', default)
              = q^beta,  q = max(0, (psi - psi0) / (1 - psi0))   (profile 'kink'),
        psi   = (x^2 + y^2) / Rc^2,  w = (x + i y) / Rc,  zeta = 2 pi nfp t / L.
    'tanh': an analytic pedestal of height amp*kink and width ~delta (in psi)
    around psi0; its nearest complex singularities sit at psi - psi0 = +-i pi
    delta / 2, so delta sets how hard it is (spectral convergence once resolved).
    'kink': F is C^{ceil(beta)-1} across the flux surface psi = psi0 (not aligned
    with any leaf boundary); f* contains q^(beta-2), so beta > 2 keeps f*
    continuous (beta = 2: f* jumps).  The Fourier term is smooth, L-
    periodic in t, regular at the axis (w^m is a polynomial) and vanishes on
    the wall; it makes phi* genuinely 3D.  kink = 0 or eps alone: smooth control.

    -Lap phi* is evaluated with the SAME PDO coefficients the solver uses
    (pdo_geo, hpsmultidomain convention), from closed-form reference
    derivatives -- exact for any geometry, no inverse map.  It is cached per
    point set (LRU, shared by with_Lam copies: it does not depend on Lam)."""

    def __init__(self, geom, pdo_geo, Lam=20.0, N="vanishing", tau_slope=0.4, amp=3.0, kink=0.5,
                 beta=2.5, psi0=0.7, eps=0.2, mode=(2, 1), cache=None, cache_size=64,
                 profile="tanh", delta=0.5):
        from collections import OrderedDict
        from pb.profiles import PBProfiles
        if profile not in ("tanh", "kink"):
            raise ValueError("profile must be 'tanh' or 'kink', got %r" % (profile,))
        if profile == "kink" and beta < 2:
            raise ValueError("beta must be >= 2 (beta < 2 makes f* unbounded at psi = psi0)")
        if profile == "tanh" and not delta > 0:
            raise ValueError("delta must be > 0")
        self.profile, self.delta = profile, float(delta)
        if not 0.0 <= psi0 < 1.0:
            raise ValueError("psi0 must be in [0, 1)")
        self.geom, self.pdo_geo = geom, pdo_geo
        self.prof = PBProfiles(geom, Lam=Lam, N=N, tau_slope=tau_slope)
        self.amp, self.kink, self.beta, self.psi0, self.eps = float(amp), float(kink), float(beta), float(psi0), float(eps)
        self.mode = tuple(int(v) for v in mode)
        self.Rc, self.L = float(geom.Rc), float(geom.L)
        self.nfp = int(getattr(geom, "nfp", 1))
        self._cache = OrderedDict() if cache is None else cache
        self.cache_size = int(cache_size)

    def with_Lam(self, Lam):
        return PBPedestal(self.geom, self.pdo_geo, Lam, self.prof.N_kind, self.prof.tau_slope, self.amp,
                          self.kink, self.beta, self.psi0, self.eps, self.mode, self._cache, self.cache_size,
                          self.profile, self.delta)

    # ---- profiles / nonlinearity (PBProfiles)
    @property
    def Lam(self):
        return self.prof.Lam

    def psi(self, W):
        return self.prof.psi(W)

    def N(self, psi):
        return self.prof.N(psi)

    def tau(self, psi):
        return self.prof.tau(psi)

    def g(self, W, s, psi=None):
        return self.prof.g(W, s, psi)

    def dg(self, W, s, psi=None):
        return self.prof.dg(W, s, psi)

    # ---- phi* and its reference derivatives
    def _F(self, psi):
        a, k, b, p0 = self.amp, self.kink, self.beta, self.psi0
        if self.profile == "tanh":
            dl = self.delta
            th = np.tanh((psi - p0) / dl)
            sech2 = 1.0 - th * th
            F = a * (-(1.0 - psi) + k * 0.5 * (1.0 + th))
            F1 = a * (1.0 + k * 0.5 * sech2 / dl)
            F2 = -a * k * th * sech2 / dl ** 2
            return F, F1, F2
        d = 1.0 - p0
        q = np.clip((psi - p0) / d, 0.0, None)
        on = psi > p0
        F = a * (-(1.0 - psi) + k * q ** b)
        F1 = a * (1.0 + k * b * q ** (b - 1) / d)
        F2 = a * k * b * (b - 1) * np.where(on, q ** (b - 2) if b > 2 else 1.0, 0.0) / d ** 2
        return F, F1, F2

    def derivs(self, W, order=2):
        """phi* and its (t, x, y) derivatives: dict with keys '', 't', 'x', 'y'
        (order >= 1), 'tt', 'tx', 'ty', 'xx', 'xy', 'yy' (order 2)."""
        W = np.asarray(W, dtype=np.float64)
        t, x, y = W[:, 0], W[:, 1], W[:, 2]
        Rc = self.Rc
        psi = (x * x + y * y) / Rc ** 2
        F, F1, F2 = self._F(psi)
        m, n = self.mode
        kk = 2 * np.pi * self.nfp / self.L
        wc = (x + 1j * y) / Rc
        E = np.exp(-1j * n * kk * t)
        Wm = wc ** m
        S = {"": (Wm * E).real}
        D = {"": F + self.eps * (1.0 - psi) * S[""]}
        if order == 0:
            return D
        zero = 0.0 * t
        px, py = 2 * x / Rc ** 2, 2 * y / Rc ** 2
        Wx = m * wc ** (m - 1) / Rc if m >= 1 else 0.0 * wc
        Wy = 1j * Wx
        Et = -1j * n * kk * E
        S.update(t=(Wm * Et).real, x=(Wx * E).real, y=(Wy * E).real)
        e = self.eps
        D["t"] = e * (1.0 - psi) * S["t"]
        D["x"] = F1 * px + e * (-px * S[""] + (1.0 - psi) * S["x"])
        D["y"] = F1 * py + e * (-py * S[""] + (1.0 - psi) * S["y"])
        if order == 1:
            return D
        Wxx = m * (m - 1) * wc ** (m - 2) / Rc ** 2 if m >= 2 else 0.0 * wc
        Wxy, Wyy = 1j * Wxx, -Wxx
        Ett = -(n * kk) ** 2 * E
        S.update(tt=(Wm * Ett).real, tx=(Wx * Et).real, ty=(Wy * Et).real,
                 xx=(Wxx * E).real, xy=(Wxy * E).real, yy=(Wyy * E).real)
        pxx = pyy = 2 / Rc ** 2
        D["tt"] = e * (1.0 - psi) * S["tt"]
        D["tx"] = e * (-px * S["t"] + (1.0 - psi) * S["tx"])
        D["ty"] = e * (-py * S["t"] + (1.0 - psi) * S["ty"])
        D["xx"] = F2 * px * px + F1 * pxx + e * (-pxx * S[""] - 2 * px * S["x"] + (1.0 - psi) * S["xx"])
        D["yy"] = F2 * py * py + F1 * pyy + e * (-pyy * S[""] - 2 * py * S["y"] + (1.0 - psi) * S["yy"])
        D["xy"] = F2 * px * py + zero + e * (-px * S["y"] - py * S["x"] + (1.0 - psi) * S["xy"])
        return D

    def phistar(self, W):
        return self.derivs(W, order=0)[""]

    def phi_b(self, W):
        return self.phistar(W)

    def _coef(self, W):
        Wt = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float64))
        out = {}
        for name in ("c11", "c22", "c33", "c12", "c13", "c23", "c1", "c2", "c3"):
            f = getattr(self.pdo_geo, name, None)
            v = 0.0 if f is None else f(Wt)
            v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float64)
            out[name] = v * np.ones(len(W))
        return out

    def neg_lap_phistar(self, W):
        """-Lap phi* at reference points W (hpsmultidomain PDO convention:
        -Lap u = -(c11 u_tt + c22 u_xx + c33 u_yy + c12 u_tx + c13 u_ty + c23 u_xy)
                 + c1 u_t + c2 u_x + c3 u_y)."""
        import hashlib
        W = np.ascontiguousarray(W, dtype=np.float64)
        key = (W.shape, hashlib.blake2b(W.tobytes(), digest_size=16).hexdigest())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        D, c = self.derivs(W, order=2), self._coef(W)
        v = -(c["c11"] * D["tt"] + c["c22"] * D["xx"] + c["c33"] * D["yy"]
              + c["c12"] * D["tx"] + c["c13"] * D["ty"] + c["c23"] * D["xy"]) \
            + c["c1"] * D["t"] + c["c2"] * D["x"] + c["c3"] * D["y"]
        self._cache[key] = v
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return v

    def fstar(self, W):
        return self.neg_lap_phistar(W) + self.g(W, self.phistar(W))

    def describe(self):
        if self.profile == "tanh":
            H = "(1 + tanh((psi - %g)/%g))/2" % (self.psi0, self.delta)
        else:
            H = "q^%g, q = (psi - %g)_+/%g" % (self.beta, self.psi0, 1 - self.psi0)
        return ("manufactured phi* = %g [-(1-psi) + %g H] + %g (1-psi) Re[w^%d e^{-i %d zeta}], H = %s;  %s"
                % (self.amp, self.kink, self.eps, self.mode[0], self.mode[1], H, self.prof.describe()))
