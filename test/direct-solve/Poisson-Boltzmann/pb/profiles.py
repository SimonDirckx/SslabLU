"""
Physical (non-manufactured) normalized PB, model (A):

    -Lap phi + Lam^2 N(psi) exp(phi / tau(psi)) = 0  in Omega,   phi = phi_b on dOmega,
    psi = (x^2 + y^2) / Rc^2 (reference coordinates, psi = 1 on the wall),
    N(psi)   = 1 - psi^2            ('vanishing': zero density at the edge)
             = 1 - psi / 2          ('linear': the manufactured-test profile),
    tau(psi) = 1 - tau_slope psi,
    phi_b = 0, no source.

Same interface as PBManufactured where the solvers use it (psi, g, dg, fstar,
Lam), plus phi_b(W) and with_Lam(Lam).  No exact solution.
"""
import numpy as np


class PBProfiles:
    def __init__(self, geom, Lam=20.0, N="vanishing", tau_slope=0.4, phi_b=0.0):
        if N not in ("vanishing", "linear"):
            raise ValueError("N must be 'vanishing' or 'linear', got %r" % (N,))
        if not 0.0 <= tau_slope < 1.0:
            raise ValueError("tau_slope must be in [0, 1) so that tau > 0 on psi <= 1")
        self.geom, self.Lam = geom, float(Lam)
        self.N_kind, self.tau_slope, self.phi_b_val = N, float(tau_slope), float(phi_b)

    def with_Lam(self, Lam):
        return PBProfiles(self.geom, Lam, self.N_kind, self.tau_slope, self.phi_b_val)

    # ---- profiles
    def psi(self, W):
        W = np.asarray(W, dtype=np.float64)
        return (W[:, 1] ** 2 + W[:, 2] ** 2) / self.geom.Rc ** 2

    def N(self, psi):
        if self.N_kind == "vanishing":
            return np.clip(1.0 - psi ** 2, 0.0, None)     # clip: rounding at psi = 1
        return 1.0 - 0.5 * psi

    def tau(self, psi):
        return 1.0 - self.tau_slope * psi

    # ---- nonlinearity and data
    def g(self, W, s, psi=None):
        psi = self.psi(W) if psi is None else psi
        return self.Lam ** 2 * self.N(psi) * np.exp(s / self.tau(psi))

    def dg(self, W, s, psi=None):
        psi = self.psi(W) if psi is None else psi
        return self.g(W, s, psi) / self.tau(psi)

    def fstar(self, W):
        return np.zeros(len(W))

    def phi_b(self, W):
        return np.full(len(W), self.phi_b_val)

    def describe(self):
        Ns = "1 - psi^2" if self.N_kind == "vanishing" else "1 - psi/2"
        return "N = %s, tau = 1 - %g psi, phi_b = %g, Lam = %g" % (
            Ns, self.tau_slope, self.phi_b_val, self.Lam)
