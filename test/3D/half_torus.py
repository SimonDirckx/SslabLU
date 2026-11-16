"""
Half twisted elliptic torus as the image of a straight cylinder.

Cylinder (computational domain, solved with the 5-patch HPS cylinder):
    disk of radius Rc = sqrt(a b)  x  z in [0, L],   L = pi R0
Map Phi: cylinder -> half torus,  (x, y, z) -> physical point
    zeta  = z / R0                     toroidal angle in [0, pi]   (arc length kept)
    X, Y  = (a/Rc) x, (b/Rc) y         disk -> ellipse, linear stretch (area kept exactly)
    alpha = twist * zeta               planar twist: rotate the ellipse in its own
                                       poloidal plane
    R     = R0 + X cos(alpha) - Y sin(alpha)
    Zp    =      X sin(alpha) + Y cos(alpha)
    phys  = (R cos zeta, R sin zeta, Zp)
The magnetic axis is the planar circle of radius R0; cross sections are planar;
the domain is not periodic (flat end caps at zeta = 0 and zeta = pi).

Pulled-back Helmholtz  -Lap u - k^2 u = 0  in cylinder coordinates (non-divergence
form, PDOPhys):
    a^{ij} = g^{ij}            inverse metric of Phi
    b^k    = g^{ij} Gamma^k_ij Gamma^k_ij = (J^{-1})_{km} d_i d_j Phi_m
    c      = -k^2
(Lap_x u = g^{ij} d_ij u - g^{ij} Gamma^k_ij d_k u, so -Lap_x u = -a:dd u + b.grad u.)
"""
import numpy as np
import sympy as sym
from stellerator import PDOPhys


class HalfTwistedTorus:
    def __init__(self, R0=3.0, a=1.2, b=0.75, twist=2.0):
        assert R0 > max(a, b), "torus must not self-intersect"
        self.R0, self.a, self.b, self.twist = R0, a, b, twist
        self.Rc = np.sqrt(a * b)            # equal cross-sectional area
        self.L = np.pi * R0                 # equal centreline length
        x, y, z = sym.symbols("x y z", real=True)
        zeta = z / R0
        al = twist * zeta
        X, Y = a / self.Rc * x, b / self.Rc * y
        R = R0 + X * sym.cos(al) - Y * sym.sin(al)
        Zp = X * sym.sin(al) + Y * sym.cos(al)
        Phi = sym.Matrix([R * sym.cos(zeta), R * sym.sin(zeta), Zp])
        V = (x, y, z)
        J = Phi.jacobian(V)
        H = [[[sym.diff(Phi[m], V[i], V[j]) for j in range(3)] for i in range(3)]
             for m in range(3)]
        lam = lambda e: sym.lambdify(V, e, "numpy")
        self._Phi = [lam(Phi[m]) for m in range(3)]
        self._J = [[lam(J[m, i]) for i in range(3)] for m in range(3)]
        self._H = [[[lam(H[m][i][j]) for j in range(3)] for i in range(3)] for m in range(3)]

    # ---------------------------------------------------------------- map
    def phi(self, Xc):
        Xc = np.asarray(Xc); one = np.ones(len(Xc))
        return np.stack([f(*Xc.T) * one for f in self._Phi], axis=1)

    def jac(self, Xc):
        Xc = np.asarray(Xc); one = np.ones(len(Xc))
        return np.stack([np.stack([self._J[m][i](*Xc.T) * one for i in range(3)], -1)
                         for m in range(3)], 1)          # J[n, m, i] = d Phi_m / d x_i

    def hess(self, Xc):
        Xc = np.asarray(Xc); one = np.ones(len(Xc))
        H = np.empty((len(Xc), 3, 3, 3))
        for m in range(3):
            for i in range(3):
                for j in range(3):
                    H[:, m, i, j] = self._H[m][i][j](*Xc.T) * one
        return H                                         # H[n, m, i, j] = d_i d_j Phi_m

    # ---------------------------------------------------------------- operator
    def coeffs(self, Xc):
        J = self.jac(Xc)
        Ji = np.linalg.inv(J)
        g_inv = Ji @ Ji.transpose(0, 2, 1)               # g^{ij}
        Gam = np.einsum("nkm,nmij->nkij", Ji, self.hess(Xc))
        bvec = np.einsum("nij,nkij->nk", g_inv, Gam)
        return g_inv, bvec

    def helmholtz(self, k):
        """PDOPhys for -Lap u - k^2 u = 0 pulled back to cylinder coordinates."""
        return PDOPhys(a=lambda X: self.coeffs(X)[0],
                       b=lambda X: self.coeffs(X)[1],
                       c=lambda X: -k ** 2 * np.ones(len(X)),
                       form="nondiv")


def point_source(k, x0):
    """Exact Helmholtz solution in physical space, singular only at x0."""
    x0 = np.asarray(x0, float)
    def u(Xp):
        r = np.linalg.norm(np.asarray(Xp) - x0, axis=1)
        return np.cos(k * r) / r
    return u
