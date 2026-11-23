"""
Radial clustering of the arch patches toward the wall.

The arch blending map (stellerator._arch_symbolic, circInterfaceTree) is
    u = (1 - x)/2 s + (1 + x)/2 cos(th),   v = (1 - x)/2 s eta + (1 + x)/2 sin(th),
linear in the radial blend parameter x in [-1, 1] (x = -1: core face, x = 1:
wall).  With stretch beta != 0 the arch reference coordinate xi is no longer x
itself but
    x = g(xi),   w = (xi + 1)/2,   g = 1 - 2 expm1(beta (1 - w)) / expm1(beta),
a smooth (analytic) monotone bijection of [-1, 1] with g(+-1) = +-1.

HPS leaves stay UNIFORM in xi (hpsmultidomain only builds uniform leaves, and
the prolongation / tree / estimator all work in leaf-index space), so the leaf
layout, leaf counts, face matching and the binary interface tree are unchanged;
only the map changes.  Physically, radial leaf k (k = 0 at the core face) has
width proportional to exp(-beta k / m): geometric grading with ratio
    sigma = exp(-beta / m)          (beta = m ln 2: each leaf half the previous).
beta > 0 clusters toward the wall, beta < 0 toward the core face, 0 = identity.

beta is a property of the GEOMETRY, not of a discretization: coarse and fine
grids of one run must use the same beta (nested leaves in xi), so the grading
ratio differs between them (sigma_c = sigma_f ** (m_f / m_c)).

The map is C-infinity, so the true solution is smooth in xi across leaf faces
and hpsmultidomain's reference normal-derivative matching inside a patch stays
correct (a piecewise-affine graded map would not be: d/dxi jumps across faces).
"""
import numpy as np


def blend_of_xi(xi, beta):
    """x = g(xi): reference coordinate -> radial blend parameter."""
    xi = np.asarray(xi, float)
    if beta == 0:
        return xi.copy()
    w = (xi + 1) / 2
    return 1 - 2 * np.expm1(beta * (1 - w)) / np.expm1(beta)


def xi_of_blend(x, beta):
    """xi = g^{-1}(x)."""
    x = np.asarray(x, float)
    if beta == 0:
        return x.copy()
    G = (x + 1) / 2
    w = 1 - np.log1p((1 - G) * np.expm1(beta)) / beta
    return 2 * w - 1


def blend_sym(xi, beta):
    """sympy expression of g(xi) (xi a sympy symbol); xi itself for beta = 0,
    so the unstretched symbolic map is exactly the original one."""
    if beta == 0:
        return xi
    import sympy as sym
    b = sym.Float(float(beta))
    w = (xi + 1) / 2
    return 1 - 2 * (sym.exp(b * (1 - w)) - 1) / (sym.exp(b) - 1)


def beta_for_ratio(sigma, m):
    """beta giving radial leaf-width ratio sigma (outer / inner neighbour) with m leaves."""
    return -m * np.log(sigma)


def radial_leaf_widths(beta, m):
    """Widths of the m radial leaves as fractions of the arch thickness
    (core face -> wall)."""
    xb = blend_of_xi(np.linspace(-1, 1, m + 1), beta)
    return np.diff(xb) / 2


def describe(beta, m):
    w = radial_leaf_widths(beta, m)
    ratio = np.exp(-beta / m) if m > 1 else 1.0
    return "m=%d: ratio %.3f, widths (core->wall) %s" % (m, ratio, " ".join("%.3f" % v for v in w))
