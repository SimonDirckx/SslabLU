"""
Focused tests for v4: variable-coefficient Helmholtz with dominant identity
Laplacian and a spatially varying zero-order term.

PDE: Delta u + c(x) u = f(x)

Tests (2x2x2 unit cube):
  A. Constant Helmholtz with loading (MMS).
     coeffs = {c11=c22=c33=1, c=k^2}
     u_exact = sin(pi x)sin(pi y)sin(pi z), boundary = 0
     forcing = (-3 pi^2 + k^2) u_exact
  B. Same as A but ct_pde=False with the constants supplied as callables
     (regression: must match A bit-for-bit).
  C. Variable-c Helmholtz: c(x) = k^2 * (1 + a sin(2 pi x) sin(2 pi y) sin(2 pi z))
     Manufactured solution u_exact = sin(pi x)sin(pi y)sin(pi z), boundary = 0.
     forcing = (Delta + c(x)) u_exact = (-3 pi^2 + c(x)) u_exact

  D. (Regression, no forcing.) Green's function Helmholtz, ct_pde=True.
"""

import mumps
import numpy as np
import scipy.sparse as sp

from SOMS3D_csr import SOMS_solver_sparse

PI = np.pi



def u_mms(X):
    """u(x) = sin(pi x) sin(pi y) sin(pi z), zero on boundary of unit cube."""
    return np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1]) * np.sin(PI * X[:, 2])


def solve_with_mumps(Sii, rhs):
    """Solve Sii @ u = rhs using python-mumps."""
    ctx = mumps.Context()
    ctx.analyze(Sii)             # symbolic factorization (uses sparsity pattern only)
    ctx.factor(Sii)              # numeric factorization
    x = ctx.solve(rhs)           # solve, returns the solution
    return x

# Replace the spsolve line in your test with:



def solve(p, nb, coeffs, ct_pde, forcing, u_exact_fn):
    Sii, Sib, ftild, XYtot, Ii, Ib = SOMS_solver_sparse(
        p, p, p, nb, nb, nb,
        coeffs=coeffs, ct_pde=ct_pde, forcing=forcing,
    )
    print("============ S done ============")
    u_e = u_exact_fn(XYtot)
    u_b = u_e[Ib]
    rhs = -(Sib @ u_b) + ftild[Ii]
    u_i = solve_with_mumps(Sii, rhs)
    err = np.linalg.norm(u_i - u_e[Ii], ord=np.inf)
    ref = max(np.linalg.norm(u_e[Ii], ord=np.inf), 1e-12)
    return err, ref, XYtot.shape[0], u_i, u_e[Ii]


def test_A_constant_helmholtz_loaded():
    print("=" * 80)
    print("Test A: constant-coefficient Helmholtz with forcing, ct_pde=True")
    print("PDE: Delta u + k^2 u = f, manufactured u = sin(pi x)sin(pi y)sin(pi z)")
    print("=" * 80)
    for k in [0.5, 2.0, 4.0]:
        coeffs = {'c11': 1.0, 'c22': 1.0, 'c33': 1.0, 'c': k**2}
        amp = -3 * PI**2 + k**2
        forcing = (lambda A=amp:
                   lambda x, y, z: A * np.sin(PI*x)*np.sin(PI*y)*np.sin(PI*z))()
        print(f"\n  k = {k}, forcing amplitude = {amp:.3f}")
        print(f"  {'p':>3} {'ndofs':>6} {'rel err':>12}")
        for p in [4, 6, 8, 10]:
            err, ref, ndof, _, _ = solve(p, 4, coeffs, True, forcing, u_mms)
            print(f"  {p:>3} {ndof:>6} {err/ref:>12.3e}")


def test_B_consistency_ct_vs_vc():
    print("\n" + "=" * 80)
    print("Test B: ct_pde=True vs ct_pde=False with constants-as-callables")
    print("Should match to machine precision.")
    print("=" * 80)
    k = 2.0
    coeffs_const = {'c11': 1.0, 'c22': 1.0, 'c33': 1.0, 'c': k**2}
    coeffs_cb = {
        'c11': lambda x,y,z: np.ones_like(x),
        'c22': lambda x,y,z: np.ones_like(x),
        'c33': lambda x,y,z: np.ones_like(x),
        'c':   lambda x,y,z: np.full_like(x, k**2),
    }
    amp = -3 * PI**2 + k**2
    forcing = lambda x, y, z: amp * np.sin(PI*x)*np.sin(PI*y)*np.sin(PI*z)

    print(f"\n  k = {k}")
    print(f"  {'p':>3} {'ndofs':>6} {'CT err':>12} {'VC err':>12} {'|CT - VC|':>12}")
    for p in [4, 6, 8]:
        err_ct, _, ndof, u_ct, _ = solve(p, 4, coeffs_const, True, forcing, u_mms)
        err_vc, _, _,    u_vc, _ = solve(p, 4, coeffs_cb,    False, forcing, u_mms)
        diff = np.max(np.abs(u_ct - u_vc))
        print(f"  {p:>3} {ndof:>6} {err_ct:>12.3e} {err_vc:>12.3e} {diff:>12.3e}")


def test_C_variable_c_helmholtz():
    print("\n" + "=" * 80)
    print("Test C: variable-coefficient Helmholtz")
    print("PDE: Delta u + c(x) u = f")
    print("c(x) = k^2 * (1 + a sin(2 pi x) sin(2 pi y) sin(2 pi z))")
    print("u_exact = sin(pi x) sin(pi y) sin(pi z), boundary = 0")
    print("=" * 80)

    k_base = 2.0
    a_mod = 0.3   # 30% modulation
    def c_fn(x, y, z):
        return k_base**2 * (1 + a_mod * np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*z))

    coeffs = {
        'c11': 1.0, 'c22': 1.0, 'c33': 1.0,
        'c': c_fn,
    }

    # forcing = (-3 pi^2 + c(x)) * u_exact
    def forcing(x, y, z):
        return (-3*PI**2 + c_fn(x, y, z)) * np.sin(PI*x)*np.sin(PI*y)*np.sin(PI*z)

    print(f"\n  k_base = {k_base}, modulation amplitude = {a_mod}")
    print(f"  {'p':>3} {'ndofs':>6} {'rel err':>12}")
    for p in [4, 6, 8, 10]:
        err, ref, ndof, _, _ = solve(p, 8, coeffs, False, forcing, u_mms)
        print(f"  {p:>3} {ndof:>6} {err/ref:>12.3e}")


def test_D_regression_homogeneous_green():
    print("\n" + "=" * 80)
    print("Test D: regression - homogeneous Helmholtz Green's function (forcing=None)")
    print("=" * 80)
    x_src = np.array([2.0, 0.5, 0.5])
    k = 1.0
    coeffs = {'c11': 1.0, 'c22': 1.0, 'c33': 1.0, 'c': k**2}

    def green_real(X):
        r = np.linalg.norm(X - x_src, axis=1)
        return np.cos(k * r) / (4 * PI * r)

    print(f"\n  k = {k}, source at {x_src}")
    print(f"  {'p':>3} {'ndofs':>6} {'rel err':>12}")
    for p in [4, 6, 8, 10]:
        Sii, Sib, ftild, XYtot, Ii, Ib = SOMS_solver_sparse(
            p, p, p, 4, 4, 4, coeffs=coeffs, ct_pde=True, forcing=None,
        )
        
        assert np.max(np.abs(ftild)) == 0.0, "ftild must be zero with forcing=None"
        u_e = green_real(XYtot)
        u_b = u_e[Ib]
        rhs = -(Sib @ u_b) + ftild[Ii]
        u_i = spla.spsolve(Sii.tocsr(), rhs)
        err = np.linalg.norm(u_i - u_e[Ii], ord=np.inf)
        ref = np.linalg.norm(u_e[Ii], ord=np.inf)
        print(f"  {p:>3} {XYtot.shape[0]:>6} {err/ref:>12.3e}")


if __name__ == "__main__":
    test_A_constant_helmholtz_loaded()
    test_B_consistency_ct_vs_vc()
    test_C_variable_c_helmholtz()
    test_D_regression_homogeneous_green()