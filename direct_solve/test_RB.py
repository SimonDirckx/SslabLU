"""
Test suite for RedBlackSolver.

Constructs random diagonally-dominant block-tridiagonal systems (nSlabs must be
a power of 2) and checks that RedBlackSolver matches a dense numpy solve to at
least 8 digits of relative accuracy.
"""

import sys
import numpy as np
from omsdirectsolve import RedBlackSolver


def make_block_tridiagonal(nSlabs: int, m: int, seed: int = 42):
    """
    Build a random diagonally-dominant block-tridiagonal system.

    Returns
    -------
    S_rk_list : list of (A_i, C_i) pairs  – off-diagonal blocks (length nSlabs)
    T         : list of diagonal blocks   (length nSlabs)
    mat       : assembled (nSlabs*m) x (nSlabs*m) dense matrix
    rhs       : random RHS vector of length nSlabs*m
    """
    rng = np.random.default_rng(seed)
    N = nSlabs * m
    mat = np.zeros((N, N))
    T, S_rk_list = [], []

    for i in range(nSlabs):
        # Diagonal block – diagonally dominant for a well-conditioned system
        B = rng.standard_normal((m, m)) + nSlabs * m * np.eye(m)
        T.append(B)
        mat[i*m:(i+1)*m, i*m:(i+1)*m] = B

        A = 0.1 * rng.standard_normal((m, m)) if i > 0        else np.zeros((m, m))
        C = 0.1 * rng.standard_normal((m, m)) if i < nSlabs-1 else np.zeros((m, m))

        if i > 0:        mat[i*m:(i+1)*m, (i-1)*m:i*m]     = A
        if i < nSlabs-1: mat[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = C

        S_rk_list.append((A, C))

    rhs = rng.standard_normal(N)
    return S_rk_list, T, mat, rhs


def run_case(nSlabs: int, m: int, seed: int = 42) -> bool:
    S_rk_list, T, mat, rhs = make_block_tridiagonal(nSlabs, m, seed)

    x_ref = np.linalg.solve(mat, rhs)

    solver = RedBlackSolver(m, cyclic=False)
    print("len(S) = ",len(S_rk_list))
    print("len(T) = ",len(T))
    solver.factorize(S_rk_list, T)
    x_rb = solver.solve(rhs)

    err = np.linalg.norm(x_rb - x_ref) / np.linalg.norm(x_ref)
    ok = err < 1e-8
    status = "PASS" if ok else "FAIL"
    print(f"  nSlabs={nSlabs:3d}, m={m:3d}: rel_err={err:.2e}  {status}")
    return ok


def main():
    print("RedBlackSolver test suite")
    print("=" * 50)

    cases = [
        # (nSlabs, m)
        (8,  3),   # canonical 8-block, 3x3 diagonal blocks
        (2,  1),
        (2,  5),
        (4,  3),
        (8,  1),
        (8,  5),
        (8,  10),
        (16, 3),
        (32, 3),
    ]

    all_pass = True
    for nSlabs, m in cases:
        ok = run_case(nSlabs, m)
        all_pass = all_pass and ok

    print("=" * 50)
    print("ALL PASS" if all_pass else "SOME CASES FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())