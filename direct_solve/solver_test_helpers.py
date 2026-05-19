import numpy as np
import time

from omsdirectsolve import *

def tridiagonal_block(n, a, b):
    """
    a = diagonal value
    b = off-diagonal value
    """
    return a * np.eye(n) + b * (np.eye(n, k=1) + np.eye(n, k=-1))

def test_prep(slabs, m, main_list, upper_off_list, lower_off_list, cyclic=False, rhs=None, x_true=None, A=None):

    S_rk_list        = [[upper_off_list[0]]] + [[lower_off_list[_], upper_off_list[_+1]] for _ in range(slabs-2)] + [[lower_off_list[-1]]]
    # Adaptation for non-cyclic:
    #if not cyclic:
    #    S_rk_list[0]  = [offdiag_block]
    #    S_rk_list[-1] = [offdiag_block.T]

    glob_target_dofs = [range(l*m, (l+1)*m) for l in range(slabs)]

    # Make the full block diag matrix:
    if A is None:
        A = np.zeros((slabs*m, slabs*m))
        for i in range(slabs):
            A[i*m:(i+1)*m, i*m:(i+1)*m] = main_list[i]
            if i != 0 or cyclic:
                prev = ((i-1) * m) % (slabs * m)
                A[i*m:(i+1)*m, prev:prev+m] = S_rk_list[i][0]
            if i != slabs-1 or cyclic:
                next = ((i+1) * m) % (slabs * m)
                A[i*m:(i+1)*m, next:next+m] = S_rk_list[i][-1]
        
    if rhs is None:
        rhs = np.random.rand(slabs*m)

    if x_true is None and A.shape[1] < 128**2:
        x_true = np.linalg.solve(A, rhs)
    elif x_true is None:
        x_true = np.zeros(A.shape[1])

    return S_rk_list, main_list, glob_target_dofs, A, rhs, x_true

#
# Testing the tridiagonal formulation
#
def test_tridiagonal_solve(slabs, m, main_list, upper_off_list, lower_off_list, cyclic=False, rhs=None, x_true=None, A=None):
    
    S_rk_list, main_list, glob_target_dofs, A, rhs, x_true = test_prep(slabs, m, main_list, upper_off_list, lower_off_list, cyclic, rhs, x_true)

    solver = BlockTridiagonalSolver(main_list[0].shape[0], cyclic=cyclic)
    tic = time.perf_counter()
    solver.factorize(S_rk_list, main_list)
    factor_toc = time.perf_counter() - tic
    tic = time.perf_counter()
    x = solver.solve(rhs, glob_target_dofs=glob_target_dofs)
    run_toc = time.perf_counter() - tic

    # Interested in the conditon number of one of the blocks:
    block_cond = -1
    if m < 2048:
        last_block = reconstruct_from_lu_factor(solver.B[-1])
        block_cond = np.linalg.cond(last_block)

    true_residual = np.linalg.norm(A @ x_true - rhs) / np.linalg.norm(rhs)
    residual = np.linalg.norm(A @ x - rhs) / np.linalg.norm(rhs)
    forward_error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    return true_residual, residual, forward_error, factor_toc, run_toc, block_cond

#
# Testing the RB formulation
#
def test_redblack_solve(slabs, m, main_list, upper_off_list, lower_off_list, cyclic=False, rhs=None, x_true=None, A=None):
    
    S_rk_list, main_list, _, A, rhs, x_true = test_prep(slabs, m, main_list, upper_off_list, lower_off_list, cyclic, rhs, x_true)

    solver = RedBlackSolver(main_list[0].shape[0], cyclic=cyclic)
    tic = time.perf_counter()
    solver.factorize(S_rk_list, main_list)
    factor_toc = time.perf_counter() - tic
    tic = time.perf_counter()
    x = solver.solve(rhs)
    run_toc = time.perf_counter() - tic

    # Interested in the conditon number of one of the blocks:
    block_cond = -1
    if m < 2048:
        last_block = solver.RB[-1][1][0]
        block_cond = np.linalg.cond(last_block)

    true_residual = np.linalg.norm(A @ x_true - rhs) / np.linalg.norm(rhs)
    residual = np.linalg.norm(A @ x - rhs) / np.linalg.norm(rhs)
    forward_error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    return true_residual, residual, forward_error, factor_toc, run_toc, block_cond

#
# For generating blocks with specified condition number:
#
def random_orthogonal(b):
    Q, R = np.linalg.qr(np.random.randn(b, b))
    # QR is not quite uniform without this sign correction
    Q *= np.sign(np.diag(R))
    return Q

def block_with_singular_values(sigmas):
    b = len(sigmas)
    U = random_orthogonal(b)
    V = random_orthogonal(b)
    return U @ np.diag(sigmas) @ V.T

def make_tridiagonal_block(b, kappa_target):
    # Diagonal blocks with prescribed condition number
    p = np.log10(kappa_target)
    sigmas = np.logspace(0, -p, b)
    
    return block_with_singular_values(sigmas)