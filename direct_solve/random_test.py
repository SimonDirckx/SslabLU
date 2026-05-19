import numpy as np

from scipy.sparse.linalg   import LinearOperator
from scipy.linalg   import lu_factor, lu_solve, block_diag

from solver_test_helpers import *

"""
    Here we will test and evaluate our tridiagonal and red-black solvers across a range of matrix structures.
"""

def test_blocks(Nx_list, slab_list, main_block_func, off_block_func, filename, cyclic=False):

    true_residual_tridiagonals = np.zeros((len(Nx_list), len(slab_list)))
    residual_tridiagonals = np.zeros((len(Nx_list), len(slab_list)))
    error_tridiagonals = np.zeros((len(Nx_list), len(slab_list)))
    factor_time_tridiagonals = np.zeros((len(Nx_list), len(slab_list)))
    run_time_tridiagonals = np.zeros((len(Nx_list), len(slab_list)))
    block_cond_tridiagonals = np.zeros((len(Nx_list), len(slab_list)))

    conds = np.zeros((len(Nx_list), len(slab_list)))
    true_residual_redblacks = np.zeros((len(Nx_list), len(slab_list)))
    residual_redblacks = np.zeros((len(Nx_list), len(slab_list)))
    error_redblacks = np.zeros((len(Nx_list), len(slab_list)))
    factor_time_redblacks = np.zeros((len(Nx_list), len(slab_list)))
    run_time_redblacks = np.zeros((len(Nx_list), len(slab_list)))
    block_cond_redblacks = np.zeros((len(Nx_list), len(slab_list)))

    for Nx_index in range(len(Nx_list)):
        Nx = Nx_list[Nx_index]
        for slab_index in range(len(slab_list)):
            slabs = slab_list[slab_index]
            m     = Nx // slabs

            print("\nFOR Nx = ", Nx, ", m = ", m, ", slabs = ", slabs)

            # Next we construct the matrix A and proper slab blocks:
            main_list      = [main_block_func(m) for _ in range(slabs)]
            upper_off_list = [off_block_func(m) for _ in range(slabs - 1)]
            lower_off_list = [off_block_func(m) for _ in range(slabs - 1)]

            A = block_diag(*main_list)
            A[:-m,m:] += block_diag(*upper_off_list)
            A[m:,:-m] += block_diag(*lower_off_list)

            #print(main_list)
            #print(A.shape)

            if cyclic:
                print("Need to figure this out!")

            #print(A)
            #print(main_list)
            #print(upper_off_list)
            #print(lower_off_list)

            cond = -1
            if Nx < 1025:
                cond = np.linalg.cond(A)

            true_sol  = np.random.rand(Nx)

            body_load = A @ true_sol

            true_residual_tridiagonal, residual_tridiagonal, error_tridiagonal, factor_time_tridiagonal, run_time_tridiagonal, block_cond_tridiagonal = test_tridiagonal_solve(slabs, m, main_list, upper_off_list, lower_off_list, rhs=body_load, x_true=true_sol, A=A)
            true_residual_redblack, residual_redblack, error_redblack, factor_time_redblack, run_time_redblack, block_cond_redblack         = test_redblack_solve(slabs, m, main_list, upper_off_list, lower_off_list, rhs=body_load, x_true=true_sol, A=A)

            print("Condition Number of A itself: ", cond)

            print("All errors are tridiagonal, then cyclic reduction.")

            print("Backward residuals for our 'true' solution, ||Au_true - b|| / ||b||:")
            print(true_residual_tridiagonal, true_residual_redblack)

            print("Backwards residuals, ||Au - b|| / ||b||:")
            print(residual_tridiagonal, residual_redblack)

            print("Forward errors, aka ||u - u_true|| / ||u_true||:")
            print(error_tridiagonal, error_redblack)

            print("Condition numbers of final block:")
            print(block_cond_tridiagonal, block_cond_redblack)

            print("Factorization times:")
            print(factor_time_tridiagonal, factor_time_redblack)

            print("Runtimes:")
            print(run_time_tridiagonal, run_time_redblack)

            
            true_residual_tridiagonals[Nx_index, slab_index] = true_residual_tridiagonal
            residual_tridiagonals[Nx_index, slab_index] = residual_tridiagonal
            error_tridiagonals[Nx_index, slab_index] = error_tridiagonal
            factor_time_tridiagonals[Nx_index, slab_index] = factor_time_tridiagonal
            run_time_tridiagonals[Nx_index, slab_index] = run_time_tridiagonal
            block_cond_tridiagonals[Nx_index, slab_index] = block_cond_tridiagonal

            conds[Nx_index, slab_index] = cond
            true_residual_redblacks[Nx_index, slab_index] = true_residual_redblack
            residual_redblacks[Nx_index, slab_index] = residual_redblack
            error_redblacks[Nx_index, slab_index] = error_redblack
            factor_time_redblacks[Nx_index, slab_index] = factor_time_redblack
            run_time_redblacks[Nx_index, slab_index] = run_time_redblack
            block_cond_redblacks[Nx_index, slab_index] = block_cond_redblack
            

    
    data = {
        "Nx_list": Nx_list,
        "slab_list": slab_list,
        "true_residual_tridiagonals": true_residual_tridiagonals,
        "residual_tridiagonals": residual_tridiagonals,
        "error_tridiagonals": error_tridiagonals,
        "factor_time_tridiagonals": factor_time_tridiagonals,
        "run_time_tridiagonals": run_time_tridiagonals,
        "block_cond_tridiagonals": block_cond_tridiagonals,
        "conds": conds,
        "true_residual_redblacks": true_residual_redblacks,
        "residual_redblacks": residual_redblacks,
        "error_redblacks": error_redblacks,
        "factor_time_redblacks": factor_time_redblacks,
        "run_time_redblacks": run_time_redblacks,
        "block_cond_redblacks": block_cond_redblacks,
    }

    with open(filename, "wb") as f:
        pickle.dump(data, f)
    
Nx_list   = [64, 128, 256, 512, 1024]
slab_list = [2, 4, 8, 16, 32]

cyclic = False
"""
def main_block_func(m):
    return 4 * np.eye(m)

def off_block_func(m):
    return -np.eye(m)
"""

def main_block_func(m):
    kappa_target = 10
    return make_tridiagonal_block(m, kappa_target)

def off_block_func(m):
    return (0.1 / np.sqrt(m)) * np.random.rand(m, m)

filename = "output_kappa10_beta1.pkl"
"""
for m in [2, 4, 8, 16, 32]:
    print("m = ", m)
    block = main_block_func(m)
    print(block)
    print(np.linalg.cond(block))
    print(np.linalg.norm(block))
"""

test_blocks(Nx_list, slab_list, main_block_func, off_block_func, filename)