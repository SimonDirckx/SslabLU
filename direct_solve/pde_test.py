import numpy as np
from scipy.sparse.linalg   import LinearOperator
from scipy.linalg   import lu_factor, lu_solve, block_diag
import pickle

from solver_test_helpers import *

slabs = 8
m      = 8
#residual_tridiagonal, error_tridiagonal = test_tridiagonal_solve(slabs, m, 3*np.eye(m), -np.eye(m), cyclic=True)
#residual_redblack, error_redblack       = test_redblack_solve(slabs, m, 3*np.eye(m), -np.eye(m), cyclic=True)

h = 0.1
main_block = (1/h**2) * tridiagonal_block(m, 4, -1)
off_block = -(1/h**2) * np.eye(m)
"""
true_residual_tridiagonal, residual_tridiagonal, error_tridiagonal = test_tridiagonal_solve(slabs, m, main_block, off_block, cyclic=True)
cond, true_residual_redblack, residual_redblack, error_redblack    = test_redblack_solve(slabs, m, main_block, off_block, cyclic=True)

print("Residuals on pde-like problem (tridiagonal, then red-black)")
print(residual_tridiagonal, residual_redblack)

print("Forward errors on pde-like problem (tridiagonal, then red-black)")
print(error_tridiagonal, error_redblack)
"""


###
### Let's test on a nice fake "PDE"
###

# Manufactured solution and body load:
def u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f(x, y):
    return 2 * np.pi**2 * u(x, y)

def disc(index, h, Nx):
    x_i = h * ((index // Nx) + 1)
    y_j = h * ((index % Nx) + 1)
    return x_i, y_j

def u_disc(index, h, Nx):
    x_i, y_j = disc(index, h, Nx)
    return u(x_i, y_j)

def f_disc(index, h, Nx):
    x_i, y_j = disc(index, h, Nx)
    return f(x_i, y_j)

Nx_list   = [16] #, 32, 64, 128]
slab_list = [2, 4, 8, 16]

cyclic = False

h_list = np.zeros(len(Nx_list))

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
    Ny = Nx
    h  = 1 / (Nx + 1)
    h_list[Nx_index] = h
    for slab_index in range(len(slab_list)):
        slabs = slab_list[slab_index]
        m     = (Nx * Ny) // slabs

        print("\nFOR Nx = ", Nx, ", h = ", h, ", slabs = ", slabs)

        # Next we construct the matrix A and proper slab blocks:
        main_block = (1/h**2) * tridiagonal_block(Ny, 4, -1)
        off_block = -(1/h**2) * np.eye(Ny)

        I = np.eye(Nx)
        T = np.diag(np.ones(Nx-1), 1) + np.diag(np.ones(Nx-1), -1)
        A = np.kron(I, main_block) + np.kron(T, off_block)

        main_list = []
        upper_off_list = []
        lower_off_list = []

        for i in range(slabs):
            main_list.append(A[i*m:(i+1)*m, i*m:(i+1)*m])
            if i > 0:
                lower_off_list.append(A[i*m:(i+1)*m, (i-1)*m:i*m])
            if i < slabs - 1:
                upper_off_list.append(A[i*m:(i+1)*m, (i+1)*m:(i+2)*m])

        #print(A)
        #print(main_list)
        #print(upper_off_list)
        #print(lower_off_list)

        cond = -1
        if slabs*m < 128**2:
            cond = np.linalg.cond(A)

        xbds = [0, 1]
        ybds = [0, 1]

        body_load = np.array([f_disc(_, h, Nx) for _ in range(Nx*Ny)])
        true_sol  = np.array([u_disc(_, h, Nx) for _ in range(Nx*Ny)])

        true_residual_tridiagonal, residual_tridiagonal, error_tridiagonal, factor_time_tridiagonal, run_time_tridiagonal, block_cond_tridiagonal = test_tridiagonal_solve(slabs, m, main_list, upper_off_list, lower_off_list, rhs=body_load, x_true=true_sol, A=A)
        true_residual_redblack, residual_redblack, error_redblack, factor_time_redblack, run_time_redblack, block_cond_redblack         = test_redblack_solve(slabs, m, main_list, upper_off_list, lower_off_list, rhs=body_load, x_true=true_sol, A=A)

        print("Condition Number: ", cond)

        print("Residuals on pde-like problem for our 'true' solution (tridiagonal, then red-black)")
        print(true_residual_tridiagonal, true_residual_redblack)

        print("Residuals on pde-like problem (tridiagonal, then red-black)")
        print(residual_tridiagonal, residual_redblack)

        print("Forward errors on pde-like problem (tridiagonal, then red-black)")
        print(error_tridiagonal, error_redblack)

        print("Condition numbers of final block (tridiagonal, then red-black)")
        print(block_cond_tridiagonal, block_cond_redblack)

        print("Factorization times (tridiagonal, red-black):")
        print(factor_time_tridiagonal, factor_time_redblack)

        print("Runtimes (tridiagonal, red-black):")
        print(run_time_tridiagonal, run_time_redblack)

        true_residual_tridiagonals[Nx_index, slab_index] = true_residual_tridiagonal
        residual_tridiagonals[Nx_index, slab_index] = residual_tridiagonal
        error_tridiagonals[Nx_index, slab_index] = error_tridiagonal
        factor_time_tridiagonals[Nx_index, slab_index] = factor_time_tridiagonal
        run_time_tridiagonals[Nx_index, slab_index] = run_time_tridiagonal
        block_cond_tridiagonals[Nx_index, slab_index] = run_time_tridiagonal

        conds[Nx_index, slab_index] = cond
        true_residual_redblacks[Nx_index, slab_index] = true_residual_redblack
        residual_redblacks[Nx_index, slab_index] = residual_redblack
        error_redblacks[Nx_index, slab_index] = error_redblack
        factor_time_redblacks[Nx_index, slab_index] = factor_time_redblack
        run_time_redblacks[Nx_index, slab_index] = run_time_redblack
        block_cond_redblacks[Nx_index, slab_index] = run_time_redblack

data = {
    "Nx_list": Nx_list,
    "h_list": h,
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

#with open("neat_fdm_results.pkl", "wb") as f:
#    pickle.dump(data, f)


"""
###
### Now let's test on a not-so-nice fake "PDE"
###

# Manufactured solution and body load:
#def u(x, y):
#    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)

def disc(index, h, Nx):
    x_i = h * ((index // Nx) + 1)
    y_j = h * ((index % Nx) + 1)
    return x_i, y_j

#def u_disc(index, h, Nx):
#    x_i, y_j = disc(index, h, Nx)
#    return u(x_i, y_j)

def f_disc(index, h, Nx):
    x_i, y_j = disc(index, h, Nx)
    return f(x_i, y_j)

Nx_list   = [16, 32, 64, 128]
slab_list = [2, 4, 8, 16]

cyclic = False

h_list = np.zeros(len(Nx_list))

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
    Ny = Nx
    h  = 1 / (Nx + 1)
    h_list[Nx_index] = h
    for slab_index in range(len(slab_list)):
        slabs = slab_list[slab_index]
        m     = (Nx * Ny) // slabs

        print("\nFOR Nx = ", Nx, ", h = ", h, ", slabs = ", slabs)

        # Next we construct the matrix A and proper slab blocks:
        main_block = (1/h**2) * tridiagonal_block(Ny, 4, -1)
        off_block = -(1/h**2) * np.eye(Ny)

        I = np.eye(Nx)
        T = np.diag(np.ones(Nx-1), 1) + np.diag(np.ones(Nx-1), -1)
        A = np.kron(I, main_block) + np.kron(T, off_block)

        main_list = []
        upper_off_list = []
        lower_off_list = []

        for i in range(slabs):
            main_list.append(A[i*m:(i+1)*m, i*m:(i+1)*m])
            if i > 0:
                lower_off_list.append(A[i*m:(i+1)*m, (i-1)*m:i*m])
            if i < slabs - 1:
                upper_off_list.append(A[i*m:(i+1)*m, (i+1)*m:(i+2)*m])

        cond = -1
        if slabs*m < 128**2:
            cond = np.linalg.cond(A)

        xbds = [0, 1]
        ybds = [0, 1]

        body_load = np.array([f_disc(_, h, Nx) for _ in range(Nx*Ny)])

        true_residual_tridiagonal, residual_tridiagonal, error_tridiagonal, factor_time_tridiagonal, run_time_tridiagonal, block_cond_tridiagonal = test_tridiagonal_solve(slabs, m, main_list, upper_off_list, lower_off_list, rhs=body_load, A=A)
        true_residual_redblack, residual_redblack, error_redblack, factor_time_redblack, run_time_redblack, block_cond_redblack         = test_redblack_solve(slabs, m, main_list, upper_off_list, lower_off_list, rhs=body_load, A=A)

        print("Condition Number: ", cond)

        print("Residuals on pde-like problem for our 'true' solution (tridiagonal, then red-black)")
        print(true_residual_tridiagonal, true_residual_redblack)

        print("Residuals on pde-like problem (tridiagonal, then red-black)")
        print(residual_tridiagonal, residual_redblack)

        print("Forward errors on pde-like problem (tridiagonal, then red-black)")
        print(error_tridiagonal, error_redblack)

        print("Condition numbers of final block (tridiagonal, then red-black)")
        print(block_cond_tridiagonal, block_cond_redblack)

        print("Factorization times (tridiagonal, red-black):")
        print(factor_time_tridiagonal, factor_time_redblack)

        print("Runtimes (tridiagonal, red-black):")
        print(run_time_tridiagonal, run_time_redblack)

        true_residual_tridiagonals[Nx_index, slab_index] = true_residual_tridiagonal
        residual_tridiagonals[Nx_index, slab_index] = residual_tridiagonal
        error_tridiagonals[Nx_index, slab_index] = error_tridiagonal
        factor_time_tridiagonals[Nx_index, slab_index] = factor_time_tridiagonal
        run_time_tridiagonals[Nx_index, slab_index] = run_time_tridiagonal
        block_cond_tridiagonals[Nx_index, slab_index] = run_time_tridiagonal

        conds[Nx_index, slab_index] = cond
        true_residual_redblacks[Nx_index, slab_index] = true_residual_redblack
        residual_redblacks[Nx_index, slab_index] = residual_redblack
        error_redblacks[Nx_index, slab_index] = error_redblack
        factor_time_redblacks[Nx_index, slab_index] = factor_time_redblack
        run_time_redblacks[Nx_index, slab_index] = run_time_redblack
        block_cond_redblacks[Nx_index, slab_index] = run_time_redblack

data = {
    "Nx_list": Nx_list,
    "h_list": h,
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

with open("messy_fdm_results.pkl", "wb") as f:
    pickle.dump(data, f)
"""


