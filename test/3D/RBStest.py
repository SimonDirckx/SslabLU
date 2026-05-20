"""
Compare RedBlackSolverHBS and RedBlackSolver level-by-level:
for every level l and every node i, compare
  - the diagonal block T[i]        (as a dense matrix)
  - its inverse T[i]^{-1}          (via solve)
between the dense LU solver and the HBS mock solver.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.sparse.linalg import LinearOperator

from scipy.linalg   import lu_factor, lu_solve, block_diag

from direct_solve.omsdirectsolve import RedBlackSolver
from direct_solve.omsdirectsolveHBS import RedBlackSolverHBS


import numpy as np
import jax.numpy as jnp
import torch
import scipy
from packaging.version import Version
import matplotlib.tri as tri

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import solver.spectralmultidomain.hps.pdo as pdo
# validation&testing
import time
from scipy.sparse.linalg import gmres
import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splinalg
import multislab.omsdirectsolve as omsdirect
#import multislab.omsdirectsolveHBS as omsdirectHBS
import direct_solve.omsdirectsolveHBS as omsdirectHBS
import direct_solve.omsdirectsolve as omsdirect
import geometry.geom_3D.cube as cube


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def local_solve(self, rhs,RB,m):
        """
        [  I   ] [ S_12 ] [  0   ] [  E?  ]
        [ S_21 ] [  I   ] [ S_23 ] [  0   ]
        [  0   ] [ S_32 ] [  I   ] [ S_34 ]
        [  F?  ] [  0   ] [ S_43 ] [  I   ]

        Solves using the Red-Black factorization. Assumes we have RB = (A_i, B_i, C_i) and v of size m * nSlabs
        """
        

        m = RB[0][0][0].shape[0]
        # Building the RHS:
        vPrimes = [rhs.copy()]
        
        # Now build the RHS:
        for l in range(len(RB) - 1):

            (SiM, _, T_inv, SiP) = RB[l]

            nSlabs   = len(SiM)
            nReduced = int(nSlabs / 2)
            vPrime   = np.zeros(m*nReduced)
            vPrev    = vPrimes[-1]
            cyclic = False
            for j in range(nReduced):
                i    = 2 * j
                prev = (i - 1) % nSlabs if (cyclic or i > 0)            else None
                next = (i + 1) % nSlabs if (cyclic or i < nSlabs - 1)   else None
                contrib = vPrev[i*m:(i+1)*m].copy()
                if prev is not None:
                    contrib -= SiM[i] @ lu_solve(T_inv[prev], vPrev[prev*m:(prev+1)*m])
                if next is not None:
                    contrib -= SiP[i] @ lu_solve(T_inv[next], vPrev[next*m:(next+1)*m])
                vPrime[j*m:(j+1)*m] = contrib

            vPrimes.append(vPrime)

        # Now get u: RB[-1] is the single-block coarsest level; vPrimes[-1] has size m (one block).
        vPrimes[-1] = lu_solve(RB[-1][2][0], vPrimes[-1])
        
        for l in range(len(RB) - 1, 0, -1):
            (SiM, _, Tinv, SiP)   = RB[l-1]
            nReduced = int(len(SiM) / 2)
            for j in range(nReduced):
                i = 2 * j
                # We fill in the odd segments of u
                vPrimes[l-1][i*m:(i+1)*m] = vPrimes[l][j*m:(j+1)*m]
                
                # Here we compute the even segments of u:
                next = (j + 1) % nReduced
                contrib = SiM[i+1] @ vPrimes[l][j*m:(j+1)*m]
                if self.cyclic or j + 1 < nReduced:
                    contrib += SiP[i+1] @ vPrimes[l][next*m:(next+1)*m]
                vPrimes[l-1][(i+1)*m:(i+2)*m] -= contrib
                vPrimes[l-1][(i+1)*m:(i+2)*m] = lu_solve(Tinv[i+1], vPrimes[l-1][(i+1)*m:(i+2)*m])

        return vPrimes[0]
def gen_tridiag_systems():
    jax_avail   = False
    torch_avail = not jax_avail
    hpsalt      = torch_avail
    kh = 10.
    if jax_avail:
        def c11(p):
            return jnp.ones_like(p[...,0])
        def c22(p):
            return jnp.ones_like(p[...,0])
        def c33(p):
            return jnp.ones_like(p[...,0])
        def c(p):
            return -kh*kh*jnp.ones_like(p[...,0])
        Helm=pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=c)


    elif torch_avail:
        def c11(p):
            return torch.ones_like(p[:,0])
        def c22(p):
            return torch.ones_like(p[:,1])
        def c33(p):
            return torch.ones_like(p[:,2])
        def c(p):
            return -kh*kh*torch.ones_like(p[:,0])
        Helm=pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)

    else:
        def c11(p):
            return np.ones_like(p[:,0])
        def c22(p):
            return np.ones_like(p[:,0])
        def c33(p):
            return np.ones_like(p[:,0])
        def c(p):
            return -kh*kh*np.ones_like(p[:,0])
        Helm=pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=c)
    def bc(p):
        source_loc = np.array([-.5,-.2,1])
        rr = np.linalg.norm(p-source_loc.T,axis=1)
        return np.real(np.exp(1j*kh*rr)/(4*np.pi*rr))
        #return np.sin(kh*(p[:,0]+p[:,1]+p[:,2])/np.sqrt(3))


    N = 9
    dSlabs,connectivity,H = cube.dSlabs(N)
    p = 6
    formulation = "hpsalt"
    p_disc = p + 2
    a = np.array([H/4,1/8,1/8])
    assembler = mA.rkHMatAssembler(p*p,150,ndim=3)
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
    OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    E = np.identity(nc)
    S_dense_list = []
    T = []
    print("len(S_rk_list) = ",len(S_rk_list),"//",len(dSlabs))
    for i in range(len(S_rk_list)):
        Sloc = []
        print(len(S_rk_list[i]))
        for j in range(len(S_rk_list[i])):
            Sloc+=[S_rk_list[i][j]@E]
        S_dense_list+=[Sloc]
        T+=[E]
    return T, S_dense_list, S_rk_list, rhstot, Stot, nc
    

def dense_to_linop(A):
    A = np.array(A)
    n = A.shape[0]
    lo = LinearOperator(
        shape=(n, n), dtype=A.dtype,
        matvec  = lambda v: A @ v,
        rmatvec = lambda v: A.T @ v,
        matmat  = lambda V: A @ V,
        rmatmat = lambda V: A.T @ V,
    )
    lo.solve = lambda v, mode='N': (
        np.linalg.solve(A, v) if mode == 'N' else np.linalg.solve(A.T, v)
    )
    lo.tree = lo.quad = None
    return lo


def make_system(nSlabs, m, seed=42):
    """
    Block-tridiagonal system with identity diagonal blocks.
    Off-diagonal blocks are random with Frobenius norm ~0.5.
    Both solvers receive the same off-diagonal blocks (no perturbation here);
    the only difference between dense and HBS comes from HBS approximation error.
    """
    rng = np.random.default_rng(seed)
    N   = nSlabs * m
    mat = np.zeros((N, N))
    S_dense, S_lo = [], []

    T_dense = [np.eye(m) for _ in range(nSlabs)]  # identity diagonal

    for i in range(nSlabs):
        mat[i*m:(i+1)*m, i*m:(i+1)*m] = np.eye(m)

        A_raw = rng.standard_normal((m, m)) if i > 0        else np.zeros((m, m))
        C_raw = rng.standard_normal((m, m)) if i < nSlabs-1 else np.zeros((m, m))
        A = 0.5 * A_raw / np.linalg.norm(A_raw) * m**0.5 if i > 0        else np.zeros((m, m))
        C = 0.5 * C_raw / np.linalg.norm(C_raw) * m**0.5 if i < nSlabs-1 else np.zeros((m, m))

        if i > 0:        mat[i*m:(i+1)*m, (i-1)*m:i*m]     = A
        if i < nSlabs-1: mat[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = C

        S_dense.append((A, C))
        S_lo.append(   (dense_to_linop(A), dense_to_linop(C)))

    return T_dense, S_dense, S_lo, mat


def materialise(op, m):
    """Return the dense matrix represented by op (LinearOperator or HBSMAT)."""
    return op.matmat(np.eye(m))


def materialise_inv(op, m):
    """Return the dense inverse represented by op via .solve."""
    return op.solve(np.eye(m))


def dense_inv_from_lu(lu_fac, m):
    """Reconstruct inverse from an lu_factor tuple."""
    return lu_solve(lu_fac, np.eye(m))


# ---------------------------------------------------------------------------
# main comparison
# ---------------------------------------------------------------------------

def gen_solvers():
    # Generate the real system from gen_tridiag_systems
    T_dense, S_dense, S_rk_list, rhs_phys, Stot, nc = gen_tridiag_systems()

    # Infer m and nSlabs from the returned data
    nSlabs = len(S_dense)
    m      = T_dense[0].shape[0]

    dense = RedBlackSolver(m, cyclic=False)
    dense.factorize(S_dense, T_dense)

    first_op = next(op for slab in S_rk_list for op in slab if op is not None)
    rk   = 100
    tree = first_op.tree
    quad = first_op.quad
    T_lo = [dense_to_linop(np.eye(m)) for _ in range(nSlabs)]
    hbs  = RedBlackSolverHBS(m, rk=rk, tree=tree, quad=quad, cyclic=False)
    hbs.factorize(S_rk_list, T_lo)


    return dense, hbs, Stot, rhs_phys, m, nSlabs

def my_solve_test(dense, hbs, Stot, rhs_phys, m, nSlabs):
    x = np.random.standard_normal(size=(Stot.shape[0],))
    b = Stot@x
    xhat = dense.solve(b)
    
    print("=======================================")
    print("dense rb solver error (random rhs) = ",np.linalg.norm(x-xhat)/np.linalg.norm(x))
    print("=======================================")
    
    xhat_rk = hbs.solve(b)
    
    
    print("=======================================")
    print("mod rb solver error = (random rhs) ",np.linalg.norm(x-xhat_rk)/np.linalg.norm(x))
    print("=======================================")


    xhat = dense.solve(rhs_phys)
    res = Stot@xhat-rhs_phys
    print("=======================================")
    print("dense rb solver res (phys rhs) = ",np.linalg.norm(res)/np.linalg.norm(rhs_phys))
    print("=======================================")

    
    xhat_rk = hbs.solve(rhs_phys)
    res = Stot@xhat-rhs_phys
    
    print("=======================================")
    print("HBS rb solver res = (phys rhs) ",np.linalg.norm(res)/np.linalg.norm(rhs_phys))
    print("=======================================")




def solve_test(dense, hbs, Stot, rhs_phys, m, nSlabs, tol=1e-3):
    """
    Two solve tests, run after factorize() has already been called on both solvers.

    Test 1 -- physics RHS:
        Solve with rhs_phys from gen_tridiag_systems.
        Check residual of HBS solution against the assembled Stot operator.

    Test 2 -- random manufactured solution:
        Pick x_exact at random, form rhs = Stot @ x_exact,
        solve with both solvers, compare to x_exact.

    Both tests include per-level forward-reduction diagnostics: the HBS vPrimes
    are compared against the dense vPrimes at each level to pinpoint where
    divergence begins.
    """
    N = nSlabs * m
    print(f"\n{'='*65}")
    print("Solve tests")
    print(f"{'='*65}")

    all_ok    = True
    tol_solve = tol

    def run_forward(solver, rhs_vec, is_hbs):
        """Manually run the forward reduction and return the list of vPrimes."""
        from scipy.linalg import lu_solve as _lu_solve
        RB  = solver.RB
        vPs = [rhs_vec.copy()]
        for l in range(len(RB) - 1):
            SiM, _, T_hbs_or_inv, SiP = RB[l]
            ns   = len(SiM)
            nr   = ns // 2
            vPrev = vPs[-1]
            vP    = np.zeros(m * nr)
            for j in range(nr):
                i    = 2 * j
                prev = i - 1 if i > 0       else None
                nxt  = i + 1 if i < ns - 1  else None
                c    = vPrev[i*m:(i+1)*m].copy()
                if is_hbs:
                    if prev is not None:
                        c -= SiM[i].matmat(T_hbs_or_inv[prev].solve(vPrev[prev*m:(prev+1)*m, np.newaxis]))[:, 0]
                    if nxt  is not None:
                        c -= SiP[i].matmat(T_hbs_or_inv[nxt ].solve(vPrev[nxt *m:(nxt +1)*m, np.newaxis]))[:, 0]
                else:
                    if prev is not None:
                        c -= SiM[i] @ _lu_solve(T_hbs_or_inv[prev], vPrev[prev*m:(prev+1)*m])
                    if nxt  is not None:
                        c -= SiP[i] @ _lu_solve(T_hbs_or_inv[nxt ], vPrev[nxt *m:(nxt +1)*m])
                vP[j*m:(j+1)*m] = c
            vPs.append(vP)
        return vPs

    def forward_diagnostics(rhs_vec, label):
        print(f"\n  Forward-reduction diagnostics ({label}):")
        vP_d = run_forward(dense, rhs_vec, is_hbs=False)
        vP_h = run_forward(hbs,   rhs_vec, is_hbs=True)
        for l in range(len(vP_d)):
            norm_d = np.linalg.norm(vP_d[l])
            err    = np.linalg.norm(vP_d[l] - vP_h[l]) / (norm_d + 1e-30)
            print(f"    vPrimes[{l}]: ‖dense‖={norm_d:.2e}  ‖Δ‖/‖dense‖={err:.2e}")
        return vP_d, vP_h

    def run_backward(solver, vPs_in, is_hbs):
        """Run back-substitution from a given list of vPrimes, return per-level snapshots."""
        from scipy.linalg import lu_solve as _lu_solve
        RB  = solver.RB
        vPs = [v.copy() for v in vPs_in]

        # coarsest solve
        if is_hbs:
            vPs[-1] = RB[-1][2][0].solve(vPs[-1])
        else:
            vPs[-1] = _lu_solve(RB[-1][2][0], vPs[-1])

        for l in range(len(RB) - 1, 0, -1):
            SiM, _, T_or_inv, SiP = RB[l - 1]
            nr = len(SiM) // 2
            for j in range(nr):
                i      = 2 * j
                next_j = (j + 1) % nr
                vPs[l-1][i*m:(i+1)*m] = vPs[l][j*m:(j+1)*m]
                c = (SiM[i+1].matmat(vPs[l][j*m:(j+1)*m, np.newaxis])[:, 0]
                     if is_hbs else
                     SiM[i+1] @ vPs[l][j*m:(j+1)*m])
                if j + 1 < nr:
                    c += (SiP[i+1].matmat(vPs[l][next_j*m:(next_j+1)*m, np.newaxis])[:, 0]
                          if is_hbs else
                          SiP[i+1] @ vPs[l][next_j*m:(next_j+1)*m])
                vPs[l-1][(i+1)*m:(i+2)*m] -= c
                vPs[l-1][(i+1)*m:(i+2)*m] = (
                    T_or_inv[i+1].solve(vPs[l-1][(i+1)*m:(i+2)*m])
                    if is_hbs else
                    _lu_solve(T_or_inv[i+1], vPs[l-1][(i+1)*m:(i+2)*m])
                )
        return vPs

    def backward_diagnostics(rhs_vec, label):
        """
        Run forward reduction, then back-substitution for both solvers.
        At each unwinding step l, compare the partial solution vPrimes[l-1]
        between dense and HBS -- showing where the back-sub diverges.
        Also cross-tests: HBS back-sub fed exact dense vPrimes, to isolate
        whether the error comes from the forward pass or the back-sub operators.
        """
        print(f"\n  Back-substitution diagnostics ({label}):")
        vP_d_fwd, vP_h_fwd = forward_diagnostics(rhs_vec, label)

        # Full dense and HBS back-subs
        vP_d_full = run_backward(dense, vP_d_fwd, is_hbs=False)
        vP_h_full = run_backward(hbs,   vP_h_fwd, is_hbs=True)

        # Cross-test: HBS back-sub operators fed the exact dense forward pass
        vP_h_exact_fwd = run_backward(hbs, vP_d_fwd, is_hbs=True)

        nlevels = len(vP_d_full)
        print(f"\n  {'l':>2}  {'‖x_d[l]‖':>10}  {'‖Δ full‖/‖x_d‖':>16}  {'‖Δ cross‖/‖x_d‖':>17}  note")
        print(f"  {'--':>2}  {'----------':>10}  {'----------------':>16}  {'-----------------':>17}  ----")
        for l in range(nlevels - 1, -1, -1):
            norm_d     = np.linalg.norm(vP_d_full[l])
            err_full   = np.linalg.norm(vP_d_full[l] - vP_h_full[l])       / (norm_d + 1e-30)
            err_cross  = np.linalg.norm(vP_d_full[l] - vP_h_exact_fwd[l])  / (norm_d + 1e-30)
            # cross isolates back-sub operator error; full includes forward error too
            note = "back-sub ops" if abs(err_cross - err_full) / (err_full + 1e-30) < 0.1 else "fwd+back"
            print(f"  {l:>2}  {norm_d:>10.2e}  {err_full:>16.2e}  {err_cross:>17.2e}  {note}")

    # ------------------------------------------------------------------
    # Test 1: physics RHS from gen_tridiag_systems
    # ------------------------------------------------------------------
    print("\n  Test 1: physics RHS")
    backward_diagnostics(rhs_phys, "physics RHS")
    x_dense_phys = dense.solve(rhs_phys)
    x_hbs_phys   = hbs.solve(rhs_phys)

    resid_dense  = np.linalg.norm(Stot @ x_dense_phys - rhs_phys) / (np.linalg.norm(rhs_phys) + 1e-30)
    resid_hbs    = np.linalg.norm(Stot @ x_hbs_phys   - rhs_phys) / (np.linalg.norm(rhs_phys) + 1e-30)
    err_vs_dense = np.linalg.norm(x_hbs_phys - x_dense_phys) / (np.linalg.norm(x_dense_phys) + 1e-30)

    ok1      = resid_hbs < tol_solve
    all_ok   = all_ok and ok1
    print(f"    dense residual       : {resid_dense:.2e}")
    print(f"    HBS   residual       : {resid_hbs:.2e}  {'OK' if ok1 else 'FAIL'}")
    print(f"    ‖x_hbs - x_dense‖   : {err_vs_dense:.2e}")

    # ------------------------------------------------------------------
    # Test 2: random manufactured solution
    # ------------------------------------------------------------------
    print("\n  Test 2: random manufactured solution")
    rng      = np.random.default_rng(0)
    x_exact  = rng.standard_normal(N)
    rhs_rand = Stot @ x_exact

    backward_diagnostics(rhs_rand, "random RHS")
    x_dense_rand  = dense.solve(rhs_rand)
    x_hbs_rand    = hbs.solve(rhs_rand)

    err_dense     = np.linalg.norm(x_dense_rand - x_exact) / (np.linalg.norm(x_exact) + 1e-30)
    err_hbs       = np.linalg.norm(x_hbs_rand   - x_exact) / (np.linalg.norm(x_exact) + 1e-30)
    err_vs_dense2 = np.linalg.norm(x_hbs_rand - x_dense_rand) / (np.linalg.norm(x_dense_rand) + 1e-30)

    ok2    = err_hbs < tol_solve
    all_ok = all_ok and ok2
    print(f"    dense error vs exact : {err_dense:.2e}")
    print(f"    HBS   error vs exact : {err_hbs:.2e}  {'OK' if ok2 else 'FAIL'}")
    print(f"    ‖x_hbs - x_dense‖   : {err_vs_dense2:.2e}")

    return all_ok


def main():
    dense, hbs, Stot, rhs_phys, m, nSlabs = gen_solvers()
    #solve_ok  = solve_test(dense, hbs, Stot, rhs_phys, m, nSlabs)
    my_solve_test(dense, hbs, Stot, rhs_phys, m, nSlabs)
    #all_ok    = levels_ok and solve_ok
    #print(f"\n{'='*65}")
    #print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 #if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())