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

import geometry.geom_3D.cube as cube
class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.resList=[]
    def __call__(self, rk=None):
        self.niter += 1
        self.resList+=[rk]
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


#### TOGGLE FOR HPSMULTIDOMAIN (SEE KUMP ET AL.)
def compare_cube(nwaves, N, p):
    jax_avail   = False
    torch_avail = not jax_avail
    hpsalt      = torch_avail

    #nwaves = 15
    kh     = 2. * np.pi * nwaves

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

    #solve_method = 'iterative'
    solve_method = 'direct'
    formulation = "hps"
    tridiag = (solve_method=='direct')

    p_disc = p

    if hpsalt:
        formulation = "hpsalt"
        p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
    a = np.array([H/8,1/8,1/8])
    assembler = mA.rkHMatAssembler(p*p,75)
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
    OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    print("computing S blocks & rhs's...")
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)
    print("done")
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    niter = 0

    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    gInfo = gmres_info()
    stol = 1e-10*H*H

    start_time = time.perf_counter()
    if Version(scipy.__version__)>=Version("1.14"):
        uhat_iter,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=500,restart=500)
    else:
        uhat_iter,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=500,restart=500)
    end_time = time.perf_counter()
    elapsed_time_iterative = end_time - start_time

    niter = gInfo.niter

    rhstot = np.zeros(shape = (Ntot,))
    for i in range(len(rhs_list)):
        rhstot[i*nc:(i+1)*nc] = rhs_list[i]

    start_time = time.perf_counter()
    T = omsdirect.build_block_tridiagonal_solver(S_rk_list)
    end_time = time.perf_counter()
    elapsed_time_direct_factor_cyclical = end_time - start_time

    start_time = time.perf_counter()
    uhat_tridiagonal  = omsdirect.block_tridiagonal_solve(OMS, T, rhstot)
    end_time = time.perf_counter()
    elapsed_time_direct_solve_cyclical = end_time - start_time

    start_time = time.perf_counter()
    RB, S = omsdirect.build_block_RB_solver(OMS, S_rk_list, rhs_list, Ntot, nc, cyclic=False)
    end_time = time.perf_counter()
    elapsed_time_direct_factor_RB = end_time - start_time

    start_time = time.perf_counter()
    uhat_redblack = omsdirect.block_RB_solve((RB, S), rhstot)
    end_time = time.perf_counter()
    elapsed_time_direct_solve_RB = end_time - start_time
    
    res_iter   = Stot @ uhat_iter   - rhstot
    res_tridiagonal = Stot @ uhat_tridiagonal - rhstot
    res_RB = Stot @ uhat_redblack - rhstot

    print(f"Elapsed time for iterative solve: {elapsed_time_iterative} seconds")
    print(f"Elapsed time for direct factorization with tridiagonal: {elapsed_time_direct_factor_cyclical} seconds")
    print(f"Elapsed time for direct solve with tridiagonal: {elapsed_time_direct_solve_cyclical} seconds")
    print(f"Elapsed time for direct factorization with red-black: {elapsed_time_direct_factor_RB} seconds")
    print(f"Elapsed time for direct solve with cyclic red-black: {elapsed_time_direct_solve_RB} seconds")

    
    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("L2 rel. res iterative    = ", np.linalg.norm(res_iter)/np.linalg.norm(rhstot))
    print("L2 rel. res tridiagonal  = ", np.linalg.norm(res_tridiagonal)/np.linalg.norm(rhstot))
    print("L2 rel. res red-black    = ", np.linalg.norm(res_RB)/np.linalg.norm(rhstot))
    print("GMRES iters              = ", niter)
    print("==================================")

    nc = OMS.nc
    err_tot_iter   = 0
    err_tot_tridiagonal = 0
    err_tot_redblack = 0

    for slabInd in range(len(dSlabs)):
        geom    = np.array(dSlabs[slabInd])
        slab_i  = oms.slab(geom,lambda p : cube.gb(p,jax_avail,torch_avail))
        solver  = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom,Helm,False,False)
        Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
        startL = slabInd-1
        startR = slabInd+1

        ghat_iter = np.zeros(shape=(XXb.shape[0],))
        ghat_iter[Igb] = bc(XXb[Igb,:])
        if startL>-1:
            ghat_iter[Il] = uhat_iter[startL*nc:(startL+1)*nc]
        if startR<len(dSlabs):
            ghat_iter[Ir] = uhat_iter[startR*nc:(startR+1)*nc]

        ghat_tridiagonal = np.zeros(shape=(XXb.shape[0],))
        ghat_tridiagonal[Igb] = bc(XXb[Igb,:])
        if startL>-1:
            ghat_tridiagonal[Il] = uhat_tridiagonal[startL*nc:(startL+1)*nc]
        if startR<len(dSlabs):
            ghat_tridiagonal[Ir] = uhat_tridiagonal[startR*nc:(startR+1)*nc]

        ghat_redblack = np.zeros(shape=(XXb.shape[0],))
        ghat_redblack[Igb] = bc(XXb[Igb,:])
        if startL>-1:
            ghat_redblack[Il] = uhat_redblack[startL*nc:(startL+1)*nc]
        if startR<len(dSlabs):
            ghat_redblack[Ir] = uhat_redblack[startR*nc:(startR+1)*nc]
        
        g = bc(XXb)

        err_loc_iter = np.linalg.norm(ghat_iter-g)/np.linalg.norm(g)
        err_tot_iter = np.max([err_loc_iter,err_tot_iter])

        err_loc_tridiagonal = np.linalg.norm(ghat_tridiagonal-g)/np.linalg.norm(g)
        err_tot_tridiagonal = np.max([err_loc_tridiagonal,err_tot_tridiagonal])

        err_loc_redblack = np.linalg.norm(ghat_redblack-g)/np.linalg.norm(g)
        err_tot_redblack = np.max([err_loc_redblack,err_tot_redblack])

        print("===================LOCAL ERR===================")
        print("err ghat iterative = ", err_loc_iter)
        print("err ghat tridiagonal = ", err_loc_tridiagonal)
        print("err ghat red-black = ", err_loc_redblack)
        print("===============================================")
    
    print("===================GLOBAL ERR===================")
    print("err_tot iterative = ",err_tot_iter)
    print("err_tot tridiagonal = ", err_tot_tridiagonal)
    print("err_tot red-black = ", err_tot_redblack)
    print("===============================================")

    return err_tot_iter, err_tot_tridiagonal, err_tot_redblack, elapsed_time_iterative, elapsed_time_direct_factor_cyclical, elapsed_time_direct_solve_cyclical, elapsed_time_direct_factor_RB, elapsed_time_direct_solve_RB

outputs = compare_cube(10, 9, 8)

print(outputs)
