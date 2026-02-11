# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version


# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import multislab.omsdirectsolve as omsdirectsolve

# validation&testing
import time
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import geometry.geom_3D.squareTorus as squareTorus

import torch

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


bnds = squareTorus.bnds

################################################################
#
#   SET-UP BVP:         Helmholtz on 3D Annulus
#   - wave number       (kh)
#   - bfield            (= kh*ones)
#   - pdo_mod           (pdo transformed to square)
#   - BC
#   - known exact sol.  (u_exact)
#
################################################################

nwaves = 15.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi

# What to modify to use the Jax-based hps ("hps") or Torch-based ("hpsalt")
jax_avail    = False
torch_avail  = True
hpsalt       = True
direct_solve = True

if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
elif torch_avail:
    def bfield(p,kh):
        return -kh*kh*torch.ones(p.shape[0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
param_geom=squareTorus.param_geom(jax_avail=jax_avail, torch_avail=torch_avail, hpsalt=hpsalt)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    z=squareTorus.z1(p,jax_avail=jax_avail,torch_avail=torch_avail)
    return np.sin(kh*z)

def u_exact(p):
    z=squareTorus.z1(p,jax_avail=jax_avail,torch_avail=torch_avail)
    return np.sin(kh*z)

N = 8
dSlabs,connectivity,H = squareTorus.dSlabs(N)

formulation = "hps"
p = 14
p_disc = p
if hpsalt:
    formulation = "hpsalt"
    p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
a = np.array([H/8.,1/8,1/8])
assembler = mA.rkHMatAssembler(p*p,50)
opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
OMS = oms.oms(dSlabs,pdo_mod,lambda p :squareTorus.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)

S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)

Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)

# TEST: zero out select S_rk_list.

start_time   = time.perf_counter()
T, smw_block = omsdirectsolve.build_block_cyclic_tridiagonal_solver(OMS, S_rk_list, rhs_list, Ntot, nc)
end_time     = time.perf_counter()
elapsed_time_direct_factor_cyclical = end_time - start_time

start_time   = time.perf_counter()
uhat_direct  = omsdirectsolve.block_cyclic_tridiagonal_solve(OMS, T, smw_block, rhstot)
end_time     = time.perf_counter()
elapsed_time_direct_solve_cyclical = end_time - start_time

start_time   = time.perf_counter()
RB, S = omsdirectsolve.build_block_RB_solver(OMS, S_rk_list, rhs_list, Ntot, nc, cyclic=True)
end_time     = time.perf_counter()
elapsed_time_direct_factor_RB = end_time - start_time

start_time   = time.perf_counter()
uhat_RB = omsdirectsolve.block_RB_solve((RB, S), rhstot)
end_time     = time.perf_counter()
elapsed_time_direct_solve_RB = end_time - start_time


gInfo = gmres_info()
stol = 1e-8*H*H

start_time = time.perf_counter()
if Version(scipy.__version__)>=Version("1.14"):
    uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=300,restart=300)
else:
    uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=300,restart=300)
end_time = time.perf_counter()
elapsed_time_iterative = end_time - start_time

stop_solve = time.time()
res = Stot@uhat-rhstot

print(f"Elapsed time for iterative solve: {elapsed_time_iterative} seconds")
print(f"Elapsed time for direct factorization with cyclic tridiagonal: {elapsed_time_direct_factor_cyclical} seconds")
print(f"Elapsed time for direct solve with cyclic tridiagonal: {elapsed_time_direct_solve_cyclical} seconds")
print(f"Elapsed time for direct factorization with red-black: {elapsed_time_direct_factor_RB} seconds")
print(f"Elapsed time for direct solve with cyclic red-black: {elapsed_time_direct_solve_RB} seconds")

#if direct_solve:
#    print("We'll use solution from direct solver to get overall result:")
#    uhat = uhat_RB

print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p_disc)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
print("GMRES iters              = ", gInfo.niter)
print("==================================")

errInf_iterative = 0.
errInf_direct = 0.
errInf_RB = 0.
nc = OMS.nc
for slabInd in range(len(connectivity)):
    geom    = np.array(dSlabs[slabInd])
    slab_i  = oms.slab(geom,lambda p : squareTorus.gb(p,jax_avail=jax_avail,torch_avail=torch_avail))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)

    startL = ((slabInd-1)%N)
    startR = ((slabInd+1)%N)

    ul_iterative = uhat[startL*nc:(startL+1)*nc]
    ur_iterative = uhat[startR*nc:(startR+1)*nc]
    g_iterative = np.zeros(shape=(XXb.shape[0],))
    g_iterative[Il]=ul_iterative
    g_iterative[Ir]=ur_iterative
    g_iterative[Igb] = bc(XXb[Igb,:])
    g_iterative=g_iterative[:,np.newaxis]
    g_iterative = torch.from_numpy(g_iterative)
    uu_iterative = solver.solver.solve_dir_full(g_iterative)

    ul_direct = uhat_direct[startL*nc:(startL+1)*nc]
    ur_direct = uhat_direct[startR*nc:(startR+1)*nc]
    g_direct = np.zeros(shape=(XXb.shape[0],))
    g_direct[Il]=ul_direct
    g_direct[Ir]=ur_direct
    g_direct[Igb] = bc(XXb[Igb,:])
    g_direct=g_direct[:,np.newaxis]
    g_direct = torch.from_numpy(g_direct)
    uu_direct = solver.solver.solve_dir_full(g_direct)

    ul_RB = uhat_RB[startL*nc:(startL+1)*nc]
    ur_RB = uhat_RB[startR*nc:(startR+1)*nc]
    g_RB = np.zeros(shape=(XXb.shape[0],))
    g_RB[Il]=ul_RB
    g_RB[Ir]=ur_RB
    g_RB[Igb] = bc(XXb[Igb,:])
    g_RB=g_RB[:,np.newaxis]
    g_RB = torch.from_numpy(g_RB)
    uu_RB = solver.solver.solve_dir_full(g_RB)

    uu0 = bc(solver.XXfull)

    uu_iterative=uu_iterative.flatten()
    errI_iterative=np.linalg.norm(uu_iterative-uu0,ord=np.inf)
    errInf_iterative = np.max([errInf_iterative,errI_iterative])
    print(errI_iterative)

    uu_direct=uu_direct.flatten()
    errI_direct=np.linalg.norm(uu_direct-uu0,ord=np.inf)
    errInf_direct = np.max([errInf_direct,errI_direct])
    print(errI_direct)

    uu_RB=uu_RB.flatten()
    errI_RB=np.linalg.norm(uu_RB-uu0,ord=np.inf)
    errInf_RB = np.max([errInf_RB,errI_RB])
    print(errI_RB)

print("sup norm error for iterative u = ",errInf_iterative)
print("sup norm error for direct cyclic tridiagonal u = ",errInf_direct)
print("sup norm error for direct red-black u = ",errInf_RB)