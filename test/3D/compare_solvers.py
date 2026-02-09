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

nwaves = 10.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi

# What to modify to use the Jax-based hps ("hps") or Torch-based ("hpsalt")
jax_avail    = False
torch_avail  = True
hpsalt       = True
direct_solve = False

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
p = 10
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

elapsed_time_direct_factor = end_time - start_time

start_time   = time.perf_counter()
uhat_direct  = omsdirectsolve.block_cyclic_tridiagonal_solve(OMS, T, smw_block, rhstot)
end_time     = time.perf_counter()
elapsed_time_direct_solve = end_time - start_time

RB, S = omsdirectsolve.build_block_RB_solver(OMS, S_rk_list, rhs_list, Ntot, nc, cyclic=True)
uhat_RB = omsdirectsolve.block_RB_solve((RB, S), rhstot)

print("First we'll compare cyclic block tridiagonal to RB, slab by slab. This is inf-norm absolute error:")
m = S_rk_list[0][0].shape[0]
for i in range(N):
    print(np.linalg.norm(uhat_RB[i*m:(i+1)*m] - uhat_direct[i*m:(i+1)*m]))


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

print("\nLet's look at absolute minima on each slab. They should be similar regardless of solver used:\n")
print("GMRES:", np.min(np.abs(uhat)))
for i in range(N):
    print(np.min(np.abs(uhat[i*m:(i+1)*m])))

print("Cyclic Diagonal:", np.min(np.abs(uhat_direct)))
for i in range(N):
    print(np.min(np.abs(uhat_direct[i*m:(i+1)*m])))

print("Red-Black:", np.min(np.abs(uhat_RB)))
for i in range(N):
    print(np.min(np.abs(uhat_RB[i*m:(i+1)*m])))

#print(np.linalg.norm(uhat_direct - uhat) / np.linalg.norm(uhat))


print("Now let's look at the relative error of iterative solve vs direct solve with Thomas Algorithm plus SMW:")
print(np.linalg.norm(uhat_direct - uhat) / np.linalg.norm(uhat))

print("And now let's look at the relative error of iterative solve vs direct solve with Red-Black Algorithm:")
print(np.linalg.norm(uhat_RB - uhat) / np.linalg.norm(uhat))


print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p_disc)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
print("GMRES iters              = ", gInfo.niter)
print("==================================")

print("Finally, we'll take the left and right sides of the overlapping slabs and compare them to the true solution, for each solver:")

errInf = 0.
nc = OMS.nc
for slabInd in range(len(connectivity)):
    geom    = np.array(dSlabs[slabInd])
    slab_i  = oms.slab(geom,lambda p : squareTorus.gb(p,jax_avail=jax_avail,torch_avail=torch_avail))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)

    startL = ((slabInd-1)%N)
    startR = ((slabInd+1)%N)
    ul = uhat[startL*nc:(startL+1)*nc]
    ur = uhat[startR*nc:(startR+1)*nc]
    g = np.zeros(shape=(XXb.shape[0],))
    g[Il]=ul
    g[Ir]=ur

    ul_true = bc(XXb[Il,:])
    ur_true = bc(XXb[Ir,:])

    err_gmresL = np.linalg.norm(ul - ul_true.detach().cpu().numpy(),ord=np.inf)
    err_gmresR = np.linalg.norm(ur - ur_true.detach().cpu().numpy(),ord=np.inf)

    ul_RB = uhat_RB[startL*nc:(startL+1)*nc]
    ur_RB = uhat_RB[startR*nc:(startR+1)*nc]

    ul_direct = uhat_direct[startL*nc:(startL+1)*nc]
    ur_direct = uhat_direct[startR*nc:(startR+1)*nc]

    err_directL = np.linalg.norm(ul_direct - ul_true.detach().cpu().numpy(),ord=np.inf)
    err_directR = np.linalg.norm(ur_direct - ur_true.detach().cpu().numpy(),ord=np.inf)

    err_RBL = np.linalg.norm(ul_RB - ul_true.detach().cpu().numpy(),ord=np.inf)
    err_RBR = np.linalg.norm(ur_RB - ur_true.detach().cpu().numpy(),ord=np.inf)

    print("GMRES errL, errR:", err_gmresL, err_gmresR)
    print("Cyclic errL, errR:", err_directL, err_directR)
    print("Red-black errL, errR:", err_RBL, err_RBR)

