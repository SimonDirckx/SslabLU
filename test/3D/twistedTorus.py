# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version
import matplotlib.tri as tri

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import multislab.omsdirectsolve as omsdirect
import torch
# validation&testing
import time
from scipy.sparse.linalg import gmres
#import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt

import geometry.geom_3D.twistedTorus as twisted

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
from petsc4py import PETSc
#try:
#    vec = PETSc.Vec().createSeq(10**9)
#    print("vec created successfully")
#except Exception as e:
#    print(e)

bnds = twisted.bnds

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

nwaves = 2.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi
kh = 1.6
# What to modify to use the Jax-based hps ("hps") or Torch-based ("hpsalt")
jax_avail   = False
torch_avail = True
hpsalt      = True


if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
elif torch_avail:
    def bfield(p,kh):
        return -kh*kh*torch.ones(p.shape[0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
param_geom=twisted.param_geom(jax_avail=jax_avail, torch_avail=torch_avail, hpsalt=hpsalt)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    z1=twisted.z1(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z2=twisted.z2(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z3=twisted.z3(p,jax_avail=jax_avail,torch_avail=torch_avail)
    rr = np.sqrt((z1-5.)**2+z2**2+z3**2)
    return np.cos(kh*rr)/(4*np.pi*rr)


N = 5
dSlabs,connectivity,H = twisted.dSlabs(N)
formulation = "hps"
solve_method = 'iterative'
#solve_method = 'direct'
HBS = False

#pvec = np.array([4,6,8,10],dtype = np.int32)
pvec = np.array([4],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
sample_time = np.zeros(shape=(len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))


p = pvec[0]
p_disc = p
if hpsalt:
    formulation = "hpsalt"
    p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt

a = np.array([H/8.,1./8,1./8])
if HBS:
    assembler = mA.rkHMatAssembler(p*p,100)
else:
    assembler = mA.denseMatAssembler()
opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)

OMS = oms.oms(dSlabs,pdo_mod,lambda p :twisted.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)
niter = 0


if solve_method == 'iterative':
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    gInfo = gmres_info()
    stol = 1e-7*H*H
    if Version(scipy.__version__)>=Version("1.14"):
        uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=1000,restart=1000)
    else:
        uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=1000,restart=1000)
    niter = gInfo.niter
elif solve_method == 'direct':
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    T,block = omsdirect.build_block_cyclic_tridiagonal_solver(OMS,S_rk_list,rhs_list,Ntot,nc)
    uhat  = omsdirect.block_cyclic_tridiagonal_solve(OMS, T, block,rhstot)

res = Stot@uhat-rhstot


print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
print("GMRES iters              = ", niter)
print("==================================")

errInf = 0.
nc = OMS.nc


nx=200
ny=200

xpts = np.linspace(-4,4,nx)
ypts = np.linspace(-4,4,ny)

ZZ = np.zeros(shape=(nx*ny,3))
ZZ[:,0] = np.kron(xpts,np.ones_like(ypts))
ZZ[:,1] = np.kron(np.ones_like(xpts),ypts)

sliceYY = np.zeros(shape=ZZ.shape)
sliceYY[:,0] = twisted.y1(ZZ,False,False)
sliceYY[:,1] = twisted.y2(ZZ,False,False)
sliceYY[:,2] = twisted.y3(ZZ,False,False)


I = np.where( (sliceYY[:,0]>=twisted.bnds[0][0]) & (sliceYY[:,0]<=twisted.bnds[1][0]) & (sliceYY[:,1]>=twisted.bnds[0][1]) & (sliceYY[:,1]<=twisted.bnds[1][1]) & (sliceYY[:,2]>=twisted.bnds[0][2]) & (sliceYY[:,2]<=twisted.bnds[1][2]) )[0]


YY = sliceYY[I,:]
gYY = np.zeros(shape=(YY.shape[0],))


sliceZZ = np.zeros(shape=(len(I),3))
sliceZZ[:,0] = twisted.z1(YY,False,False)
sliceZZ[:,1] = twisted.z2(YY,False,False)
sliceZZ[:,2] = twisted.z3(YY,False,False)

ucheck = np.zeros(shape=(Stot.shape[0],))
rhscheck = np.zeros(shape=(Stot.shape[1],))

XXcprev = np.zeros(shape=(nc,3))

ul_list = []
ur_list = []

for slabInd in range(len(dSlabs)):
    geom    = np.array(dSlabs[slabInd])
    I0 = np.where(  (YY[:,0]>=geom[0,0]) & (YY[:,0]<=geom[1,0]) & (YY[:,1]>=geom[0,1]) & (YY[:,1]<=geom[1,1]) & (YY[:,2]>=geom[0,2]) & (YY[:,2]<=geom[1,2]) )[0]
    YY0 = YY[I0,:]
    slab_i  = oms.slab(geom,lambda p : twisted.gb(p,jax_avail=jax_avail,torch_avail=torch_avail))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
    startL = ((slabInd-1)%N)
    startR = ((slabInd+1)%N)
    ul = uhat[startL*nc:(startL+1)*nc]
    ur = uhat[startR*nc:(startR+1)*nc]

    ul_list.append(ul)
    ur_list.append(ur)
        

###
###
### Jax run as comparison
###
###
bnds = twisted.bnds

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

nwaves = 2.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi
kh = 1.6
# What to modify to use the Jax-based hps ("hps") or Torch-based ("hpsalt")
jax_avail   = True
torch_avail = False
hpsalt      = False


if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
elif torch_avail:
    def bfield(p,kh):
        return -kh*kh*torch.ones(p.shape[0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
param_geom=twisted.param_geom(jax_avail=jax_avail, torch_avail=torch_avail, hpsalt=hpsalt)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    z1=twisted.z1(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z2=twisted.z2(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z3=twisted.z3(p,jax_avail=jax_avail,torch_avail=torch_avail)
    rr = np.sqrt((z1-5.)**2+z2**2+z3**2)
    return np.cos(kh*rr)/(4*np.pi*rr)


N = 5
dSlabs,connectivity,H = twisted.dSlabs(N)
formulation = "hps"
solve_method = 'iterative'
#solve_method = 'direct'
HBS = False

#pvec = np.array([4,6,8,10],dtype = np.int32)
pvec = np.array([4],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
sample_time = np.zeros(shape=(len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))


p = pvec[0]
p_disc = p
if hpsalt:
    formulation = "hpsalt"
    p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt

a = np.array([H/8.,1./8,1./8])
if HBS:
    assembler = mA.rkHMatAssembler(p*p,100)
else:
    assembler = mA.denseMatAssembler()
opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)

OMS = oms.oms(dSlabs,pdo_mod,lambda p :twisted.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
S_rk_list_jax, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)
niter = 0


if solve_method == 'iterative':
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list_jax,rhs_list,Ntot,nc,dbg=2)
    gInfo = gmres_info()
    stol = 1e-7*H*H
    if Version(scipy.__version__)>=Version("1.14"):
        uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=1000,restart=1000)
    else:
        uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=1000,restart=1000)
    niter = gInfo.niter
elif solve_method == 'direct':
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list_jax,rhs_list,Ntot,nc,dbg=2)
    T,block = omsdirect.build_block_cyclic_tridiagonal_solver(OMS,S_rk_list_jax,rhs_list,Ntot,nc)
    uhat  = omsdirect.block_cyclic_tridiagonal_solve(OMS, T, block,rhstot)

res = Stot@uhat-rhstot


print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
print("GMRES iters              = ", niter)
print("==================================")

errInf = 0.
nc = OMS.nc


nx=200
ny=200

xpts = np.linspace(-4,4,nx)
ypts = np.linspace(-4,4,ny)

ZZ = np.zeros(shape=(nx*ny,3))
ZZ[:,0] = np.kron(xpts,np.ones_like(ypts))
ZZ[:,1] = np.kron(np.ones_like(xpts),ypts)

sliceYY = np.zeros(shape=ZZ.shape)
sliceYY[:,0] = twisted.y1(ZZ,False,False)
sliceYY[:,1] = twisted.y2(ZZ,False,False)
sliceYY[:,2] = twisted.y3(ZZ,False,False)


I = np.where( (sliceYY[:,0]>=twisted.bnds[0][0]) & (sliceYY[:,0]<=twisted.bnds[1][0]) & (sliceYY[:,1]>=twisted.bnds[0][1]) & (sliceYY[:,1]<=twisted.bnds[1][1]) & (sliceYY[:,2]>=twisted.bnds[0][2]) & (sliceYY[:,2]<=twisted.bnds[1][2]) )[0]


YY = sliceYY[I,:]
gYY = np.zeros(shape=(YY.shape[0],))


sliceZZ = np.zeros(shape=(len(I),3))
sliceZZ[:,0] = twisted.z1(YY,False,False)
sliceZZ[:,1] = twisted.z2(YY,False,False)
sliceZZ[:,2] = twisted.z3(YY,False,False)

ucheck = np.zeros(shape=(Stot.shape[0],))
rhscheck = np.zeros(shape=(Stot.shape[1],))

XXcprev = np.zeros(shape=(nc,3))

ul_list_jax = []
ur_list_jax = []

for slabInd in range(len(dSlabs)):
    geom    = np.array(dSlabs[slabInd])
    I0 = np.where(  (YY[:,0]>=geom[0,0]) & (YY[:,0]<=geom[1,0]) & (YY[:,1]>=geom[0,1]) & (YY[:,1]<=geom[1,1]) & (YY[:,2]>=geom[0,2]) & (YY[:,2]<=geom[1,2]) )[0]
    YY0 = YY[I0,:]
    slab_i  = oms.slab(geom,lambda p : twisted.gb(p,jax_avail=jax_avail,torch_avail=torch_avail))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
    startL = ((slabInd-1)%N)
    startR = ((slabInd+1)%N)
    ul = uhat[startL*nc:(startL+1)*nc]
    ur = uhat[startR*nc:(startR+1)*nc]

    ul_list_jax.append(ul)
    ur_list_jax.append(ur)

print("Torch operators")
print(S_rk_list)

print("Jax operators")
print(S_rk_list_jax)

print("Torch Arrays:")
print(ul_list)
print(ur_list)

print("Jax Arrays:")
print(ul_list_jax)
print(ur_list_jax)

print(type(S_rk_list[0][0]))
print(type(S_rk_list_jax[0][0]))

print(type(ul_list[0]))
print(type(ul_list_jax[0]))

#
# Now comparing the two:
#
print("Relative Errors for S_rk_list:")
for i in range(len(ul_list)):
    print(np.linalg.norm(S_rk_list[i][0] - S_rk_list_jax[i][0]) / np.linalg.norm(S_rk_list_jax[i][0]))
    print(np.linalg.norm(S_rk_list[i][-1] - S_rk_list_jax[i][-1]) / np.linalg.norm(S_rk_list_jax[i][-1]))


print("Relative Errors for uhat:")
for i in range(len(ul_list)):
    print(np.linalg.norm(ul_list[i] - ul_list_jax[i]) / np.linalg.norm(ul_list_jax[i]))
    print(np.linalg.norm(ur_list[i] - ur_list_jax[i]) / np.linalg.norm(ur_list_jax[i]))