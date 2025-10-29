import numpy as np
import hps.pdo as pdo
import matplotlib.pyplot as plt
import multislab.oms as oms
import matAssembly.matAssembler as mA
import solver.solver as solverWrap
import solver.spectralmultidomain.hps.pdo as pdo
from solver.stencil.stencilSolver import stencilSolver as stencil
import solver.stencil.geom as stencilGeom
from matplotlib import cm
from multislab.oms import slab
from scipy.sparse.linalg import LinearOperator
from solver.spectral import spectralSolver as spectral
import geometry.geom_2D.square as square
from scipy.sparse.linalg import gmres
from packaging.version import Version
import scipy
import jax.numpy as jnp

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


laplace = True

if laplace:
    def c11(p):
        return jnp.ones_like(p[...,0])
    def c22(p):
        return jnp.ones_like(p[...,0])
else:
    def c11(p):
        return (1.+.5*jnp.cos(2*jnp.pi*p[...,0]))
    def c22(p):
        return (1.+.5*p[...,0]*p[...,0]*jnp.sin(3*jnp.pi*p[...,1]))

diff_op=pdo.PDO2d(c11=c11,c22=c22)

def bc(p):
    return np.random.standard_normal(size=(p.shape[0],))

kvec = [2,3,4,5,6]
nGMRES_vec = np.zeros(shape=(len(kvec),))
Hvec = np.zeros(shape=(len(kvec),))

method = 'hps'
jax_avail = True
torch_avail = not jax_avail
p = 10
for indk in range(len(kvec)):
    k = kvec[indk]
    N = (2**k)
    dSlabs,connectivity,H = square.dSlabs(N)
    Hvec[indk] = H
    print(" H = ",H)
    a = [1./128,1./128.]

    assembler = mA.denseMatAssembler()
    opts = solverWrap.solverOptions(method,[p,p],a)
    OMS = oms.oms(dSlabs,diff_op,lambda p :square.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    print("computing S blocks & rhs's...")
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)
    print("done")
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    gInfo = gmres_info()
    stol = 1e-5*H*H

    if Version(scipy.__version__)>=Version("1.14"):
        uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=500,restart=500)
    else:
        uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=500,restart=500)
    niter = gInfo.niter    
    nGMRES_vec[indk] = niter
    res = Stot@uhat-rhstot
    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
    print("GMRES iters              = ", niter)
    print("==================================")

eqstr = 'VC'
if laplace:
    eqstr='laplace'

fileName = 'nGMRES_'+eqstr+'.csv'
gmresMat = np.zeros(shape=(len(kvec),2))
gmresMat[:,0] = Hvec
gmresMat[:,1] = nGMRES_vec
with open(fileName,'w') as f:
    f.write('H,nGMRES\n')
    np.savetxt(f,gmresMat,fmt='%.8e',delimiter=',')