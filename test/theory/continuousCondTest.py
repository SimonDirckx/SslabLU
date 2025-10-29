# basic packages
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
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo_alt
import solver.spectralmultidomain.hps.pdo as pdo
import multislab.omsdirectsolve as omsdirect
import solver.spectral.spectralSolver as spectral
import solver.stencil.stencilSolver as stencil







# validation&testing
import time
from scipy.sparse.linalg import gmres
#import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt

import geometry.geom_2D.square as square
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



laplace = False

if laplace:
    def c11(p):
        return np.ones_like(p[:,0])
    def c22(p):
        return np.ones_like(p[:,0])
else:
    def c11(p):
        return (1.+.5*np.cos(2*np.pi*p[:,0]))
    def c22(p):
        return (1.+.5*p[:,0]*p[:,0]*np.sin(3*np.pi*p[:,1]))

diff_op=pdo.PDO2d(c11,c22)

def bc(p):
    return np.ones_like(p[:,0])


#formulation = 'spectral'
formulation = 'stencil'

N = 8
dSlabs,connectivity,H = square.dSlabs(N)
print(connectivity)
pvec = np.array([64],dtype = np.int64)
#pvec = np.array([8,10,12,14,16,18,20],dtype = np.int64)
for indp in range(len(pvec)):
    p = pvec[indp]
    if formulation=='spectral':
        py = p
        px = 2*(256//N)
    else:
        py = p
        px = 2*(256//N)+1
    assembler = mA.denseMatAssembler()
    opts = solverWrap.solverOptions(formulation,[px,py])
    OMS = oms.oms(dSlabs,diff_op,lambda p :square.gb(p,False,False),opts,connectivity)
    print("computing S blocks & rhs's...")
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=0)
    print("done")
    Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=0)
    Sdense = Stot@np.identity(Stot.shape[1])
    nc = OMS.nc
    if formulation=='spectral':
        x,w = spectral.clenshaw_curtis_compute(py+1)
        w0 = w[1:py]
        w0 = w0[::-1]
        w0 = np.sqrt(w0/2)
        W = np.kron(np.identity(N-1),np.diag(w0))
        SW = W@(Sdense@(np.linalg.inv(W)))
    else:
        SW = Sdense
    mu_block = 0.#np.linalg.norm(SW-SW.T,ord=2)
    for i in range(N-1):
        Sij = SW[i*nc:(i+1)*nc,:][:,(i+1)*nc:(i+2)*nc]
        Sji = SW[(i+1)*nc:(i+2)*nc,:][:,i*nc:(i+1)*nc]
        err = np.linalg.norm(Sij-Sji.T,ord=np.inf)
        print("i//err = ",i,"//",err)
        mu_block = np.max([mu_block,err])


    [_,s,_] = np.linalg.svd(SW)
    [e,_] = np.linalg.eig(SW)
    e = np.sort(abs(e))
    e=e[::-1]

    kh_rho = max(e)/min(e)
    kh_2 = max(s)/min(s)
    mu_inf = max(s-e)

    print("kh_2/kh_rho = ",kh_2/kh_rho)
    print("mu_inf = ",mu_inf)
    print("mu_block = ",mu_block)


