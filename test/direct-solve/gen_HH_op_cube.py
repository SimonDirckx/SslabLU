import numpy as np
import direct_solve.omsdirectsolveHBS as omsdirect

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
from scipy.sparse.linalg import LinearOperator
import matAssembly.HBS.slabTree as slabTree



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

def get_HH_op_cube(kh,N,p,a,dense=True,rk=0):
    #### TOGGLE FOR HPSMULTIDOMAIN (SEE KUMP ET AL.)
    jax_avail   = False
    torch_avail = not jax_avail
    hpsalt      = torch_avail
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


    dSlabs,connectivity,H = cube.dSlabs(N)
    print("H/2 = ",H/2)
    print("a = ",a)
    #a = np.array([H/2,1/16,1/16])
    print("connectivity = ",connectivity)
    formulation = "hps"
    p_disc = p
    if hpsalt:
        formulation = "hpsalt"
        p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
    if dense:
        assembler = mA.denseMatAssembler()
    else:
        assembler = mA.rkHMatAssembler(p*p,rk,ndim=3)
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a,reduced_gpu=True)
    OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity,stiff_mat_const=True)
    S_rk_list, _, _, _ = OMS.construct_Stot_helper(bc, assembler, dbg=2)
    tree = slabTree.slabTree(assembler.XXI,False,p*p)
    return S_rk_list,tree