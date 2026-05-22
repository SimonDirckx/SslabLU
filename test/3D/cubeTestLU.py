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
from scipy.sparse.linalg import LinearOperator




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


#### TOGGLE FOR HPSMULTIDOMAIN (SEE KUMP ET AL.)
jax_avail   = False
torch_avail = not jax_avail
hpsalt      = torch_avail
kh = 25.
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


N = 3
dSlabs,connectivity,H = cube.dSlabs(N)
print("len dSlabs = ",len(dSlabs))
print("H = ",H)
print("connectivity = ",connectivity)
pvec = np.array([8],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
sample_time = np.zeros(shape=(len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))

solve_method = 'iterative'
formulation = "hps"
tridiag = (solve_method=='direct')
for indp in range(len(pvec)):
    p = pvec[indp]
    p_disc = p
    if hpsalt:
        formulation = "hpsalt"
        p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
    a = np.array([H/8,1/8,1/8])
    assembler = mA.rkHMatAssembler(p*p,200,ndim=3)
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
    OMS_lu = oms.oms_lu(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    OMS_rk = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    tic = time.time()
    S_lu_list, rhs_list, Ntot, nc = OMS_lu.construct_Stot_helper(bc)
    t_constr_lu = time.time()-tic
    tic = time.time()
    S_rk_list, rhs_list, Ntot, nc = OMS_rk.construct_Stot_helper(bc,assembler)
    t_constr_rk = time.time()-tic
    Stot_lu,_  = OMS_lu.construct_Stot_and_rhstot_linearOperator(S_lu_list,rhs_list,Ntot,nc,dbg=2)
    Stot_rk,_  = OMS_rk.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
    v= np.random.standard_normal(size=(Stot_rk.shape[0],))
    tic = time.time()
    u = Stot_lu@v
    t_lu = time.time()-tic
    tic = time.time()
    u = Stot_rk@v
    t_rk = time.time()-tic

    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("npan_dim                 = ",(int)(H/a[0]),',',(int)(.5/a[1]))
    print("nc                       = ",OMS_lu.nc)
    print("time construct LU        = ", t_constr_lu)
    print("time construct rk        = ", t_constr_rk)
    print("time matvec LU           = ", t_lu)
    print("time matvec LU           = ", t_rk)
    print("==================================")