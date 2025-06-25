import numpy as np
import jax.numpy as jnp
import solver.spectralmultidomain.hps.pdo as pdo
from packaging.version import Version
import scipy

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
from scipy.sparse.linalg import gmres


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

kh=5.
def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,0])
def c33(p):
    return jnp.ones_like(p[...,0])
def bfield(p):
    return kh*kh*jnp.ones_like(p[...,:])
helmholtz = pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=pdo.const(-kh*kh))

bnds = [[0.,0.,0.],[1.,1.,1.]]
box_geom   = jnp.array(bnds)

def gb(p):
    return np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14 or np.abs(p[2]-bnds[0][2])<1e-14 or np.abs(p[2]-bnds[1][2])<1e-14
def bc(p):
    return jnp.ones_like(p[...,0])

H = 1./64.
N = (int)(1./H)

slabs = []
for n in range(N):
    bnds_n = [[n*H,0.,0.],[(n+1)*H,1.,1.]]
    slabs+=[bnds_n]

connectivity = []
for i in range(N-1): 
    connectivity+=[[i,i+1]]

if_connectivity = []
for i in range(N-1):
    if i==0:
        if_connectivity+=[[-1,(i+1)]]
    elif i==N-2:
        if_connectivity+=[[(i-1),-1]]
    else:
        if_connectivity+=[[(i-1),(i+1)]]


p = 10
a = [H/2.,1/8,1/8]

assembler = mA.rkHMatAssembler(p,p)
opts = solverWrap.solverOptions('hps',[p,p,p],a)
#assembler = mA.denseMatAssembler()#((p+2)*(p+2),50)
OMS = oms.oms(slabs,helmholtz,gb,opts,connectivity,if_connectivity)
print("computing Stot & rhstot...")
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,1)
print("done")


gInfo = gmres_info()
stol = 1e-10*H*H

if Version(scipy.__version__)>=Version("1.14"):
    uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=200,restart=200)
else:
    uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=200,restart=200)
res = Stot@uhat-rhstot

print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
print("GMRES iters              = ", gInfo.niter)
print("==================================")


