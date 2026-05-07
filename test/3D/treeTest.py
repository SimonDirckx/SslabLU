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
import multislab.omsdirectsolveHBS as omsdirectHBS


import geometry.geom_3D.cube as cube
from matAssembly.HBS.simpleoctree import simpletree as tree
import matAssembly.HBS.slabTree as slabTree


def compute_c0_L0(XX):
    """
    compute the center and charasteristic length of the domain to be clustered
    (this is necessary only becuase there is a bug in simpletree)
    
    """
    N,ndim = XX.shape
    XX_np = XX
    if torch.is_tensor(XX):
        XX_np = XX_np.detach().cpu().numpy()
    c0 = np.sum(XX_np,axis=0)/N
    L0 = np.max(np.max(XX_np,axis=0)-np.min(XX_np,axis=0)) #too tight for some reason
    return c0,L0+1e-5


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
kh = 2.
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


N = 8
dSlabs,connectivity,H = cube.dSlabs(N)
p = 6

p_disc = p
formulation = "hpsalt"
p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
a = np.array([H/2,1/32,1/32])
slabInd = 1
geom    = np.array(dSlabs[slabInd])
slab_i  = oms.slab(geom,lambda p : cube.gb(p,jax_avail,torch_avail))
opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
solver  = oms.solverWrap.solverWrapper(opts)
solver.construct(geom,Helm,False,False)
Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
XXc = XXi[Ic,:]
XXr = XXb[Ir,:]
XXl = XXb[Il,:]
assembler = mA.rkHMatAssembler(p*p,200)
c0,L0 = compute_c0_L0(XXc)
tree0 =  slabTree.slabTree(XXc,False,p*p)
L = tree0.nlevels-3
leaves = tree0.get_boxes_level(L-1)
print("number of leaves = ",len(leaves))
plt.figure(1)
for leaf in leaves:
    I = tree0.get_box_inds(leaf)
    plt.scatter(XXc[I,1],XXc[I,2],label=str(leaf))
plt.legend()
plt.figure(2)
for leaf in leaves:
    I = tree0.get_box_inds(leaf)
    plt.scatter(XXr[I,1],XXr[I,2],label=str(leaf))
plt.legend()
plt.figure(3)
for leaf in leaves:
    I = tree0.get_box_inds(leaf)
    plt.scatter(XXl[I,1],XXl[I,2],label=str(leaf))
plt.legend()
plt.show()