from scipy.sparse.linalg   import LinearOperator
import numpy as np
from matAssembly.HBS import HBSTree as HBS
import solver.solver as solver
import time
import matplotlib.pyplot as plt
from matAssembly.HBS.simpleoctree import simpletree as tree
import torch

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsaltGeom
import SOMS3D as SOMS
import matAssembly.matAssembler as mA
import solver.solver as solver
import scipy.linalg as sclinalg
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

k = 2
Lx = 1/4
Ly = 1
Lz = 1
kh = 2.
def  c11(p):
    return torch.ones_like(p[:,0])
def  c22(p):
    return torch.ones_like(p[:,1])
def  c33(p):
    return torch.ones_like(p[:,2])
def  c(p):
    return kh*torch.ones_like(p[:,1])
HH = pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)

cx = Lx/2
bnds = np.array([[0,0,0],[Lx,Ly,Lz]])
Om = hpsaltGeom.BoxGeometry(bnds)
nby = 16
nbz = 16
nbx = 2
ax = .5*(bnds[1,0]/nbx)
ay = .5*(bnds[1,1]/nby)
az = .5*(bnds[1,2]/nbz)

#isotropic disc
py=8
px=8
pz=8

solver_hps = hpsalt.Domain_Driver(Om, HH, kh, np.array([ax,ay,az]), [px+1,py+1,pz+1], 3)
solver_hps.build("reduced_cpu", "MUMPS",verbose=False)

XX = (solver_hps.XX).cpu().detach().numpy()

Jb = solver_hps._Jx
Ji = solver_hps.Ji

print("Ji size = ",len(Ji))
print("Jb size = ",len(Jb))

XXi = XX[Ji,:]
XXb = XX[Jb,:]


Jc = np.where(np.abs(XXi[:,0]-cx)<1e-14)[0]
Jl = np.where((XXb[:,0]==0))[0]
print("Jc size = ",len(Jc))
print("Jl size = ",len(Jl))
Aib = np.array(solver_hps.Aix.todense())
SS = -(solver_hps.solver_Aii@Aib[:,Jl])[Jc,:]
XXl = XXb[Jl,:]
XXc = XXi[Jc,:]
XXb2d = XXb[Jl,1:3]

tree0 =  tree.BalancedTree(XXl,(py-1)*(pz-1))

perm = []

for leaf in tree0.get_leaves():
    perm += tree0.get_box_inds(leaf).tolist()

Sp = SS[perm,:][:,perm]
v = np.random.standard_normal(size=(Sp.shape[1],))
vperm = v[perm]
u = Sp@vperm
u[perm] = u
ucheck = SS@v
print("matvec err = ",np.linalg.norm(u-ucheck)/np.linalg.norm(ucheck))


# level 1 approx:
leaves = tree0.get_leaves()
N = SS.shape[0]
Nleaves = len(leaves)
print("Nleaves = ",Nleaves)
k = 2*(py-1)*(pz-1)
print("rank = ",k)
nl = (py-1)*(pz-1)

Utot = np.zeros(shape=(N,Nleaves*k))
Vtot = np.zeros(shape=(N,Nleaves*k))
Dtot = np.zeros(shape=(N,N))
Ktot = np.zeros(shape = (Nleaves*k,Nleaves*k))

print("tot MATS made")
def compute_col_space(M,k):
    [Q,_,_] = sclinalg.qr(M,mode='economic',pivoting=True)
    k0 = min(k,Q.shape[1])
    return Q[:,:k0]

for leaf in leaves:
    neigh_inds = tree0.get_neigh_inds(leaf)
    inds = tree0.get_box_inds(leaf)
    inds0 = np.append(inds,neigh_inds)
    indsc = [i for i in range(N) if i not in inds]
    indsc0 = [i for i in range(N) if i not in inds0]
    s = np.linalg.svd(SS[:,inds][indsc,:],compute_uv=False)
    s0 = np.linalg.svd(SS[:,inds][indsc0,:],compute_uv=False)
    rk = sum(s>s[0]*1e-6)
    rk0 = sum(s0>s0[0]*1e-6)
    print("=============")
    print("rk = ",rk)
    print("rk0 = ",rk0)
    print("=============")
