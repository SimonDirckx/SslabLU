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
Lx = 1/8
Ly = 1
Lz = 1
kh = 20
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
py=7
px=7
pz=7

solver_hps = hpsalt.Domain_Driver(Om, HH, kh, np.array([ax,ay,az]), [px+1,py+1,pz+1], 3)
solver_hps.build("reduced_cpu", "MUMPS",verbose=False)

XX = solver_hps.XX
XXfull = solver_hps.XXfull

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

c0,L0 = compute_c0_L0(XXb[Jl,:])
tree0 =  tree.BalancedTree(XXb[Jl,:],(py-1)*(pz-1),c0,L0)

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

for ind in range(len(tree0.get_leaves())):
    inds = np.append(np.arange(0,(ind-1)*(py-1)*(pz-1)),np.arange((ind+1)*(py-1)*(pz-1),Sp.shape[1]))
    Ssub = Sp[:,inds][ind*(py-1)*(pz-1):(ind+1)*(py-1)*(pz-1),:]
    [_,s,_] = np.linalg.svd(Ssub)
    plt.figure(1)
    plt.semilogy(s)
    plt.show()
