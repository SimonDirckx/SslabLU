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
import matAssembly.HBS.HBSnew as HBSnew
import solver.solver as solver
import scipy.linalg as sclinalg
from matplotlib.colors import ListedColormap

import matAssembly.HBS.ULVsparse as ULVsparse
import time


cmap = ListedColormap([
    (0, 0, 0, 0),   
    (0, 0, 0, 1)    
])

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


Lx = 1/2
Ly = 1
Lz = 1
kh = 2.
ky = 2
kz = 2
kx = np.sqrt(ky**2+kz**2)
def  c11(p):
    return torch.ones_like(p[:,0])
def  c22(p):
    return torch.ones_like(p[:,1])
def  c33(p):
    return torch.ones_like(p[:,2])
def  c(p):
    return kh*torch.ones_like(p[:,1])
def  bc(p):
    return np.sin(np.pi*ky*p[:,1])*np.sin(np.pi*kz*p[:,2])*np.sinh(kx*np.pi*(Lx-p[:,0]))/np.sinh(kx*np.pi*Lx)
OP = pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33)

cx = Lx/2
bnds = np.array([[0,0,0],[Lx,Ly,Lz]])
Om = hpsaltGeom.BoxGeometry(bnds)
nby = 8
nbz = 8
nbx = 4
ax = .5*(bnds[1,0]/nbx)
ay = .5*(bnds[1,1]/nby)
az = .5*(bnds[1,2]/nbz)

px=7
py=7
pz=7


Sii,Sib,XYtot,Ii,Ib = SOMS.SOMS_solver(px,py,pz,nbx,nby,nbz,Lx,Ly,Lz,0,0)


XXi = XYtot[Ii,:]
XXb = XYtot[Ib,:]


Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0))[0]

XXi = XYtot[Ii,:]
XXb = XYtot[Ib,:]



Jc = np.where(np.abs(XXi[:,0]-cx)<1e-14)[0]
Jl = np.where((XXb[:,0]==0))[0]

Aib = Sib
SS = -(np.linalg.solve(Sii,Sib[:,Jl]))[Jc,:]
XXl = XXb[Jl,:]
XXc = XXi[Jc,:]
XXb2d = XXb[Jl,1:3]

tree0 =  tree.BalancedTree(XXl,(py-1)*(pz-1))

perm = []

for leaf in tree0.get_leaves():
    perm += tree0.get_box_inds(leaf).tolist()

Sp = SS[perm,:][:,perm]


k = 2*(py-1)*(pz-1)
nl = (py-1)*(pz-1)

HBSmat = HBSnew.HBSMAT(SS,tree0)
tic = time.time()
HBSmat.construct(k)
toc= time.time()
print("HBS construct time = ",toc-tic)
print("Block solve time = ",HBSmat.blockSolveTime)
print("Null time = ",HBSmat.nullTime)
print("set-up time = ",HBSmat.setupTime)
print("D time = ",HBSmat.DTime)


v = np.random.standard_normal(size=(SS.shape[1],))
u = SS@v
utest = HBSmat.matvec(v)
print("matvec err = ",np.linalg.norm(u-utest)/np.linalg.norm(u))
v = np.random.standard_normal(size=(SS.shape[0],))
u = SS.T@v
utest = HBSmat.matvec(v,mode='T')
print("matvecT err = ",np.linalg.norm(u-utest)/np.linalg.norm(u))


Umats = []
Vmats = []
Dmats = []

U0 = HBSmat.Umats[0]
V0 = HBSmat.Vmats[0]
D0 = HBSmat.Dmats[0]

Nb = len(tree0.get_leaves())

n = U0.shape[0]//Nb
k0 = U0.shape[1]
Umat = np.zeros(shape = (U0.shape[0],Nb*k0))
Vmat = np.zeros(shape = (V0.shape[0],Nb*k0))
Dmat = np.zeros(shape = (Nb*k0,Nb*k0))
for i in range(Nb):
    Umat[i*n:(i+1)*n,:][:,i*k0:(i+1)*k0] = U0[i*n:(i+1)*n,:]
    Vmat[i*n:(i+1)*n,:][:,i*k0:(i+1)*k0] = V0[i*n:(i+1)*n,:]
    Dmat[i*k0:(i+1)*k0,:][:,i*k0:(i+1)*k0] = D0[i*k0:(i+1)*k0,:]

Umats+=[Umat]
Vmats+=[Vmat]
Dmats+=[Dmat]

################################

U1 = HBSmat.Umats[1]
V1 = HBSmat.Vmats[1]
D1 = HBSmat.Dmats[1]

Nb = len(tree0.get_boxes_level(2))
n = U1.shape[0]//Nb
k = U1.shape[1]
Umat = np.zeros(shape = (U1.shape[0],Nb*k))
Vmat = np.zeros(shape = (V1.shape[0],Nb*k))
Dmat = np.zeros(shape = (Nb*n,Nb*n))
for i in range(Nb):
    Umat[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = U1[i*n:(i+1)*n,:]
    Vmat[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = V1[i*n:(i+1)*n,:]
    Dmat[i*n:(i+1)*n,:][:,i*n:(i+1)*n] = D1[i*n:(i+1)*n,:]

Umats+=[Umat]
Vmats+=[Vmat]
Dmats+=[Dmat]

############################
U2 = HBSmat.Umats[2]
V2 = HBSmat.Vmats[2]
D2 = HBSmat.Dmats[2]

Nb = len(tree0.get_boxes_level(1))
n = U2.shape[0]//Nb
k = U2.shape[1]
Umat = np.zeros(shape = (U2.shape[0],Nb*k))
Vmat = np.zeros(shape = (V2.shape[0],Nb*k))
Dmat = np.zeros(shape = (Nb*n,Nb*n))
for i in range(Nb):
    Umat[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = U2[i*n:(i+1)*n,:]
    Vmat[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = V2[i*n:(i+1)*n,:]
    Dmat[i*n:(i+1)*n,:][:,i*n:(i+1)*n] = D2[i*n:(i+1)*n,:]

Umats+=[Umat]
Vmats+=[Vmat]
Dmats+=[Dmat]

############################
D3 = HBSmat.Dmats[3]

Nb = len(tree0.get_boxes_level(0))
n = D3.shape[1]
Dmat = np.zeros(shape = (Nb*n,Nb*n))
for i in range(Nb):
    Dmat[i*n:(i+1)*n,:][:,i*n:(i+1)*n] = D3[i*n:(i+1)*n,:]


Dmats+=[Dmat]


SHBS = Dmats[-1]

for lvl in range(len(Umats)-1,-1,-1):
    SHBS = Umats[lvl]@SHBS@Vmats[lvl].T+Dmats[lvl]


print(" SHBS err ",np.linalg.norm(SHBS-Sp)/np.linalg.norm(Sp))
SHBSperm = np.zeros(shape=SHBS.shape)
SHBSperm[:,perm] = SHBS
SHBSperm[perm,:] = SHBSperm



v = np.random.standard_normal(size=(SS.shape[1],))
u = SHBSperm@v
utest = HBSmat.matvec(v)
print("matvec err = ",np.linalg.norm(u-utest)/np.linalg.norm(u))


