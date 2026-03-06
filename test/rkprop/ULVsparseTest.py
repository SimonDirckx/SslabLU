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
nbx = 2
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
xprime = np.random.standard_normal(size=(Sp.shape[1],))
x = xprime.copy()
xprime[perm] = xprime
y = SS@xprime
y = y[perm]
yhat = Sp@x
print("perm err = ",np.linalg.norm(y-yhat))


#xprime = np.random.standard_normal(size=(Sp.shape[1],))
#x = np.zeros(shape = xprime.shape)
#x[perm] = xprime
#y = SS.T@x
#yprime = y[perm]
#yprimehat = Sp.T@xprime
#print("perm err = ",np.linalg.norm(yprime-yprimehat))





XXlp = XXl[perm,:]
XXcp = XXc[perm,:]

tree_perm =  tree.BalancedTree(XXlp,(py-1)*(pz-1))
leaves = tree_perm.get_leaves()
N = Sp.shape[0]
Nleaves = len(leaves)
k = (py-1)*(pz-1)//2
k0 = (py-1)*(pz-1)
nl = (py-1)*(pz-1)




dataHBS=0

Utot1 = np.zeros(shape=(N,Nleaves*k0))
Vtot1 = np.zeros(shape=(N,Nleaves*k0))
Dtot1 = np.zeros(shape=(N,N))

def compute_col_space(M,k):
    Om = np.random.standard_normal(size=(M.shape[1],k+10))
    [Q,_,_] = sclinalg.qr(M@Om,mode='economic',pivoting=True)
    k0 = min(k,Q.shape[1])
    return Q[:,:k0]
for leaf_ind in range(len(leaves)):
    leaf = leaves[leaf_ind]
    inds = np.sort(tree_perm.get_box_inds(leaf))
    indsc = [i for i in range(N) if i not in inds]
    U = compute_col_space(Sp[inds,:][:,indsc],k)
    V = compute_col_space(Sp[indsc,:][:,inds].T,k)
    k0 = U.shape[1]
    Utot1[leaf_ind*nl:(leaf_ind+1)*nl,:][:,leaf_ind*k0:(leaf_ind+1)*k0] = U
    Vtot1[leaf_ind*nl:(leaf_ind+1)*nl,:][:,leaf_ind*k0:(leaf_ind+1)*k0] = V
    D = Sp[inds,:][:,inds]- U@(U.T@Sp[inds,:][:,inds]@V)@V.T
    Dtot1[leaf_ind*nl:(leaf_ind+1)*nl,:][:,leaf_ind*nl:(leaf_ind+1)*nl] = D
    dataHBS += U.nbytes+V.nbytes+D.nbytes
Ktot1 = Utot1.T@Sp@Vtot1

nrmSp = np.linalg.norm(Sp)

boxes = tree_perm.get_boxes_level(tree_perm.nlevels-2)
Nb = len(boxes)
Utot2 = np.zeros(shape=(Ktot1.shape[0],Nb*k))
Vtot2 = np.zeros(shape=(Ktot1.shape[0],Nb*k))
Dtot2 = np.zeros(shape=(Ktot1.shape[0],Ktot1.shape[0]))


for ind_box in range(Nb):
    inds = np.arange(4*ind_box*k0,4*(ind_box+1)*k0)
    indsc = [i for i in range(Ktot1.shape[1]) if i not in inds]
    K_row = Ktot1[inds,:][:,indsc]
    K_col = Ktot1[indsc,:][:,inds]
    U = compute_col_space(K_row,k)
    V = compute_col_space(K_col.T,k)
    Utot2[4*ind_box*k0:4*(ind_box+1)*k0,:][:,ind_box*k:(ind_box+1)*k] = U
    Vtot2[4*ind_box*k0:4*(ind_box+1)*k0,:][:,ind_box*k:(ind_box+1)*k] = V
    D = Ktot1[inds,:][:,inds]- U@(U.T@Ktot1[inds,:][:,inds]@V)@V.T
    Dtot2[4*ind_box*k0:4*(ind_box+1)*k0,:][:,4*ind_box*k0:4*(ind_box+1)*k0] = D 
    dataHBS += U.nbytes+V.nbytes+D.nbytes


Ktot2 = Utot2.T@Ktot1@Vtot2

boxes = tree_perm.get_boxes_level(tree_perm.nlevels-3)
Nb = len(boxes)
Utot3 = np.zeros(shape=(Ktot2.shape[0],Nb*k))
Vtot3 = np.zeros(shape=(Ktot2.shape[0],Nb*k))
Dtot3 = np.zeros(shape=(Ktot2.shape[0],Ktot2.shape[0]))


for ind_box in range(Nb):
    inds = np.arange(4*ind_box*k,4*(ind_box+1)*k)
    indsc = [i for i in range(Ktot2.shape[1]) if i not in inds]
    K_row = Ktot2[inds,:][:,indsc]
    K_col = Ktot2[indsc,:][:,inds]
    U = compute_col_space(K_row,k)
    V = compute_col_space(K_col.T,k)
    errU = np.linalg.norm(K_row - U@(U.T@K_row))
    errV = np.linalg.norm(K_col.T - V@(V.T@K_col.T))
    Utot3[4*ind_box*k:4*(ind_box+1)*k,:][:,ind_box*k:(ind_box+1)*k] = U
    Vtot3[4*ind_box*k:4*(ind_box+1)*k,:][:,ind_box*k:(ind_box+1)*k] = V
    D = Ktot2[inds,:][:,inds]- U@(U.T@Ktot2[inds,:][:,inds]@V)@V.T
    Dtot3[4*ind_box*k:4*(ind_box+1)*k,:][:,4*ind_box*k:4*(ind_box+1)*k] = D 
    dataHBS += U.nbytes+V.nbytes+D.nbytes
Ktot3 = Utot3.T@Ktot2@Vtot3

Dtot4 = Ktot3
dataHBS+=Dtot4.nbytes


Khat3 = Dtot4
Khat2 = Utot3@(Khat3@Vtot3.T)+Dtot3
Khat1 = Utot2@(Khat2@Vtot2.T)+Dtot2
SHBS = Utot1@(Khat1@Vtot1.T)+Dtot1

print("SHBS err = ",np.linalg.norm(Sp-SHBS))

# accumul sparse:

#lvl1
boxes = tree_perm.get_boxes_level(tree_perm.nlevels-1)
Nb = len(boxes)

Utot1_sparse = np.zeros(shape = (Nb*nl,k0))
Vtot1_sparse = np.zeros(shape = (Nb*nl,k0))
Dtot1_sparse = np.zeros(shape = (Nb*nl,nl))



for i in range(Nb):
    Dtot1_sparse[i*nl:(i+1)*nl,:] = Dtot1[i*nl:(i+1)*nl,:][:,i*nl:(i+1)*nl].copy()
    Utot1_sparse[i*nl:(i+1)*nl,:] = Utot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()
    Vtot1_sparse[i*nl:(i+1)*nl,:] = Vtot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()


#lvl2
boxes   = tree_perm.get_boxes_level(tree_perm.nlevels-2)
Nb      = len(boxes)
n       = 4*k0
Utot2_sparse = np.zeros(shape = (Nb*n,k))
Vtot2_sparse = np.zeros(shape = (Nb*n,k))
Dtot2_sparse = np.zeros(shape = (Nb*n,n))

for i in range(Nb):
    Dtot2_sparse[i*n:(i+1)*n,:] = Dtot2[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()
    Utot2_sparse[i*n:(i+1)*n,:] = Utot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
    Vtot2_sparse[i*n:(i+1)*n,:] = Vtot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()

#lvl3
boxes   = tree_perm.get_boxes_level(tree_perm.nlevels-3)
Nb      = len(boxes)
n       = 4*k
Utot3_sparse = np.zeros(shape = (Nb*n,k))
Vtot3_sparse = np.zeros(shape = (Nb*n,k))
Dtot3_sparse = np.zeros(shape = (Nb*n,n))
for i in range(Nb):
    Dtot3_sparse[i*n:(i+1)*n,:] = Dtot3[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()
    Utot3_sparse[i*n:(i+1)*n,:] = Utot3[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
    Vtot3_sparse[i*n:(i+1)*n,:] = Vtot3[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()

#lvl4
boxes   = tree_perm.get_boxes_level(tree_perm.nlevels-4)
Nb      = len(boxes)
n       = 4*k
Utot4_sparse = np.zeros(shape = (Nb*n,k))
Vtot4_sparse = np.zeros(shape = (Nb*n,k))
Dtot4_sparse = np.zeros(shape = (Nb*n,n))
for i in range(Nb):
    Dtot4_sparse[i*n:(i+1)*n,:] = Dtot4[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()

Utot4_sparse = np.identity(n)
Vtot4_sparse = 1.

Dmats = [Dtot1_sparse,Dtot2_sparse,Dtot3_sparse,Dtot4_sparse]
Vmats = [Vtot1_sparse,Vtot2_sparse,Vtot3_sparse,Vtot4_sparse]
Umats = [Utot1_sparse,Utot2_sparse,Utot3_sparse,Utot3_sparse]
Nbvec = [len(tree_perm.get_boxes_level(tree_perm.nlevels-1)),len(tree_perm.get_boxes_level(tree_perm.nlevels-2)),\
         len(tree_perm.get_boxes_level(tree_perm.nlevels-3)),len(tree_perm.get_boxes_level(tree_perm.nlevels-4))]

ticULV = time.time()
Qtot,Rtot,Wtot,NNvec,NNQvec,NNRvec,NNWvec = ULVsparse.compute_ULV(Umats,Dmats,Vmats,Nbvec)
tocULV = time.time()

x= np.random.standard_normal(size=(SHBS.shape[1],2))
b = SHBS@x



ticSolve = time.time()
u = np.linalg.solve(SHBS,b)
tocSolve = time.time()

ticSolveULV = time.time()
rhs = ULVsparse.apply_cbd(Qtot,b,Nbvec,NNvec,NNQvec,mode='T')
uhat = ULVsparse.solve_R(Rtot,rhs,Nbvec,NNvec,NNRvec)
u = ULVsparse.apply_cbd(Wtot,uhat,Nbvec,NNvec,NNQvec)
tocSolveULV = time.time()

print("u err = ",np.linalg.norm(u-x)/np.linalg.norm(x))
print("solve time = ", tocSolve-ticSolve)
print("solve ULV time = ", tocSolveULV-ticSolveULV)
print("ULV fact. time = ", tocULV-ticULV)

Qtotmat = ULVsparse.apply_cbd(Qtot,np.identity(Qtot.shape[0]),Nbvec,NNvec,NNQvec,mode='T')
Wtotmat = ULVsparse.apply_cbd(Wtot,np.identity(Wtot.shape[0]),Nbvec,NNvec,NNQvec)

SS = Qtotmat@SHBS@Wtotmat
plt.figure(1)
plt.spy(Qtotmat.T,precision = 1e-8)
plt.figure(2)
plt.spy(Wtotmat,precision = 1e-8)
plt.figure(3)
plt.spy(SS,precision = 1e-8)
plt.show()


