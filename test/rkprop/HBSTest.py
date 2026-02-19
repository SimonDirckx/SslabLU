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
Lx = 1/2
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

px=11
py=11
pz=11

#solver_hps = hpsalt.Domain_Driver(Om, OP, 0., np.array([ax,ay,az]), [px+1,py+1,pz+1], 3)
#solver_hps.build("reduced_cpu", "MUMPS",verbose=False)

#XX = (solver_hps.XX).cpu().detach().numpy()

#Jb = solver_hps._Jx
#Ji = solver_hps.Ji

#print("Ji size = ",len(Ji))
#print("Jb size = ",len(Jb))

#XXi = XX[Ji,:]
#XXb = XX[Jb,:]

Sii,Sib,XYtot,Ii,Ib = SOMS.SOMS_solver(px,py,pz,nbx,nby,nbz,Lx,Ly,Lz,0,0)


XXi = XYtot[Ii,:]
XXb = XYtot[Ib,:]


Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0))[0]

XXi = XYtot[Ii,:]
XXb = XYtot[Ib,:]



Jc = np.where(np.abs(XXi[:,0]-cx)<1e-14)[0]
Jl = np.where((XXb[:,0]==0))[0]
print("Jc size = ",len(Jc))
print("Jl size = ",len(Jl))
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

XXlp = XXl[perm,:]
XXcp = XXc[perm,:]

tree_perm =  tree.BalancedTree(XXlp,(py-1)*(pz-1))


leaves = tree_perm.get_leaves()
N = Sp.shape[0]
Nleaves = len(leaves)
#print("Nleaves = ",Nleaves)
k = 150
k0 = (py-1)*(pz-1)
#k=75
#print("rank = ",k)
nl = (py-1)*(pz-1)
dataHBS=0

Utot = np.zeros(shape=(N,Nleaves*k0))
Vtot = np.zeros(shape=(N,Nleaves*k0))
Dtot = np.zeros(shape=(N,N))

#print("tot MATS made")
def compute_col_space(M,k):
    [Q,_,_] = sclinalg.qr(M,mode='economic',pivoting=True)
    k0 = min(k,Q.shape[1])
    return Q[:,:k0]
for leaf_ind in range(len(leaves)):
    leaf = leaves[leaf_ind]
    print("leaf = ",leaf)
    inds = np.sort(tree_perm.get_box_inds(leaf))
    indsc = [i for i in range(N) if i not in inds]
    U = compute_col_space(Sp[inds,:][:,indsc],k)
    V = compute_col_space(Sp[indsc,:][:,inds].T,k)
    k0 = U.shape[1]
    errU = np.linalg.norm(Sp[inds,:][:,indsc] - U@(U.T@Sp[inds,:][:,indsc]))/np.linalg.norm(Sp[inds,:][:,indsc])
    errV = np.linalg.norm(Sp[indsc,:][:,inds].T - V@(V.T@Sp[indsc,:][:,inds].T))/np.linalg.norm(Sp[indsc,:][:,inds])
    Utot[leaf_ind*nl:(leaf_ind+1)*nl,:][:,leaf_ind*k0:(leaf_ind+1)*k0] = U
    Vtot[leaf_ind*nl:(leaf_ind+1)*nl,:][:,leaf_ind*k0:(leaf_ind+1)*k0] = V
    D = Sp[inds,:][:,inds]- U@(U.T@Sp[inds,:][:,inds]@V)@V.T
    Dtot[leaf_ind*nl:(leaf_ind+1)*nl,:][:,leaf_ind*nl:(leaf_ind+1)*nl] = D
    dataHBS += U.nbytes+V.nbytes+D.nbytes
    print("errU,errV = ",errU," , ",errV)
Ktot = Utot.T@Sp@Vtot

nrmSp = np.linalg.norm(Sp)

print("nrmSp = ", nrmSp)
print("errSp = ", np.linalg.norm(Sp-Utot@Ktot@Vtot.T-Dtot))

boxes = tree_perm.get_boxes_level(tree_perm.nlevels-2)
Nb = len(boxes)
Utot1 = np.zeros(shape=(Ktot.shape[0],Nb*k))
Vtot1 = np.zeros(shape=(Ktot.shape[0],Nb*k))
Dtot1 = np.zeros(shape=(Ktot.shape[0],Ktot.shape[0]))


print("Nb // Nl/4 = ",Nb,"//",Nleaves//4)
for ind_box in range(Nb):
    inds = np.arange(4*ind_box*k0,4*(ind_box+1)*k0)
    print("n2,i1 = ",Ktot.shape," , ",inds[-1])
    indsc = [i for i in range(Ktot.shape[1]) if i not in inds]
    K_row = Ktot[inds,:][:,indsc]
    K_col = Ktot[indsc,:][:,inds]
    U = compute_col_space(K_row,k)
    V = compute_col_space(K_col.T,k)
    errU = np.linalg.norm(K_row - U@(U.T@K_row))
    errV = np.linalg.norm(K_col.T - V@(V.T@K_col.T))
    Utot1[4*ind_box*k0:4*(ind_box+1)*k0,:][:,ind_box*k:(ind_box+1)*k] = U
    Vtot1[4*ind_box*k0:4*(ind_box+1)*k0,:][:,ind_box*k:(ind_box+1)*k] = V
    D = Ktot[inds,:][:,inds]- U@(U.T@Ktot[inds,:][:,inds]@V)@V.T
    Dtot1[4*ind_box*k0:4*(ind_box+1)*k0,:][:,4*ind_box*k0:4*(ind_box+1)*k0] = D 
    dataHBS += U.nbytes+V.nbytes+D.nbytes
    print("errU,errV = ",errU," , ",errV)
Ktot1 = Utot1.T@Ktot@Vtot1

print("errK = ", np.linalg.norm(Ktot-Utot1@Ktot1@Vtot1.T-Dtot1))
Khat = Utot1@Ktot1@Vtot1.T+Dtot1
Shat = Utot@Khat@Vtot.T+Dtot
print("errTot = ", np.linalg.norm(Sp-Shat))

boxes = tree_perm.get_boxes_level(tree_perm.nlevels-3)
Nb = len(boxes)
Utot2 = np.zeros(shape=(Ktot1.shape[0],Nb*k))
Vtot2 = np.zeros(shape=(Ktot1.shape[0],Nb*k))
Dtot2 = np.zeros(shape=(Ktot1.shape[0],Ktot1.shape[0]))

print("Nb = ",Nb)
print("Ktot1 shape = ",Ktot1.shape)

for ind_box in range(Nb):
    inds = np.arange(4*ind_box*k,4*(ind_box+1)*k)
    indsc = [i for i in range(Ktot1.shape[1]) if i not in inds]
    K_row = Ktot1[inds,:][:,indsc]
    K_col = Ktot1[indsc,:][:,inds]
    U = compute_col_space(K_row,k)
    V = compute_col_space(K_col.T,k)
    errU = np.linalg.norm(K_row - U@(U.T@K_row))
    errV = np.linalg.norm(K_col.T - V@(V.T@K_col.T))
    Utot2[4*ind_box*k:4*(ind_box+1)*k,:][:,ind_box*k:(ind_box+1)*k] = U
    Vtot2[4*ind_box*k:4*(ind_box+1)*k,:][:,ind_box*k:(ind_box+1)*k] = V
    D = Ktot1[inds,:][:,inds]- U@(U.T@Ktot1[inds,:][:,inds]@V)@V.T
    Dtot2[4*ind_box*k:4*(ind_box+1)*k,:][:,4*ind_box*k:4*(ind_box+1)*k] = D 
    dataHBS += U.nbytes+V.nbytes+D.nbytes
    print("errU,errV = ",errU," , ",errV)
Ktot2 = Utot2.T@Ktot1@Vtot2
print("Ktot2.shape = ",Ktot2.shape)
#dataHBS+=Ktot2.nbytes

Khat = Utot2@Ktot2@Vtot2.T+Dtot2
print("errK = ", np.linalg.norm(Ktot1-Khat))
Khat = Utot1@Khat@Vtot1.T+Dtot1
Shat = Utot@Khat@Vtot.T+Dtot


print("errTot = ", np.linalg.norm(Sp-Shat))
print("compression = ",dataHBS/Sp.nbytes)
