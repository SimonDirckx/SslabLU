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
import stree


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

px=11
py=11
pz=11



solver_S = hpsalt.Domain_Driver(Om, OP, 0, a=np.array([ax,ay,az]), p=np.array([px+1,py+1,pz+1]), d=3)
solver_S.build("reduced_cpu", "MUMPS", verbose=False)

XX = solver_S.XX
XXfull = solver_S._XXfull
Ii = solver_S._Ji
Jb = solver_S._Jx
Sib = solver_S.Aix
solver_S.setup_solver_Aii()
solver_ii = solver_S.solver_Aii

XXi = solver_S.XX[Ii,:]
XXb = solver_S.XX[Jb,:]

Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0))[0]



def smatmat(v,I,J,transpose=False):        
    if (v.ndim == 1):
        v_tmp = v[:,np.newaxis]
    else:
        v_tmp = v

    if (not transpose):
        result = (solver_ii@(Sib[:,J]@v_tmp))[I,:]
    else:
        result      = np.zeros(shape=(len(Ii),v_tmp.shape[1]))
        result[I,:] = v_tmp
        result      = Sib[:,J].T @ (solver_ii.T@(result))
    if (v.ndim == 1):
        result = result.flatten()
    return result



SS = LinearOperator(shape=(len(Jc),len(Jl)),\
        matvec = lambda v:smatmat(v,Jc,Jl), rmatvec = lambda v:smatmat(v,Jc,Jl,transpose=True),\
        matmat = lambda v:smatmat(v,Jc,Jl), rmatmat = lambda v:smatmat(v,Jc,Jl,transpose=True))

XXl = XXb[Jl,:].detach().cpu().numpy()
XXc = XXi[Jc,:].detach().cpu().numpy()
k = 2*(py-1)*(pz-1)
nl = (py-1)*(pz-1)

quad=False
if quad:
    tree0 =  tree.BalancedTree(XXl,(py-1)*(pz-1))
else:
    tree0  = stree.stree(XXl,nl)
#print("L = ",tree_test.nlevels)


HBSmat = HBSnew.HBSMAT(SS,tree0,quad)
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



HBSmat.compute_ULV()

x = np.random.standard_normal(size = (SS.shape[0],))
b = SS@x
xhat = HBSmat.solve(b)
print("solve err = ",np.linalg.norm(x-xhat)/np.linalg.norm(x))
print("solve res = ",np.linalg.norm(b-SS@xhat)/np.linalg.norm(b))

print("cond(S) = ",np.linalg.cond(SS@np.identity(SS.shape[1])))

'''

##################################################################################

Umats = []
Vmats = []
Dmats = []
L = tree0.nlevels

for lvl in range(L-1):
    
    Ul = HBSmat.Umats[lvl]
    Vl = HBSmat.Vmats[lvl]
    Dl = HBSmat.Dmats[lvl]
    Nb = len(tree0.get_boxes_level(L-lvl-1))
    print("lvl//Nb = ",lvl,"//",Nb)
    n = Ul.shape[0]//Nb
    k = Ul.shape[1]
    Umat = np.zeros(shape = (Ul.shape[0],Nb*k))
    Vmat = np.zeros(shape = (Vl.shape[0],Nb*k))
    Dmat = np.zeros(shape = (Nb*n,Nb*n))
    for i in range(Nb):
        Umat[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = Ul[i*n:(i+1)*n,:]
        Vmat[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = Vl[i*n:(i+1)*n,:]
        Dmat[i*n:(i+1)*n,:][:,i*n:(i+1)*n] = Dl[i*n:(i+1)*n,:]
    Umats+=[Umat]
    Vmats+=[Vmat]
    Dmats+=[Dmat]

D0 = HBSmat.Dmats[L-1]
Nb = len(tree0.get_boxes_level(0))
n = D0.shape[1]
Dmat = np.zeros(shape = (Nb*n,Nb*n))
for i in range(Nb):
    Dmat[i*n:(i+1)*n,:][:,i*n:(i+1)*n] = D0[i*n:(i+1)*n,:]

Dmats+=[Dmat]

SHBS = Dmats[-1]

for lvl in range(len(Umats)-1,-1,-1):
    SHBS = Umats[lvl]@SHBS@(Vmats[lvl].T)+Dmats[lvl]


perm = []

for leaf in tree0.get_leaves():
    perm += tree0.get_box_inds(leaf).tolist()

SHBSperm = np.zeros(shape=SHBS.shape)
SHBSperm[:,perm] = SHBS.copy()
SHBSperm[perm,:] = SHBSperm
#print(" SS err ",np.linalg.norm(SHBSperm-SS,ord=2)/np.linalg.norm(SS,ord=2))


v = np.random.standard_normal(size=(SS.shape[1],))
u = SHBSperm@v
utest = HBSmat.matvec(v)
print("matvec err = ",np.linalg.norm(u-utest)/np.linalg.norm(u))


HBSmat_test = HBSnew.HBSMAT(SHBSperm,tree0)
HBSmat_test.construct(k)

v = np.random.standard_normal(size=(SS.shape[1],))
utest = HBSmat_test.matvec(v)
u = SHBSperm@v
print("rec matvec HBS err = ",np.linalg.norm(u-utest)/np.linalg.norm(u))

v = np.random.standard_normal(size=(SS.shape[1],))
utest = HBSmat_test.matvec(v,mode='T')
u = SHBSperm.T@v
print("rec matvecT HBS err = ",np.linalg.norm(u-utest)/np.linalg.norm(u))



x = np.random.standard_normal(size = (SS.shape[0],))
b = SHBSperm@x
HBSmat_test.compute_ULV()
xhat = HBSmat_test.solve(b)
print("solve err = ",np.linalg.norm(x-xhat)/np.linalg.norm(x))
'''