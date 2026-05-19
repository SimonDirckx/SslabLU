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
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator

kh = 25.
def c11(p):
    return torch.ones_like(p[:,0])
def c22(p):
    return torch.ones_like(p[:,1])
def c33(p):
    return torch.ones_like(p[:,2])
def c(p):
    return -kh*kh*torch.ones_like(p[:,0])
Helm=pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)

px = 6; py = 6; pz = 6
dx = .1; dy = 1; dz = 1
nbx = 2; nby = 8; nbz = 8
ax = .5*dx/nbx ; ay = .5*dy/nby ; az = .5*dz/nbz
Om = np.array([[0, 0, 0],[2*dx, dy, dz]])

opts = solverWrap.solverOptions('hpsalt',[px+2,py+2,pz+2],np.array([ax,ay,az]))
solver = solverWrap.solverWrapper(opts)
solver.construct(Om,Helm)
assembler = mA.rkHMatAssembler(px*pz,72,ndim=3)
XXb = solver.XX[solver.Ib,:]
XXi = solver.XX[solver.Ii,:]
Il = np.where(XXb[:,0]==0)[0]
Ic = np.where(XXi[:,0]==dx)[0]
XXl = XXb[Il,:]
XXc = XXi[Ic,:]



A_solver = solver.solver_ii
def smatmat(v,I,J,transpose=False):
    
    if (v.ndim == 1):
    
        v_tmp = v[:,np.newaxis]
    else:
        v_tmp = v

    if (not transpose):
        result = (A_solver@(solver.Aib[:,J]@v_tmp))[I,:]
    else:
        result      = np.zeros(shape=(len(solver.Ii),v.shape[1]))
        result[I,:] = v_tmp
        result      = solver.Aib[:,J].T @ (A_solver.T@(result))
    if (v.ndim == 1):
        result = result.flatten()
    return result

Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
    matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
    matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))

st_l = solverWrap.stMap(Linop_l,XXb[Il,:],XXi[Ic,:],solver.solver_ii.shape[0],solver.solver_ii.shape[1])


sH = assembler.assemble(st_l)

B = np.random.standard_normal(size=(sH.shape[1],50))
print("Hmat err = ",np.linalg.norm(st_l.A@B-sH@B)/np.linalg.norm(st_l.A@B))
print("Hmat data : ",sH.nbytes)
print("dense data = ",st_l.A.shape[0]*st_l.A.shape[1]*8)
perm = sH.perm
SDense = st_l.A@np.identity(st_l.A.shape[0])

SDense_perm = SDense.copy()
SDense_perm = SDense_perm[perm,:][:,perm]
tree = assembler.matOpts.tree
start = 0
for node in tree.get_boxes_level(tree.nlevels-2):
    inds = tree.get_box_inds(node)
    indsc = [i for i in range(st_l.A.shape[0]) if not i in inds]
    block = np.concatenate([SDense[inds,:][:,:inds[0]],SDense[inds,:][:,inds[-1]:]],axis=1)
    rk = np.linalg.matrix_rank(block,rtol = 1e-8)
    print("rank of block = ",rk)
    print("err perm = ",np.linalg.norm(SDense_perm[start:start+len(inds),:]-SDense[inds,:][:,perm]))
    start = start+len(inds)
    plt.figure(1)
    plt.scatter(XXl[inds,1],XXl[inds,2],label = 'tau')
    plt.legend()
    plt.figure(2)
    plt.scatter(XXc[indsc,1],XXc[indsc,2],label = 'tauc')
    plt.legend()
    plt.show()

start = 0
plt.figure(1)
for i in range(nby*nbz):
    #plt.scatter(XXl[perm[start:start+py*pz],1],XXl[perm[start:start+py*pz],2],label = i)
    plt.scatter(XXl[start:start+py*pz,1],XXl[start:start+py*pz,2],label = i)
    start+=py*pz
plt.legend()
plt.show()
