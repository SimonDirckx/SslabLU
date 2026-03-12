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
N = 2**16
nl = 16*16
Nleaves = N//nl
k = 32
k0 = min(nl,k)
L = (int)(np.log2(Nleaves))//2 + 1


Nb = Nleaves
Nbvec = [Nb]
Utot1_sparse = np.zeros(shape = (Nb*nl,k0))
Vtot1_sparse = np.zeros(shape = (Nb*nl,k0))
Dtot1_sparse = np.zeros(shape = (Nb*nl,nl))

for i in range(Nb):
    Dtot1_sparse[i*nl:(i+1)*nl,:] = np.random.standard_normal(size = (nl,nl))#Dtot1[i*nl:(i+1)*nl,:][:,i*nl:(i+1)*nl].copy()
    Utot1_sparse[i*nl:(i+1)*nl,:] = sclinalg.orth(np.random.standard_normal(size = (nl,k0)))#Utot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()
    Vtot1_sparse[i*nl:(i+1)*nl,:] = sclinalg.orth(np.random.standard_normal(size = (nl,k0)))#Vtot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()
Umats = [Utot1_sparse]
Vmats = [Vtot1_sparse]
Dmats = [Dtot1_sparse]


for ind in range(1,L):
    Nb      = Nb//4
    Nbvec+=[Nb]
    if ind==1:
        n = 4*k0
    else:
        n = 4*k
    Utot_sparse = np.zeros(shape = (Nb*n,k))
    Vtot_sparse = np.zeros(shape = (Nb*n,k))
    Dtot_sparse = np.zeros(shape = (Nb*n,n))

    for i in range(Nb):
        Dtot_sparse[i*n:(i+1)*n,:] = np.random.standard_normal(size = (n,n))#Dtot2[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()
        Utot_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Utot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
        Vtot_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Vtot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
    if ind<L-1:
        Umats += [Utot_sparse]
        Vmats += [Vtot_sparse]
    Dmats += [Dtot_sparse]


ticULV = time.time()
Qtot,Rtot,Wtot,NNvec,NNQvec,NNRvec,NNWvec = ULVsparse.compute_ULV(Umats,Dmats,Vmats,Nbvec)
tocULV = time.time()


SHBS = HBSnew.HBSMAT()
SHBS.set_mats(Umats,Dmats,Vmats,Nbvec)
x= np.random.standard_normal(size=(SHBS.shape[1],2))
b = SHBS.matvec(x)


ticSolveULV = time.time()
rhs = ULVsparse.apply_cbd(Qtot,b,Nbvec,NNvec,NNQvec,mode='T')
uhat = ULVsparse.solve_R(Rtot,rhs,Nbvec,NNvec,NNRvec)
u = ULVsparse.apply_cbd(Wtot,uhat,Nbvec,NNvec,NNQvec)
tocSolveULV = time.time()



print("solve err = ",np.linalg.norm(u-x)/np.linalg.norm(x))
print("solve ULV time = ", tocSolveULV-ticSolveULV)
print("ULV fact. time = ", tocULV-ticULV)
print("Ndofs = ", SHBS.shape[0])




