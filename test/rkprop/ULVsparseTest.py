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

nl = 32*32
k = 64
k0 = min(nl,k)
Nleaves = 64
N = Nleaves*nl

Nb = Nleaves
Nbvec = [Nb]
Utot1_sparse = np.zeros(shape = (Nb*nl,k0))
Vtot1_sparse = np.zeros(shape = (Nb*nl,k0))
Dtot1_sparse = np.zeros(shape = (Nb*nl,nl))



for i in range(Nb):
    Dtot1_sparse[i*nl:(i+1)*nl,:] = np.random.standard_normal(size = (nl,nl))#Dtot1[i*nl:(i+1)*nl,:][:,i*nl:(i+1)*nl].copy()
    Utot1_sparse[i*nl:(i+1)*nl,:] = sclinalg.orth(np.random.standard_normal(size = (nl,k0)))#Utot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()
    Vtot1_sparse[i*nl:(i+1)*nl,:] = sclinalg.orth(np.random.standard_normal(size = (nl,k0)))#Vtot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()


#lvl2
Nb      = Nb//4
Nbvec+=[Nb]
n       = 4*k0
Utot2_sparse = np.zeros(shape = (Nb*n,k))
Vtot2_sparse = np.zeros(shape = (Nb*n,k))
Dtot2_sparse = np.zeros(shape = (Nb*n,n))

for i in range(Nb):
    Dtot2_sparse[i*n:(i+1)*n,:] = np.random.standard_normal(size = (n,n))#Dtot2[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()
    Utot2_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Utot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
    Vtot2_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Vtot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()

#lvl3
Nb      = Nb//4
Nbvec+=[Nb]
n       = 4*k
Utot3_sparse = np.zeros(shape = (Nb*n,k))
Vtot3_sparse = np.zeros(shape = (Nb*n,k))
Dtot3_sparse = np.zeros(shape = (Nb*n,n))
for i in range(Nb):
    Dtot3_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,n)))#Dtot3[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()
    Utot3_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Utot3[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
    Vtot3_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Vtot3[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()

#lvl4

Nb      = Nb//4
Nbvec+=[Nb]
n       = 4*k
Dtot4_sparse = np.zeros(shape = (Nb*n,n))
for i in range(Nb):
    Dtot4_sparse[i*n:(i+1)*n,:] = sclinalg.orth(np.random.standard_normal(size = (n,n)))#Dtot4[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()


Dmats = [Dtot1_sparse,Dtot2_sparse,Dtot3_sparse,Dtot4_sparse]
Vmats = [Vtot1_sparse,Vtot2_sparse,Vtot3_sparse]
Umats = [Utot1_sparse,Utot2_sparse,Utot3_sparse]

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




