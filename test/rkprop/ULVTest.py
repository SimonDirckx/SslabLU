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

import ULVdense

nl = 4*4
L = 5
N = (4**(L-1))*nl

Nleaves = N//nl
Lcheck = (int)(np.log2(Nleaves))//2 + 1
print("L-Lcheck = ",L-Lcheck)
k = nl//4
k0 = min(nl,k)



Nb = Nleaves
Nbvec = [Nb]
Utot1_sparse = np.zeros(shape = (Nb*nl,Nb*k0))
Vtot1_sparse = np.zeros(shape = (Nb*nl,Nb*k0))
Dtot1_sparse = np.zeros(shape = (Nb*nl,Nb*nl))

for i in range(Nb):
    Dtot1_sparse[i*nl:(i+1)*nl,:][:,i*nl:(i+1)*nl] = np.random.standard_normal(size = (nl,nl))#Dtot1[i*nl:(i+1)*nl,:][:,i*nl:(i+1)*nl].copy()
    Utot1_sparse[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0] = sclinalg.orth(np.random.standard_normal(size = (nl,k0)))#Utot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()
    Vtot1_sparse[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0] = sclinalg.orth(np.random.standard_normal(size = (nl,k0)))#Vtot1[i*nl:(i+1)*nl,:][:,i*k0:(i+1)*k0].copy()
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
    Utot_sparse = np.zeros(shape = (Nb*n,Nb*k))
    Vtot_sparse = np.zeros(shape = (Nb*n,Nb*k))
    Dtot_sparse = np.zeros(shape = (Nb*n,Nb*n))

    for i in range(Nb):
        Dtot_sparse[i*n:(i+1)*n,:][:,i*n:(i+1)*n] = np.random.standard_normal(size = (n,n))#Dtot2[i*n:(i+1)*n,:][:,i*n:(i+1)*n].copy()
        Utot_sparse[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Utot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
        Vtot_sparse[i*n:(i+1)*n,:][:,i*k:(i+1)*k] = sclinalg.orth(np.random.standard_normal(size = (n,k)))#Vtot2[i*n:(i+1)*n,:][:,i*k:(i+1)*k].copy()
    if ind<L-1:
        Umats += [Utot_sparse]
        Vmats += [Vtot_sparse]
    Dmats += [Dtot_sparse]


SHBS = Dmats[-1]

for i in range(len(Vmats)-1,-1,-1):
    SHBS= Umats[i]@SHBS@Vmats[i].T+Dmats[i]


Qlist,Wlist,Uulist,Rlist,R_off_list,NNvec = ULVdense.compute_ULV(Umats,Dmats,Vmats,Nbvec)

x = np.random.standard_normal(size=(SHBS.shape[0],5))
rhs = SHBS@x
y = ULVdense.solve_ULV(Umats,Dmats,Qlist,Wlist,Uulist,Rlist,R_off_list,NNvec,rhs)
print("solve err = ",np.linalg.norm(x-y)/np.linalg.norm(x))

plt.figure(1)
plt.plot(x-y)
plt.show()