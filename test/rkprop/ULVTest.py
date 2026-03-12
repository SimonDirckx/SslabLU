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

N = 2**12
nl = 8*8
Nleaves = N//nl
k = nl//4
k0 = min(nl,k)
L = (int)(np.log2(Nleaves))//2 + 1


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


Qlist,Wlist,Ulist,Rlist,R_off_list,NNvec = ULVdense.compute_ULV(Umats,Dmats,Vmats,Nbvec)


SS = SHBS
for i in range(len(Qlist)):
    SS[:,NNvec[i]:] = SS[:,NNvec[i]:]@Wlist[i]
    SS[NNvec[i]:,:] = Qlist[i].T@SS[NNvec[i]:,:]
for i in range(len(Ulist)):
    plt.figure(1)
    plt.spy(Ulist[i],precision=1e-8)
    plt.show()
    print("NNZ = ",np.count_nonzero(abs(Ulist[i])>1e-8))
    print("NNZ U = ",np.count_nonzero(abs(Umats[i])>1e-8))
plt.figure(1)
plt.spy(SS,precision=1e-8)
plt.show()
for i in range(len(Ulist)):
    plt.figure(1)
    mtest = Ulist[i]@Dmats[i+1]@Wlist[i+1]
    plt.spy(mtest,precision=1e-8)
    plt.figure(2)
    plt.spy(SS[NNvec[i]:NNvec[i+1],:][:,NNvec[i+1]:NNvec[i+2]],precision=1e-8)
    plt.figure(3)
    plt.spy(SS,precision=1e-8)
    plt.show()

#plt.figure(1)
#mtest = Ulist[1][NNvec[0]:NNvec[1],:]@Dmats[2]@Wlist[2][:,:NNvec[3]-NNvec[2]]+Ulist[0][NNvec[0]:NNvec[1],:]@Dmats[1]@Wlist[1][:,NNvec[2]-NNvec[1]:]@Wlist[2][:,:NNvec[3]-NNvec[2]]+R_off_list[0]@Wlist[1][:,NNvec[2]-NNvec[1]:]@Wlist[2][:,:NNvec[3]-NNvec[2]]
#plt.spy(mtest,precision=1e-8)
#plt.figure(2)
#plt.spy(SS[NNvec[0]:NNvec[1],:][:,NNvec[2]:NNvec[3]]-mtest,precision=1e-8)
#plt.show()




