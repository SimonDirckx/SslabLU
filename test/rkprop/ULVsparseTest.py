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
import matAssembly.HBS.ULVsparse_torch as ULVsparse_torch
import ULVdense
import time
import torch

torchbool = True
nl = 16*16
Nvec = np.array([2**14,2**16,2**18],dtype=np.int64)#np.array([2**14,2**16,2**18,2**20],dtype=np.int64)
t_ULV_vec = np.zeros(shape = Nvec.shape)
t_solve_vec = np.zeros(shape = Nvec.shape)
for indN in range(len(Nvec)):
    N = Nvec[indN]
    Nleaves = N//nl
    L = (int)(np.log2(Nleaves))//2 + 1
    kvec = np.array([32,64,128],dtype=np.int64)

    k = 32
    k0 = min(nl,k)


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

    if not torchbool:
        ticULV = time.time()
        Q1list,Q2list,W1list,W2list,Uulist,Rlist,R_off_list,NNvec = ULVsparse.compute_ULV(Umats,Dmats,Vmats,Nbvec)
        tocULV = time.time()


        SHBS = HBSnew.HBSMAT()
        SHBS.set_mats(Umats,Dmats,Vmats,Nbvec)
        x= np.random.standard_normal(size=(SHBS.shape[1],2))
        b = SHBS.matvec(x)

        ticSolveULV = time.time()
        xhat = ULVsparse.solve(Umats,Dmats,Q1list,Q2list,W1list,W2list,Uulist,Rlist,R_off_list,NNvec,Nbvec,b)
        tocSolveULV = time.time()
        t_ULV_vec[indN] = tocULV-ticULV
        t_solve_vec[indN] = tocSolveULV-ticSolveULV
        print("solve err = ",np.linalg.norm(xhat-x)/np.linalg.norm(x))
        print("solve ULV time = ", t_solve_vec[indN])
        print("ULV fact. time = ", t_ULV_vec[indN])
        print("Ndofs = ", Dmats[0].shape[0])
    else:
        Dmats_torch = [torch.from_numpy(D) for D in Dmats]
        Umats_torch = [torch.from_numpy(U) for U in Umats]
        Vmats_torch = [torch.from_numpy(V) for V in Vmats]
        Nbvec_torch = torch.from_numpy(np.array(Nbvec,dtype=np.int64))
        ticULV = time.time()
        Qlist,W1list,Uulist,Rlist,NNvec= ULVsparse_torch.compute_ULV(Umats_torch,Dmats_torch,Vmats_torch,Nbvec_torch)
        tocULV = time.time()
        SHBS = HBSnew.HBSMAT()
        SHBS.set_mats(Umats,Dmats,Vmats,Nbvec)
        x= np.random.standard_normal(size=(SHBS.shape[1],2))
        b = SHBS.matvec(x)
        print(NNvec)
        NNvec = [int(N) for N in NNvec]
        ticSolveULV = time.time()
        xhat = ULVsparse_torch.solve(Umats_torch,Dmats_torch,Qlist,W1list,Vmats_torch,Uulist,Rlist,NNvec,Nbvec,torch.from_numpy(b))
        tocSolveULV = time.time()
        xhat = xhat.detach().cpu().numpy()
        
        t_solve_vec[indN] = tocSolveULV-ticSolveULV
        t_ULV_vec[indN] = tocULV-ticULV
        print("solve err = ",np.linalg.norm(xhat-x)/np.linalg.norm(x))
        print("solve ULV time = ", t_solve_vec[indN])
        print("ULV fact. time = ", t_ULV_vec[indN])
        print("Ndofs = ", Dmats[0].shape[0])
plt.figure(1)
plt.loglog(Nvec,t_ULV_vec)
plt.loglog(Nvec,Nvec*1.1*(t_ULV_vec[0]/Nvec[0]),linestyle='dashed')
plt.legend(['tULV','O(N)'])

plt.figure(2)
plt.loglog(Nvec,t_solve_vec)
plt.loglog(Nvec,Nvec*1.1*(t_solve_vec[0]/Nvec[0]),linestyle='dashed')
plt.legend(['tsol','O(N)'])

plt.show()