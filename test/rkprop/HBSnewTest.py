import numpy as np
import time
import matplotlib.pyplot as plt
import torch

import matAssembly.HBS.HBSnew as HBSnew
import matAssembly.HBS.HBStorch as HBStorch
import scipy.linalg as sclinalg

import matAssembly.HBS.ULVsparse as ULVsparse
import matAssembly.HBS.ULVsparse_torch as ULVsparse_torch
import torch.linalg as tla

torchbool = False
nl = 8*8
Nvec = np.array([2**16],dtype=np.int64)#np.array([2**14,2**16,2**18,2**20],dtype=np.int64)
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


    ticULV = time.time()
    Qlist,Wlist,Uulist,Rlist,NNvec = ULVsparse.compute_ULV(Umats,Dmats,Vmats,Nbvec)
    tocULV = time.time()

    #tic = time.time()
    SHBS = HBSnew.HBSMAT()
    SHBS.set_mats(Umats,Dmats,Vmats,Nbvec)
    
    SHBS0 = HBSnew.HBSMAT(SHBS)
    SHBS0.set_Nbvec(Nbvec)
    tic = time.time()
    SHBS0.construct(k+5,True)
    print("==========================")
    print("HBS time = ",time.time()-tic)
    print("==========================")
    x= np.random.standard_normal(size=(SHBS.shape[1],))
    b = SHBS.matvec(x,mode='T')
    xhat = SHBS0.solve(b,mode='T')
    print("==========================")
    print("solve err = ",np.linalg.norm(x-xhat)/np.linalg.norm(x))
    print("==========================")
    