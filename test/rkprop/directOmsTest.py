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


def gen_random_HBS(k,nl,L,quad = True):
    Nbvec = [];
    if quad:
        fac = 4
    else:
        fac = 2
    N = nl*(fac**(L-1))
    Nb = N//nl
    Nbvec+=[Nb]
    k0 = min(k,nl)
    Utot1_sparse = np.zeros(shape = (Nb*nl,k0))
    Vtot1_sparse = np.zeros(shape = (Nb*nl,k0))
    Dtot1_sparse = np.zeros(shape = (Nb*nl,nl))


    for i in range(Nb):
        U = np.linalg.qr(np.random.standard_normal(size = (nl,k0)),mode='reduced')[0]
        V = np.linalg.qr(np.random.standard_normal(size = (nl,k0)),mode='reduced')[0]
        D = np.random.standard_normal(size = (nl,nl))

        Dtot1_sparse[i*nl:(i+1)*nl,:] = D-U@(U.T@D@V)@V.T
        Utot1_sparse[i*nl:(i+1)*nl,:] = U
        Vtot1_sparse[i*nl:(i+1)*nl,:] = V

    Umats = [Utot1_sparse]
    Vmats = [Vtot1_sparse]
    Dmats = [Dtot1_sparse]



    for ind in range(1,L):
        Nb      = Nb//fac
        Nbvec+=[Nb]
        if ind==1:
            n = fac*k0
        else:
            n = fac*k
        Utot_sparse = np.zeros(shape = (Nb*n,k))
        Vtot_sparse = np.zeros(shape = (Nb*n,k))
        Dtot_sparse = np.zeros(shape = (Nb*n,n))

        for i in range(Nb):
            U = np.linalg.qr(np.random.standard_normal(size = (n,k)),mode='reduced')[0]
            V = np.linalg.qr(np.random.standard_normal(size = (n,k)),mode='reduced')[0]
            D = np.random.standard_normal(size = (n,n))
            if ind<L-1:
                D = D-U@(U.T@D@V)@V.T
            Dtot_sparse[i*n:(i+1)*n,:] = D
            Utot_sparse[i*n:(i+1)*n,:] = U
            Vtot_sparse[i*n:(i+1)*n,:] = V

        if ind<L-1:
            Umats += [Utot_sparse]
            Vmats += [Vtot_sparse]
        Dmats += [Dtot_sparse]

    SHBS = HBSnew.HBSMAT()
    SHBS.set_mats(Umats,Dmats,Vmats,Nbvec,fac=fac)
    return SHBS,Nbvec




quad=   True
k   =   16
nl  =   16
L   =   10
SHBS,Nbvec = gen_random_HBS(k,nl,L,quad)
    
SHBS0 = HBSnew.HBSMAT(SHBS,tree=None,quad=quad)
SHBS0.set_Nbvec(Nbvec)
tic = time.time()
SHBS0.construct(k,True)
print("==========================")
print("HBS time = ",time.time()-tic)
print("==========================")


x= np.random.standard_normal(size=(SHBS.shape[1],10))
b = SHBS.matmat(x)
bhat = SHBS0.matmat(x)
xhat = SHBS0.solve(b)
print("==========================")
print("matvec err = ",np.linalg.norm(b-bhat)/np.linalg.norm(b))
print("solve err = ",np.linalg.norm(x-xhat)/np.linalg.norm(x))
print("==========================")


x= np.random.standard_normal(size=(SHBS.shape[1],10))
b = SHBS.rmatmat(x)
bhat = SHBS0.rmatmat(x)
xhat = SHBS0.solve(b,mode='T')
print("==========================")
print("transpose matvec err = ",np.linalg.norm(b-bhat)/np.linalg.norm(b))
print("transpose solve err = ",np.linalg.norm(x-xhat)/np.linalg.norm(x))
print("==========================")



