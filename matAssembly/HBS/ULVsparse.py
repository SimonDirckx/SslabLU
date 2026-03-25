
import numpy as np
import scipy.linalg as splinalg
import matplotlib.pyplot as plt
import time
import torch
import torch.linalg as tla


'''
Factorize Q^* S W = R
with R upper triangular 

Q,R,W given in reduced format

'''

def block_qr(A,n,k,Nb):
    C = torch.zeros(size = (n*Nb,n-k))
    for i in range(Nb):
        Q,_ = tla.qr(A[i*n:(i+1)*n,:],mode='complete')
        C[i*n:(i+1)*n,:] = Q[:,k:]
    return C


def compute_QR_sparse(Dtot,Wtot,Nb,k):
    if Nb==1:
        D = Dtot
        [Q,Ru] = np.linalg.qr(D)
        R22=0
        N = Ru.shape[0]

    else:
        n = Dtot.shape[0]//Nb
        NN = (n-k)*Nb
        Q = np.zeros(shape = (Nb*n,n))
        Ru = np.zeros(shape = (Nb*(n-k),n))
        R22 = np.zeros(shape = (Nb*k,k))
        for box_ind in range(Nb):
            D = Dtot[box_ind*n:(box_ind+1)*n,:]@W[box_ind*n:(box_ind+1)*n,:]
            [Qloc,R]   = tla.qr(torch.from_numpy(D),mode = 'reduced')
            Qloc = (Qloc.detach().cpu().numpy())
            R = (R.detach().cpu().numpy())
            Q[box_ind*n:(box_ind+1)*n,:]           = Qloc
            Ru[box_ind*(n-k):(box_ind+1)*(n-k),:]  = R[:n-k,:]
            R22[box_ind*k:(box_ind+1)*k,:]          = R[:,n-k:][n-k:,:]
    return Q,Ru,R22,NN


def compute_QRW_sparse(Dtot,Vtot,Nb):
    
    '''
    
    Given (repr. of) R_{ell}' and V_{ell}, compute Q_{ell}, W_{ell} and upper triangular matrix
    R such that Q_{ell}^* R_{ell}'W_{ell} is upper triangular

    '''
    
    tVc = 0
    tQ = 0
    tmv = 0
    tinit = 0
    if Nb==1:
        D = Dtot
        [Q,Ru] = np.linalg.qr(D)
        R22=0
        W1 = np.identity(Ru.shape[1])
        W2 = np.zeros(shape = (Ru.shape[0],0))
        NN = Ru.shape[0]

    else:
        tic = time.time()
        k = Vtot.shape[1]
        n = Vtot.shape[0]//Nb
        NN = (n-k)*Nb
        Q = np.zeros(shape = (Nb*n,n))
        W1 = np.zeros(shape = (Nb*n,n-k))
        W2 = Vtot[:,:k]
        Ru = np.zeros(shape = (NN,n))
        R22 = np.zeros(shape = (Nb*k,k))
        tinit = time.time()-tic
        tic = time.time()
        print("n = ",n)
        print("k = ",k)
        if n>k:
            W1 = (block_qr(torch.from_numpy(W2),n,k,Nb)).detach().cpu().numpy()
        else:
            W1 = np.zeros(shape=(n*Nb,0))
        tVc+=time.time()-tic
        for box_ind in range(Nb):
            tic = time.time()
            if n>k:
                D = Dtot[box_ind*n:(box_ind+1)*n,:]@np.append(W1[box_ind*n:(box_ind+1)*n,:],W2[box_ind*n:(box_ind+1)*n,:],axis = 1)
            else:
                D = Dtot[box_ind*n:(box_ind+1)*n,:]@W2[box_ind*n:(box_ind+1)*n,:]
            [Qloc,R]   = tla.qr(torch.from_numpy(D),mode = 'reduced')
            Qloc = (Qloc.detach().cpu().numpy())
            R = (R.detach().cpu().numpy())
            tQ+=time.time()-tic
            tic = time.time()
            Q[box_ind*n:(box_ind+1)*n,:]           = Qloc
            Ru[box_ind*(n-k):(box_ind+1)*(n-k),:]  = R[:n-k,:]
            R22[box_ind*k:(box_ind+1)*k,:]          = R[:,n-k:][n-k:,:]
            tmv += time.time()-tic
    print("tVc//tQ//tmv//tinit = ",tVc,"//",tQ,"//",tmv,"//",tinit)
    W = np.append(W1,W2,axis=1)
    return Q,W,Ru,R22,NN
def sparse_block_mult(A,B,NbA,NbB,mode='N'):
    
    '''
    Multiply block diag matrices
    INPUT  
        A,B     :   Block diagonal matrices (reduced form)
        NbA,NbA :   Number of blocks for A and B
        mode    :   wheter A@B (Normal, 'N') or A.T@B (Transpose,'T')
    OUTPUT
        C       :   product of A and B, in reduced form
    '''
    if mode=='N':
        
        na = A.shape[0]//NbA
        ka = A.shape[1]
        nb = B.shape[0]//NbB
        kb = B.shape[1]
        C = np.zeros(shape = (A.shape[0],B.shape[1]))
        if NbB>=NbA:
            fac = (NbB//NbA)
            for i in range(NbA):
                Bsub = np.zeros(shape = (fac*nb,fac*kb))
                Asub = A[i*na:(i+1)*na,:]
                for j in range(fac):
                    Bsub[j*nb:(j+1)*nb,:][:,j*kb:(j+1)*kb] = B[i*nb+j*nb:i*nb+(j+1)*nb,:]
                C[i*na:(i+1)*na,:] = Asub@Bsub
        else:
            fac = (NbA//NbB)
            startA=0
            for i in range(NbB):
                Asub = np.zeros(shape = (fac*na,fac*ka))
                Bsub = B[i*nb:(i+1)*nb,:]
                for j in range(fac):
                    Asub[j*na:(j+1)*na,:][:,j*ka:(j+1)*ka] = A[startA+j*na:startA+(j+1)*na,:]
                C[startA:startA+fac*na,:] = Asub@Bsub
                startA+=fac*na
    elif mode=='T':
        # this assumes NbB>NbA
        na = A.shape[0]//NbA
        ka = A.shape[1]
        nb = B.shape[0]//NbB
        kb = B.shape[1]
        fac = (NbB//NbA)
        C = np.zeros(shape = (ka*NbA,kb*fac))
        for i in range(NbA):
            Bsub = np.zeros(shape = (fac*nb,fac*kb))
            Asub = A[i*na:(i+1)*na,:]
            for j in range(fac):
                Bsub[j*nb:(j+1)*nb,:][:,j*kb:(j+1)*kb] = B[i*na+j*nb:i*na+(j+1)*nb,:]
            C[i*ka:(i+1)*ka,:] = Asub.T@Bsub

    else:
        raise(ValueError("mode not recognized"))


    return C

def block_diag_add(A,B,NbA,NbB):
    '''
    Add block diag matrices
    INPUT  
        A,B     :   Block diagonal matrices (reduced form)
        NbA,NbA :   Number of blocks for A and B
    OUTPUT
        C       :   sum of A and B, in reduced form
    '''

    assert(A.shape[0]==B.shape[0])
    Nb = min(NbA,NbB)
    kA = A.shape[1]
    kB = B.shape[1]
    nA=A.shape[0]//NbA
    nB=B.shape[0]//NbB
    k = min(kA,kB)
    fac = max(kA//k,kB//k)
    if kA>=kB:
        C=A
        for i in range(NbA):
            for j in range(fac):
                C[i*nA+j*nB:i*nA+(j+1)*nB,:][:,j*kB:(j+1)*kB]+=B[i*nA+j*nB:i*nA+(j+1)*nB,:]
    else:
        raise(ValueError("put smol frist"))
    return C

def apply_sparse_block(A,B,Nb,mode='N'):
    
    #A is block matrix
    if mode=='N':
        k = A.shape[1]
        n = A.shape[0]//Nb
        C = np.zeros(shape = (A.shape[0],B.shape[1]))
        for i in range(Nb):
            C[i*n:(i+1)*n,:] = A[i*n:(i+1)*n,:]@B[i*k:(i+1)*k,:]
    elif mode=='T':
        k = A.shape[1]
        n = A.shape[0]//Nb
        C = np.zeros(shape = (k*Nb,B.shape[1]))
        for i in range(Nb):
            C[i*k:(i+1)*k,:] = A[i*n:(i+1)*n,:].T@B[i*n:(i+1)*n,:]
    else:
        raise ValueError("mode not recognized")
    
    return C
def apply_sparse_block_from_right(A,B,Nb):
    
    k = B.shape[1]
    n = B.shape[0]//Nb
    C = np.zeros(shape = (A.shape[0],B.shape[1]*Nb))
    for i in range(Nb):
        C[:,i*k:(i+1)*k] = A[:,i*n:(i+1)*n]@B[i*n:(i+1)*n,:]
    return C

def block_solve(A,B,Nb,mode='N'):
    if mode == 'N':
        n = A.shape[0]//Nb
        C = np.zeros(shape = (A.shape[0],B.shape[1]))
        for i in range(Nb):
            C[i*n:(i+1)*n,:] = np.linalg.solve(A[i*n:(i+1)*n,:],B[i*n:(i+1)*n,:])
    elif mode=='T':
        n = A.shape[0]//Nb
        C = np.zeros(shape = (A.shape[0],B.shape[1]))
        for i in range(Nb):
            C[i*n:(i+1)*n,:] = np.linalg.solve(A[i*n:(i+1)*n,:].T,B[i*n:(i+1)*n,:])
    else:
        raise ValueError("Mode not recognized")
    return C

#Q1,Q2,W1,W2,R11,R12,R22,NN = compute_QRW_sparse(Rprime,Vmats[0],Nbvec[0])


def compute_ULV(Umats,Dmats,Vmats,Nbvec):
    '''
    computes ULV decomp

    INPUTS
        Umats   : list (!) of U matrices in HBS decomp
        Dmats   : list (!) of D matrices in HBS decomp
        Vmats   : list (!) of V matrices in HBS decomp
        Nbvec   : "Number-of-blocks" vector, Nbvec[i] = #blocks at level i (counted from leaf level)

    OUTPUTS
        Qtot    : sparse rep of total Q matrix
        Rtot    : sparse rep of total R matrix
        Wtot    : sparse rep of total W matrix
        NNvec, NNQvec, NNRvec, NNWvec   :   companion vecs for the cbd mats
                                            loosely speaking, bookkeeping for matmul

    '''
    
    NNvec = np.zeros(shape=(0,),dtype=np.int64)
    NNvec = np.append(NNvec,0)
    Qlist = []
    Wlist = []
    Uulist  = []
    Rlist = []
    torch.set_default_dtype(torch.float64)
    for i in range(len(Dmats)):
        print("lvl = ",i)
        print("Nbvec = ",Nbvec)
        
        if i==0:
            Rprime = Dmats[0]
            Q,W,Ru,R_22,NN = compute_QRW_sparse(Rprime,Vmats[0],Nbvec[0])
        else:
            Rhat = sparse_block_mult(Uhat,Dmats[i],Nbvec[i-1],Nbvec[i])
            Rhat = block_diag_add(Rhat,R_22,Nbvec[i],Nbvec[i-1])
            if i<len(Vmats):
                Q,W,Ru,R_22,NN = compute_QRW_sparse(Rhat,Vmats[i],Nbvec[i])
            else:
                Q,W,Ru,R_22,NN = compute_QRW_sparse(Rhat,None,Nbvec[i])
        NNvec = np.append(NNvec,NNvec[-1]+NN)



        if i<len(Umats):
            n = Umats[i].shape[0]//Nbvec[i]
            k = Umats[i].shape[1]
            if i == 0:
                Uu = sparse_block_mult(Q[:,:n-k],Umats[0],Nbvec[0],Nbvec[0],mode='T')
                Ud = sparse_block_mult(Q[:,n-k:],Umats[0],Nbvec[0],Nbvec[0],mode='T')
                Uulist+=[Uu]
                Uhat=Ud
                
            else:
                Uhat = sparse_block_mult(Uhat,Umats[i],Nbvec[i-1],Nbvec[i])
                Uu = sparse_block_mult(Q[:,:n-k],Uhat,Nbvec[i],Nbvec[i],mode='T')
                Ud = sparse_block_mult(Q[:,n-k:],Uhat,Nbvec[i],Nbvec[i],mode='T')
                Uulist+=[Uu]
                Uhat=Ud
                
        Qlist+=[Q]
        Wlist+=[W]
        Rlist+=[Ru]
        
    return Qlist,Wlist,Uulist,Rlist,NNvec

def solve(Umats,Dmats,Qlist,Wlist,Uulist,Rlist,NNvec,Nbvec,rhs,mode='N'):
    if mode=='N':
        L = len(Dmats)
        if rhs.ndim == 1:
            rhshat = rhs[:,np.newaxis].copy()
        else:
            rhshat = rhs.copy()
        for i in range(len(Qlist)):
            rtmp = rhshat[NNvec[i]:,:].copy()
            if i<len(Qlist)-1:
                n = Umats[i].shape[0]//Nbvec[i]
                k = Umats[i].shape[1]
                rhshat[NNvec[i]:NNvec[i+1],:] = apply_sparse_block(Qlist[i][:,:n-k],rtmp,Nbvec[i],mode='T')
                rhshat[NNvec[i+1]:,:] = apply_sparse_block(Qlist[i][:,n-k:],rtmp,Nbvec[i],mode='T')
            else:
                rhshat[NNvec[i]:NNvec[i+1],:] = apply_sparse_block(Qlist[i],rtmp,Nbvec[i],mode='T')
        
        y = np.zeros(shape=rhshat.shape)

        y[NNvec[L-1]:,:] = block_solve(Rlist[L-1],rhshat[NNvec[L-1]:,:],Nbvec[L-1])
        v = apply_sparse_block(Dmats[L-1],y[NNvec[L-1]:,:],Nbvec[L-1])

        for i in range(L-2,-1,-1):
            n = Umats[i].shape[0]//Nbvec[i]
            k = Umats[i].shape[1]
            rhs0    =   rhshat[NNvec[i]:NNvec[i+1],:]\
                        -apply_sparse_block(Uulist[i],v,Nbvec[i])\
                        -apply_sparse_block(Rlist[i][:,n-k:],y[NNvec[i+1]:,:],Nbvec[i])
            y[NNvec[i]:NNvec[i+1],:]    = block_solve(Rlist[i][:,:n-k],rhs0,Nbvec[i]).copy()
            y[NNvec[i]:,:]              = apply_sparse_block(Wlist[i][:,:n-k],y[NNvec[i]:NNvec[i+1],:],Nbvec[i])\
                                        +apply_sparse_block(Wlist[i][:,n-k:],y[NNvec[i+1]:,:],Nbvec[i])
            v       = apply_sparse_block(Umats[i],v,Nbvec[i]).copy()\
                    +apply_sparse_block(Dmats[i],y[NNvec[i]:,:],Nbvec[i])
    elif mode=='T':
        L = len(Dmats)
        if rhs.ndim == 1:
            rhshat = rhs[:,None].copy()
        else:
            rhshat = rhs.copy()
        y = rhshat.copy()
        v = np.zeros(shape=(Umats[0].shape[0],rhshat.shape[1]))
        for i in range(L-1):
            n = Umats[i].shape[0]//Nbvec[i]
            k = Umats[i].shape[1]
            rhscopy = rhshat.copy()
            rhshat[:NNvec[i+1]-NNvec[i],:] = apply_sparse_block(Wlist[i][:,:(n-k)],rhscopy,Nbvec[i],mode='T')
            rhshat[NNvec[i+1]-NNvec[i]:,:] = apply_sparse_block(Wlist[i][:,(n-k):],rhscopy,Nbvec[i],mode='T')
            y[NNvec[i]:NNvec[i+1],:] = block_solve(Rlist[i][:,:n-k],rhshat[:NNvec[i+1]-NNvec[i],:],Nbvec[i],mode='T')
            v=apply_sparse_block(Uulist[i],y[NNvec[i]:NNvec[i+1],:],Nbvec[i],mode='T')+apply_sparse_block(Umats[i],v,Nbvec[i],mode='T')
            rhshat = rhshat[NNvec[i+1]-NNvec[i]:,:]-apply_sparse_block(Rlist[i][:,n-k:],y[NNvec[i]:NNvec[i+1],:],Nbvec[i],mode='T')-apply_sparse_block(Dmats[i+1],v,Nbvec[i+1],mode='T')
        y[NNvec[-2]:,:] = block_solve(Rlist[-1],rhshat,Nbvec[-1],mode='T')
        for i in range(len(Qlist)-1,-1,-1):
            if i<len(Qlist)-1:
                n = Umats[i].shape[0]//Nbvec[i]
                k = Umats[i].shape[1]
                y[NNvec[i]:,:] = apply_sparse_block(Qlist[i][:,:n-k],y[NNvec[i]:NNvec[i+1],:],Nbvec[i])\
                    +apply_sparse_block(Qlist[i][:,n-k:],y[NNvec[i+1]:,:],Nbvec[i])
            else:
                y[NNvec[i]:NNvec[i+1],:] = apply_sparse_block(Qlist[i],y[NNvec[i]:NNvec[i+1],:],Nbvec[i])
    else:
        raise NotImplementedError("mode not recognized")
    if rhs.ndim==1:
            y = y.flatten()
    return y