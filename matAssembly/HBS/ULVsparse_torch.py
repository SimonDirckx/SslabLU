
import time
import torch
import torch.linalg as tla
import matplotlib.pyplot as plt

'''
Factorize Q^* S W = R
with R upper triangular 

Q,R,W given in reduced format

'''
def convert_to_torch_tens(A,Nb):
    n = A.shape[0]//Nb
    k = A.shape[1]
    B = torch.zeros(size = (Nb,n,k))
    for i in range(Nb):
        B[i,:,:] = A[i*n:(i+1)*n,:]
    return B
def convert_to_blkdiag(A):
    n = A.shape[1]
    k = A.shape[2]
    Nb = A.shape[0]
    B = torch.zeros(size = (n*Nb,k))
    for i in range(Nb):
        B[i*n:(i+1)*n,:] = A[i,:,:]
    return B

def block_qr(A,n,k,Nb):
    C = torch.zeros(size = (n*Nb,n-k))
    for i in range(Nb):
        Q,_ = tla.qr(A[i*n:(i+1)*n,:],mode='complete')
        C[i*n:(i+1)*n,:] = Q[:,k:]
    return C
def block_qr_tens(A):
    n = A.shape[1]
    k = A.shape[2]
    Nb = A.shape[0]
    C = torch.zeros(size = (Nb,n,n-k))
    for i in range(Nb):
        Q,_ = tla.qr(A[i,:,:],mode='complete')
        C[i,:,:] = Q[:,k:]
    return C

def block_Q_and_R(W1,W2,Dtot,Nb):
    k = W2.shape[1]
    n = Dtot.shape[1]
    Q = torch.zeros(size = (n*Nb,n))
    R = torch.zeros(size = (n*Nb,n))
    
    for i in range(Nb):
        D = Dtot[i*n:(i+1)*n,:]@torch.cat((W1[i*n:(i+1)*n,:],W2[i*n:(i+1)*n,:]),axis = 1)
        [Q[i*n:(i+1)*n,:],R[i*n:(i+1)*n,:]]   = tla.qr(D,mode = 'reduced')
        # = Q0
        # = R0
    return Q,R
def block_Q_and_R_tens(W12,Dtot):
    n = Dtot.shape[1]
    Nb = Dtot.shape[0]
    Q = torch.zeros(size = (Nb,n,n))
    R = torch.zeros(size = (Nb,n,n))
    for i in range(Nb):
        [Q[i,:,:],R[i,:,:]]   = tla.qr(Dtot[i,:,:]@W12[i,:,:])
    return Q,R





def compute_QRW_sparse(Dtot,Vtot,Nb,device):
    
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
        [Q,Ru] = tla.qr(D[0,:,:],mode='reduced')
        Ru = Ru[None,:,:]
        Q = Q[None,:,:]
        R22=0
        W1 = None
        NN = Ru.shape[1]

    else:
        tic = time.time()
        k = Vtot.shape[2]
        n = Vtot.shape[1]
        NN = (n-k)*Nb
        tinit = time.time()-tic
        tic = time.time()
        if n>k:
            W1 = block_qr_tens(Vtot).to(device)
        else:
            W1 = torch.zeros(size=(Nb,n,0)).to(device)
        tVc+=time.time()-tic
        tic = time.time()
        W12 = torch.cat((W1,Vtot),axis=2)
        Q,R = block_Q_and_R_tens(W12,Dtot.to(device))
        tQ += time.time()-tic
        tic = time.time()
        Ru = R[:,:n-k,:]
        R22 = R[:,n-k:,n-k:]
        tmv += time.time()-tic
    print("tVc//tQ//tmv//tinit = ",tVc,"//",tQ,"//",tmv,"//",tinit)
    return Q,W1,Ru,R22,NN
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
        C = torch.zeros(size = (A.shape[0],B.shape[1]))
        if NbB>=NbA:
            fac = (NbB//NbA)
            for i in range(NbA):
                Bsub = torch.zeros(size = (fac*nb,fac*kb))
                Asub = A[i*na:(i+1)*na,:]
                for j in range(fac):
                    Bsub[j*nb:(j+1)*nb,:][:,j*kb:(j+1)*kb] = B[i*nb+j*nb:i*nb+(j+1)*nb,:]
                C[i*na:(i+1)*na,:] = Asub@Bsub
        else:
            fac = (NbA//NbB)
            startA=0
            for i in range(NbB):
                Asub = torch.zeros(size = (fac*na,fac*ka))
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
        C = torch.zeros(size = (ka*NbA,kb*fac))
        for i in range(NbA):
            Bsub = torch.zeros(size = (fac*nb,fac*kb))
            Asub = A[i*na:(i+1)*na,:]
            for j in range(fac):
                Bsub[j*nb:(j+1)*nb,:][:,j*kb:(j+1)*kb] = B[i*na+j*nb:i*na+(j+1)*nb,:]
            C[i*ka:(i+1)*ka,:] = Asub.T@Bsub

    else:
        raise(ValueError("mode not recognized"))


    return C
def sparse_block_mult_tens(A,B,mode='N'):
    
    '''
    Multiply block diag matrices
    INPUT  
        A,B     :   Block diagonal matrices (reduced form)
        NbA,NbA :   Number of blocks for A and B
        mode    :   wheter A@B (Normal, 'N') or A.T@B (Transpose,'T')
    OUTPUT
        C       :   product of A and B, in reduced form
    '''
    NbA = A.shape[0]
    NbB = B.shape[0]
    na = A.shape[1]
    ka = A.shape[2]
    nb = B.shape[1]
    kb = B.shape[2]
    if mode=='N':
        # this assumes NbA>=NbB
        fac = (NbA//NbB)
        C = torch.zeros(size = (NbB,fac*na,kb))
        
        #startA=0
        for i in range(NbB):
            Asub = torch.zeros(size = (fac*na,fac*ka)).to(A.get_device())
            Bsub = B[i,:,:]
            for j in range(fac):
                Asub[j*na:(j+1)*na,:][:,j*ka:(j+1)*ka] = A[fac*i+j,:,:]#startA+j*na:startA+(j+1)*na,:]
            C[i,:,:] = Asub@B[i,:,:]
            #startA+=fac*na
    elif mode=='T':
        # this assumes NbB=NbA
        C = torch.zeros(size = (NbA,ka,kb)).to(A.get_device())
        for i in range(NbA):
            C[i,:,:] = A[i,:,:].T@B[i,:,:]

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
def block_diag_add_tens(A,B):
    '''
    Add block diag matrices
    INPUT  
        A,B     :   Block diagonal matrices (reduced form)
        NbA,NbA :   Number of blocks for A and B
    OUTPUT
        C       :   sum of A and B, in reduced form
    '''
    kA = A.shape[2]
    kB = B.shape[2]
    NbA = A.shape[0]
    NbB = B.shape[0]
    nA=A.shape[1]
    nB=B.shape[1]
    assert(NbA*nA==NbB*nB)
    k = min(kA,kB)
    fac = max(kA//k,kB//k)
    if kA>=kB:
        C=A
        for i in range(NbA):
            for j in range(fac):
                C[i,j*nB:(j+1)*nB,:][:,j*kB:(j+1)*kB]+=B[i*fac+j,:,:]
    else:
        raise(ValueError("put smol frist"))
    return C

def apply_sparse_block(A,B,Nb,mode='N'):
    
    #A is block matrix
    if mode=='N':
        k = A.shape[1]
        n = A.shape[0]//Nb
        C = torch.zeros(size = (A.shape[0],B.shape[1]))
        for i in range(Nb):
            C[i*n:(i+1)*n,:] = A[i*n:(i+1)*n,:]@B[i*k:(i+1)*k,:]
    elif mode=='T':
        k = A.shape[1]
        n = A.shape[0]//Nb
        C = torch.zeros(size = (k*Nb,B.shape[1]))
        for i in range(Nb):
            C[i*k:(i+1)*k,:] = A[i*n:(i+1)*n,:].T@B[i*n:(i+1)*n,:]
    else:
        raise ValueError("mode not recognized")
    
    return C
def apply_sparse_block_from_right(A,B,Nb):
    
    k = B.shape[1]
    n = B.shape[0]//Nb
    C = torch.zeros(size = (A.shape[0],B.shape[1]*Nb))
    for i in range(Nb):
        C[:,i*k:(i+1)*k] = A[:,i*n:(i+1)*n]@B[i*n:(i+1)*n,:]
    return C

def block_solve(A,B,Nb,mode='N'):
    if mode == 'N':
        n = A.shape[0]//Nb
        C = torch.zeros(size = (A.shape[0],B.shape[1]))
        for i in range(Nb):
            C[i*n:(i+1)*n,:] = tla.solve(A[i*n:(i+1)*n,:],B[i*n:(i+1)*n,:])
    elif mode=='T':
        n = A.shape[0]//Nb
        C = torch.zeros(size = (A.shape[0],B.shape[1]))
        for i in range(Nb):
            C[i*n:(i+1)*n,:] = tla.solve(A[i*n:(i+1)*n,:].T,B[i*n:(i+1)*n,:])
    else:
        raise ValueError("Mode not recognized")
    return C



def compute_ULV(Umats,Dmats,Vmats,Nbvec,device,Utens,Dtens,Vtens):
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
    
    NNvec = [0]
    W1list = []
    Uulist  = []
    Rlist = []
    Qlist = []
    for i in range(len(Dmats)):
        
        if i==0:
            Rprime = Dtens[0]
            tic = time.time()
            Q,W1,Ru,R_22,NN = compute_QRW_sparse(Rprime.to(device),Vtens[0].to(device),Nbvec[0],device)
            print("TIME = ",time.time()-tic)
            n = Vtens[0].shape[1]
            k = Vtens[0].shape[2]
        else:
            Rhat = sparse_block_mult_tens(Uhat.to(device),Dtens[i].to(device))
            Rhat = block_diag_add_tens(Rhat,R_22)
            
            if i<len(Vtens):
                tic = time.time()
                Q,W1,Ru,R_22,NN = compute_QRW_sparse(Rhat,Vtens[i].to(device),Nbvec[i],device)
                print("TIME = ",time.time()-tic)
                n = Vtens[i].shape[1]
                k = Vtens[i].shape[2]
            else:
                tic = time.time()
                Q,W1,Ru,R_22,NN = compute_QRW_sparse(Rhat,None,Nbvec[i],device)
                print("TIME = ",time.time()-tic)
        NNvec += [NNvec[-1]+NN]

        tic = time.time()
        if i<len(Umats):
            W1list+=[W1]
            if i == 0:
                Uu = sparse_block_mult_tens(Q[:,:,:(n-k)].to(device),Utens[0].to(device),mode='T')
                Ud = sparse_block_mult_tens(Q[:,:,(n-k):].to(device),Utens[0].to(device),mode='T')
                Uulist+=[Uu]
                Uhat=Ud.to(device)
                
            else:
                Uhat = sparse_block_mult_tens(Uhat,Utens[i].to(device))
                Uu = sparse_block_mult_tens(Q[:,:,:(n-k)].to(device),Uhat.to(device),mode='T')
                Ud = sparse_block_mult_tens(Q[:,:,(n-k):].to(device),Uhat.to(device),mode='T')
                Uulist+=[Uu]
                Uhat=Ud.to(device)
        print("TIME UU = ",time.time()-tic)
        Rlist+=[Ru]
        Qlist+=[Q]
        
        
    return Qlist,W1list,Uulist,Rlist,NNvec

def solve(Umats,Dmats,Qlist,W1list,W2list,Uulist,Rlist,NNvec,Nbvec,rhs):
    L = len(Dmats)
    if rhs.ndim == 1:
        rhshat = rhs[:,None].detach().clone()
    else:
        rhshat = rhs.detach().clone()
    for i in range(len(Qlist)):
        rtmp = rhshat[NNvec[i]:,:].detach().clone()
        if i<len(Qlist)-1:
            n = Umats[i].shape[0]//Nbvec[i]
            k = Umats[i].shape[1]
            rhshat[NNvec[i]:NNvec[i+1],:] = apply_sparse_block(Qlist[i][:,:(n-k)],rtmp,Nbvec[i],mode='T')
            rhshat[NNvec[i+1]:,:] = apply_sparse_block(Qlist[i][:,(n-k):],rtmp,Nbvec[i],mode='T')
        else:
            rhshat[NNvec[i]:NNvec[i+1],:] = apply_sparse_block(Qlist[i],rtmp,Nbvec[i],mode='T')
    
    y = torch.zeros(size=rhshat.shape)

    y[NNvec[L-1]:,:] = block_solve(Rlist[L-1],rhshat[NNvec[L-1]:,:],Nbvec[L-1])
    v = apply_sparse_block(Dmats[L-1],y[NNvec[L-1]:,:],Nbvec[L-1])

    for i in range(L-2,-1,-1):
        n = Umats[i].shape[0]//Nbvec[i]
        k = Umats[i].shape[1]
        rhs0    =   rhshat[NNvec[i]:NNvec[i+1],:]\
                    -apply_sparse_block(Uulist[i],v,Nbvec[i])\
                    -apply_sparse_block(Rlist[i][:,n-k:],y[NNvec[i+1]:,:],Nbvec[i])
        y[NNvec[i]:NNvec[i+1],:]    = block_solve(Rlist[i][:,:n-k],rhs0,Nbvec[i]).detach().clone()
        y[NNvec[i]:,:]              = apply_sparse_block(W1list[i],y[NNvec[i]:NNvec[i+1],:],Nbvec[i])\
                                    +apply_sparse_block(W2list[i],y[NNvec[i+1]:,:],Nbvec[i])
        v       = apply_sparse_block(Umats[i],v,Nbvec[i]).detach().clone()\
                +apply_sparse_block(Dmats[i],y[NNvec[i]:,:],Nbvec[i])
    if rhs.ndim==1:
        y = torch.flatten(y)
    return y
