
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
def block_Q_and_R_tens(W1,W2,Dtot):
    n = Dtot.shape[1]
    Nb = Dtot.shape[0]
    k = W2.shape[2]
    Q = torch.zeros(size = (Nb,n,n))
    R = torch.zeros(size = (Nb,n,n))
    
    for i in range(Nb):
        D = Dtot[i,:,:]@W1[i,:,:]
        a,tau   = torch.geqrf(D)
        R[i,:,:] = torch.ormqr(a, tau, torch.cat((Dtot[i,:,:]@W1[i,:,:],Dtot[i,:,:]@W2[i,:,:]),axis=1),left=True, transpose=True)
        Q[i,:,:] = torch.ormqr(a, tau, torch.eye(n))
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
        [Q,Ru] = tla.qr(D,mode='reduced')
        R22=0
        W1 = torch.eye(Ru.shape[1])
        NN = Ru.shape[0]

    else:
        tic = time.time()
        k = Vtot.shape[1]
        n = Vtot.shape[0]//Nb
        NN = (n-k)*Nb
        W1 = torch.zeros(size = (Nb*n,n-k))
        Ru = torch.zeros(size = (NN,n))
        R22 = torch.zeros(size = (Nb*k,k))
        tinit = time.time()-tic
        VV = convert_to_torch_tens(Vtot[:,:k],Nb).to(device)
        tic = time.time()
        if n>k:
            Wprime = block_qr_tens(VV)
        else:
            Wprime = torch.zeros(size=(n*Nb,0))
        tVc+=time.time()-tic
        Dprime = convert_to_torch_tens(Dtot,Nb).to(device)
        tic = time.time()
        Q,R = block_Q_and_R_tens(Wprime,VV,Dprime)
        tQ += time.time()-tic
        W1 = convert_to_blkdiag(Wprime)
        Q = convert_to_blkdiag(Q)
        R = convert_to_blkdiag(R)
        tic = time.time()
        for box_ind in range(Nb):
            Ru[box_ind*(n-k):(box_ind+1)*(n-k),:] = R[box_ind*n:(box_ind+1)*n,:][:n-k,:]
            R22[box_ind*k:(box_ind+1)*k,:]          = R[box_ind*n:(box_ind+1)*n,n-k:][n-k:,:]
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



def compute_ULV(Umats,Dmats,Vmats,Nbvec,device):
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
    R_off_list = []
    Rulist = []
    Qlist = []
    for i in range(len(Dmats)):
        print("lvl = ",i)
        print("Nbvec = ",Nbvec)
        
        if i==0:
            Rprime = Dmats[0]
            Q,W1,Ru,R_22,NN = compute_QRW_sparse(Rprime,Vmats[0],Nbvec[0],device)
            n = Vmats[0].shape[0]//Nbvec[0]
            k = Vmats[0].shape[1]
        else:
            Rhat = sparse_block_mult(Uhat,Dmats[i],Nbvec[i-1],Nbvec[i])
            Rhat = block_diag_add(Rhat,R_22,Nbvec[i],Nbvec[i-1])
            if i<len(Vmats):
                Q,W1,Ru,R_22,NN = compute_QRW_sparse(Rhat,Vmats[i],Nbvec[i],device)
                n = Vmats[i].shape[0]//Nbvec[i]
                k = Vmats[i].shape[1]
            else:
                Q,W1,Ru,R_22,NN = compute_QRW_sparse(Rhat,None,Nbvec[i],device)
        NNvec += [NNvec[-1]+NN]



        if i<len(Umats):
            if i == 0:
                Uu = sparse_block_mult(Q[:,:(n-k)],Umats[0],Nbvec[0],Nbvec[0],mode='T')
                Ud = sparse_block_mult(Q[:,(n-k):],Umats[0],Nbvec[0],Nbvec[0],mode='T')
                Uulist+=[Uu]
                Uhat=Ud
                
            else:
                Uhat = sparse_block_mult(Uhat,Umats[i],Nbvec[i-1],Nbvec[i])
                Uu = sparse_block_mult(Q[:,:(n-k)],Uhat,Nbvec[i],Nbvec[i],mode='T')
                Ud = sparse_block_mult(Q[:,(n-k):],Uhat,Nbvec[i],Nbvec[i],mode='T')
                Uulist+=[Uu]
                Uhat=Ud
        Rlist+=[Ru]
        Qlist+=[Q]
        W1list+=[W1]
        
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
