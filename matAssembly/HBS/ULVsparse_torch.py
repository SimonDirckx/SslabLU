import time
import torch
import torch.linalg as tla
import matplotlib.pyplot as plt

'''
Factorize Q^* S W = R
with R upper triangular 

Q,R,W given in reduced format

'''
def convert_to_torch_tens(A, Nb, device):
    """(Nb*n, k) -> (Nb, n, k).  A view when A is contiguous (the permuted
    gathers in constructHBS/constructHBS_ULV are), so no second copy of the
    samples is made; callers may modify the result in place, which modifies A."""
    A = A.to(device)
    n = A.shape[0] // Nb
    return A[:Nb * n].reshape(Nb, n, A.shape[1])
def convert_to_blkdiag(A):
    n = A.shape[1]
    k = A.shape[2]
    Nb = A.shape[0]
    B = torch.zeros(size = (n*Nb,k))
    for i in range(Nb):
        B[i*n:(i+1)*n,:] = A[i,:,:]
    return B

def block_qr_tens(A,device):
    n = A.shape[1]
    k = A.shape[2]
    Nb = A.shape[0]
    C = torch.zeros(size = (Nb,n,n-k),device=device)
    for i in range(Nb):
        Q,_ = tla.qr(A[i,:,:],mode='complete')
        C[i,:,:] = Q[:,k:]
    return C

def block_Q_and_R(W1,W2,Dtot,Nb,device):
    k = W2.shape[1]
    n = Dtot.shape[1]
    Q = torch.zeros(size = (n*Nb,n),device=device)
    R = torch.zeros(size = (n*Nb,n),device=device)
    
    for i in range(Nb):
        D = Dtot[i*n:(i+1)*n,:]@torch.cat((W1[i*n:(i+1)*n,:],W2[i*n:(i+1)*n,:]),axis = 1)
        [Q[i*n:(i+1)*n,:],R[i*n:(i+1)*n,:]]   = tla.qr(D,mode = 'reduced')
        # = Q0
        # = R0
    return Q,R
def block_Q_and_R_tens(W12,Dtot,device):
    n = Dtot.shape[1]
    Nb = Dtot.shape[0]
    Q = torch.zeros(size = (Nb,n,n),device=device)
    R = torch.zeros(size = (Nb,n,n),device=device)
    for i in range(Nb):
        [Q[i,:,:],R[i,:,:]]   = tla.qr(Dtot[i,:,:]@W12[i,:,:])
    return Q,R
def compute_QR_sparse(Dtot,Wtot,k,device):
    Nb = Dtot.shape[0]
    if Nb==1:
        D = Dtot
        [Q,Ru] = tla.qr(D)
        R22=0
        NN = Ru.shape[0]

    else:
        n = Dtot.shape[1]
        NN = (n-k)*Nb
        Q = torch.zeros(size = (Nb,n,n),device=device)
        Ru = torch.zeros(size = (Nb,(n-k),n),device=device)
        R22 = torch.zeros(size = (Nb,k,k),device=device)
        for i in range(Nb):
            D = Dtot[i,:,:]@Wtot[i,:,:]
            [Qloc,R]   = tla.qr(D,mode = 'reduced')
            Q[i,:,:]           = Qloc
            Ru[i,:,:]  = R[:n-k,:]
            R22[i,:,:]          = R[:,n-k:][n-k:,:]
    return Q,Ru,R22,NN

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
        W12 = None
        NN = Ru.shape[1]

    else:
        tic = time.time()
        k = Vtot.shape[2]
        n = Vtot.shape[1]
        NN = (n-k)*Nb
        tinit = time.time()-tic
        tic = time.time()
        if n>k:
            W1 = block_qr_tens(Vtot,device)
        else:
            W1 = torch.zeros(size=(Nb,n,0),device=device)
        tVc+=time.time()-tic
        tic = time.time()
        W12 = torch.cat((W1,Vtot),axis=2)
        Q,R = block_Q_and_R_tens(W12,Dtot,device)
        tQ += time.time()-tic
        tic = time.time()
        Ru = R[:,:n-k,:]
        R22 = R[:,n-k:,n-k:]
        tmv += time.time()-tic
    #print("tVc//tQ//tmv//tinit = ",tVc,"//",tQ,"//",tmv,"//",tinit)
    return Q,W12,Ru,R22,NN
def sparse_block_mult_tens(A,B,device,mode='N'):
    
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
        C = torch.zeros(size = (NbB,fac*na,kb),device=device)
        
        #startA=0
        for i in range(NbB):
            Asub = torch.zeros(size = (fac*na,fac*ka),device=device)
            for j in range(fac):
                Asub[j*na:(j+1)*na,:][:,j*ka:(j+1)*ka] = A[fac*i+j,:,:]#startA+j*na:startA+(j+1)*na,:]
            C[i,:,:] = Asub@B[i,:,:]
            #startA+=fac*na
    elif mode=='T':
        # this assumes NbB=NbA
        C = torch.zeros(size = (NbA,ka,kb),device=device)
        for i in range(NbA):
            C[i,:,:] = A[i,:,:].T@B[i,:,:]

    else:
        raise(ValueError("mode not recognized"))


    return C


def block_diag_add_tens(A,B,device):
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

def apply_sparse_block_tens(A,B,device,mode='N'):
    """C[i] = A[i] @ B[i]  (mode='N')  or  A[i].T @ B[i]  (mode='T'),
    with A (Nb, n, k) and B stored flat as (Nb*kB, nrhs).

    The loop form issued 2*Nb kernel launches per call -- one gemm plus one
    slice-assign into a zeroed buffer -- each on a matrix far too small to
    fill the device.  torch.bmm dispatches the whole batch as a single
    cublasDgemmStridedBatched.  Same transformation as
    HBStorch.block_matvec, which this now mirrors."""
    Nb   = A.shape[0]
    n    = A.shape[1]
    k    = A.shape[2]
    nrhs = B.shape[1]
    nB   = k if mode == 'N' else n
    if B.shape[0] != Nb * nB:
        raise ValueError(
            f"apply_sparse_block_tens: mode={mode!r} expects B with "
            f"{Nb*nB} rows (Nb={Nb} x {nB}), got {B.shape[0]}")
    Bm = B.reshape(Nb, nB, nrhs)
    if mode == 'N':
        C = torch.bmm(A, Bm)                      # (Nb, n, nrhs)
    elif mode == 'T':
        C = torch.bmm(A.mT, Bm)                   # (Nb, k, nrhs)
    else:
        raise ValueError("mode not recognized")
    return C.reshape(Nb * C.shape[1], nrhs)

def block_solve_tens(A,B,device,mode='N'):
    """Solve A[i] X[i] = B[i]  (mode='N')  or  A[i]^T X[i] = B[i]  (mode='T')
    for upper triangular A of shape (Nb, m, m), B stored flat as (Nb*m, nrhs).

    Every caller passes R or its leading (n-k)x(n-k) block, both exactly upper
    triangular from Householder QR.  One batched trsm replaces a Python loop
    of Nb pivoted LU factorizations, each of which also synchronized on its
    info check.  No singularity check: a zero pivot yields inf/nan instead of
    the exception torch.linalg.solve would raise."""
    Nb, m = A.shape[0], A.shape[1]
    nrhs  = B.shape[1]
    if B.shape[0] != Nb * m:
        raise ValueError(f"block_solve_tens: expected {Nb*m} rows, got {B.shape[0]}")
    if m == 0:
        return B.new_empty((0, nrhs))
    Bm = B.reshape(Nb, m, nrhs)
    if mode == 'N':
        X = torch.linalg.solve_triangular(A, Bm, upper=True)
    elif mode == 'T':
        X = torch.linalg.solve_triangular(A.mT, Bm, upper=False)
    else:
        raise ValueError("Mode not recognized")
    return X.reshape(Nb * m, nrhs)


def _W_parts(Wlist, Vmats, i, n, k):
    """(W1, V) at level i.  Wlist[i] is either the complement W1 alone
    (n x (n-k), current HBSMAT) or the legacy concatenation [W1 | V]."""
    W = Wlist[i]
    if W.shape[2] == n:
        return W[:, :, :n-k], W[:, :, n-k:]
    if Vmats is None:
        raise ValueError("Wlist holds only the complement W1; pass Vmats to solve()")
    return W, Vmats[i]



def compute_ULV(Utens,Dtens,Vtens,Nbvec,device):
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
    Wlist = []
    Uulist  = []
    Rlist = []
    Qlist = []
    for i in range(len(Dtens)):
        
        if i==0:
            Rprime = Dtens[0]
            tic = time.time()
            Q,W,Ru,R_22,NN = compute_QRW_sparse(Rprime,Vtens[0],Nbvec[0],device)
            n = Vtens[0].shape[1]
            k = Vtens[0].shape[2]
        else:
            Rhat = sparse_block_mult_tens(Uhat,Dtens[i],device)
            Rhat = block_diag_add_tens(Rhat,R_22,device)
            
            if i<len(Vtens):
                Q,W,Ru,R_22,NN = compute_QRW_sparse(Rhat,Vtens[i],Nbvec[i],device)
                n = Vtens[i].shape[1]
                k = Vtens[i].shape[2]
            else:
                tic = time.time()
                Q,W,Ru,R_22,NN = compute_QRW_sparse(Rhat,None,Nbvec[i],device)
        NNvec += [NNvec[-1]+NN]

        tic = time.time()
        if i<len(Utens):
            Wlist+=[W[:, :, :n-k].contiguous()]    # V half duplicates Vtens[i]
            if i == 0:
                Uu = sparse_block_mult_tens(Q[:,:,:(n-k)],Utens[0],device,mode='T')
                Ud = sparse_block_mult_tens(Q[:,:,(n-k):],Utens[0],device,mode='T')
                Uulist+=[Uu]
                Uhat=Ud.to(device)
                
            else:
                Uhat = sparse_block_mult_tens(Uhat,Utens[i],device)
                Uu = sparse_block_mult_tens(Q[:,:,:(n-k)],Uhat,device,mode='T')
                Ud = sparse_block_mult_tens(Q[:,:,(n-k):],Uhat,device,mode='T')
                Uulist+=[Uu]
                Uhat=Ud
        Rlist+=[Ru]
        Qlist+=[Q]
        
        
    return Qlist,Wlist,Uulist,Rlist,NNvec

def solve(Umats,Dmats,Qlist,Wlist,Uulist,Rlist,NNvec,rhs,device,mode='N',Vmats=None):
    """Apply the inverse (mode='N') or inverse transpose (mode='T') of the
    HBS operator from its ULV factors.

    Nothing is written into rhs or into any shared buffer, so there are no
    defensive clones.  The sweeps carry the not-yet-eliminated part as a
    running tensor instead of overwriting slices of a full-length copy, which
    is what previously forced a clone of the whole tail at every level.

    Vmats is required when Wlist stores only the complement W1 (what
    HBSMAT builds); with legacy [W1 | V] lists it may be omitted.
    NNvec is no longer needed and kept only for signature compatibility.
    """
    if mode not in ('N', 'T'):
        raise NotImplementedError("mode not recognized")
    L = len(Dmats)
    was_vector = (rhs.ndim == 1)
    r = rhs[:, None] if was_vector else rhs
    r = r.to(device)                    # no copy if already there; never written
    nrhs = r.shape[1]

    if mode == 'N':
        # ---- Q^* sweep: one bmm per level, peel off the eliminated rows ----
        chat = []
        x = r
        for i in range(L):
            Q = Qlist[i]
            Nb, n = Q.shape[0], Q.shape[1]
            C = torch.bmm(Q.mT, x.reshape(Nb, n, nrhs))        # (Nb, n, nrhs)
            if i < L - 1:
                k = Umats[i].shape[2]
                chat.append(C[:, :n-k, :].reshape(-1, nrhs))
                x = C[:, n-k:, :].reshape(-1, nrhs)
            else:
                chat.append(C.reshape(-1, nrhs))

        # ---- back substitution; x holds the solution in level-(i+1) coords
        x = block_solve_tens(Rlist[L-1], chat[L-1], device)
        if L > 1:
            v = apply_sparse_block_tens(Dmats[L-1], x, device)
        for i in range(L-2, -1, -1):
            Nb, n, k = Umats[i].shape
            W1, V = _W_parts(Wlist, Vmats, i, n, k)
            rhs0 = chat[i] \
                 - apply_sparse_block_tens(Uulist[i], v, device) \
                 - apply_sparse_block_tens(Rlist[i][:, :, n-k:], x, device)
            yi = block_solve_tens(Rlist[i][:, :, :n-k], rhs0, device)
            x  = apply_sparse_block_tens(W1, yi, device) \
               + apply_sparse_block_tens(V, x, device)
            if i > 0:      # v is only consumed by the next (finer) level
                v = apply_sparse_block_tens(Umats[i], v, device) \
                  + apply_sparse_block_tens(Dmats[i], x, device)

    else:
        # ---- forward sweep: W^* and R^{-*}, level by level ------------------
        ys = []
        v  = None                        # U_0^T v with v = 0 on the first level
        for i in range(L-1):
            Nb, n, k = Umats[i].shape
            W1, V = _W_parts(Wlist, Vmats, i, n, k)
            r1 = apply_sparse_block_tens(W1, r, device, mode='T')
            r2 = apply_sparse_block_tens(V,  r, device, mode='T')
            yi = block_solve_tens(Rlist[i][:, :, :n-k], r1, device, mode='T')
            ys.append(yi)
            v_new = apply_sparse_block_tens(Uulist[i], yi, device, mode='T')
            if v is not None:
                v_new = v_new + apply_sparse_block_tens(Umats[i], v, device, mode='T')
            v = v_new
            r = r2 \
              - apply_sparse_block_tens(Rlist[i][:, :, n-k:], yi, device, mode='T') \
              - apply_sparse_block_tens(Dmats[i+1], v, device, mode='T')
        x = block_solve_tens(Rlist[L-1], r, device, mode='T')

        # ---- Q sweep back down: one bmm per level ---------------------------
        for i in range(L-1, -1, -1):
            Q = Qlist[i]
            if i == L - 1:
                x = apply_sparse_block_tens(Q, x, device)
            else:
                Nb, n = Q.shape[0], Q.shape[1]
                k = Umats[i].shape[2]
                z = torch.cat((ys[i].reshape(Nb, n-k, nrhs),
                               x.reshape(Nb, k, nrhs)), dim=1)
                x = torch.bmm(Q, z).reshape(-1, nrhs)

    return x[:, 0] if was_vector else x
