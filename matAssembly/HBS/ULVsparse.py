
import numpy as np
import matplotlib.pyplot as plt



'''
Factorize Q^* S W = R
with R upper triangular 
(I don't know why the term 'ULV' is used and at this point I'm too afraid to ask)

Q,R,W given in reduced format ('Compound Block Diagonal',cbd)

'''



'''

proposed cbd class definition??

class cbd:
    def __init__(self,A,NNvec,Nbvec,dir = 1):
        # dir = 0 -> vertically stacked
        # dir = 1 -> horizontally stacked
        self.dir = dir
        self.NNvec = NNvec
        self.A = A
        self.Nbvec = Nbvec
        sz2 = 0
        for ind in range(len(NNvec)-1):
            k = (self.NNvec[ind+1]-self.NNvec[ind])
            sz2 += k*Nbvec[ind]
        self.sz = [A.shape[0],sz2]
    
    def apply_to_dense(self,B,mode='N'):
        
        if self.dir == 1:
            if mode=='N':
                C = np.zeros(shape=(self.sz[0],B.shape[1]))
                start = 0
                stop = 0
                for ind in range(len(self.NNvec)-1):
                    Asub = self.A[:,self.NNvec[ind]:self.NNvec[ind+1]]
                    Nb = self.Nbvec[ind]
                    k = (self.NNvec[ind+1]-self.NNvec[ind])
                    sz = k*Nb
                    stop = stop+sz
                    n = self.A.shape[0]//Nb
                    Bsub = B[start:stop,:]
                    for i in range(Nb):
                        C[i*n:(i+1)*n,:] += Asub[i*n:(i+1)*n,:]@Bsub[i*k:(i+1)*k,:]
                    start = stop
            elif mode=='T':
                C = np.zeros(shape=(self.sz[1],B.shape[1]))
                start = 0
                stop = 0
                for ind in range(len(self.NNvec)-1):
                    Asub = self.A[:,self.NNvec[ind]:self.NNvec[ind+1]]
                    Nb = self.Nbvec[ind]
                    k = (self.NNvec[ind+1]-self.NNvec[ind])
                    sz = k*Nb
                    stop = stop+sz
                    n = self.A.shape[0]//Nb
                    for i in range(Nb):
                        C[start+i*k:start+(i+1)*k,:] = Asub[i*n:(i+1)*n,:].T@B[i*n:(i+1)*n,:]
                    start = stop
            else:
                raise ValueError("mode not recognized")
        else:
            if mode=='N':
                C = np.zeros(shape=(self.sz[0],B.shape[1]))
                start = 0
                stop = 0
                for ind in range(len(self.NNvec)-1):
                    Asub = self.A[:,self.NNvec[ind]:self.NNvec[ind+1]]
                    Nb = self.Nbvec[ind]
                    k = (self.NNvec[ind+1]-self.NNvec[ind])
                    sz = k*Nb
                    stop = stop+sz
                    n = self.A.shape[0]//Nb
                    Bsub = B[start:stop,:]
                    for i in range(Nb):
                        C[i*n:(i+1)*n,:] += Asub[i*n:(i+1)*n,:]@Bsub[i*k:(i+1)*k,:]
                    start = stop
            elif mode=='T':
                C = np.zeros(shape=(self.sz[1],B.shape[1]))
                start = 0
                stop = 0
                for ind in range(len(self.NNvec)-1):
                    Asub = self.A[:,self.NNvec[ind]:self.NNvec[ind+1]]
                    Nb = self.Nbvec[ind]
                    k = (self.NNvec[ind+1]-self.NNvec[ind])
                    sz = k*Nb
                    stop = stop+sz
                    n = self.A.shape[0]//Nb
                    for i in range(Nb):
                        C[start+i*k:start+(i+1)*k,:] = Asub[i*n:(i+1)*n,:].T@B[i*n:(i+1)*n,:]
                    start = stop

    
        return C
'''      
    



def compute_QRW_sparse(Dtot,Vtot,Nb):
    
    '''
    
    Given (repr. of) R_{ell}' and V_{ell}, compute Q_{ell}, W_{ell} and upper triangular matrix
    R such that Q_{ell}^* R_{ell}'W_{ell} is upper triangular

    '''
    

    if Nb==1:
        D = Dtot
        [Q,R] = np.linalg.qr(D)
        Q1 = Q
        Q2 = 0
        
        R11 = R
        
        R12=0
        R22=0
        W1 = np.identity(R.shape[1])
        W2 = 0
        NN = R.shape[0]
    else:
        k = Vtot.shape[1]
        n = Vtot.shape[0]//Nb
        NN = (n-k)*Nb
        Q1 = np.zeros(shape = (Nb*n,n-k))
        Q2 = np.zeros(shape = (Nb*n,k))
        W1 = np.zeros(shape = (Nb*n,n-k))
        W2 = np.zeros(shape = (Nb*n,k))
        R11 = np.zeros(shape = (NN,n-k))
        R12 = np.zeros(shape = (NN,k))
        R22 = np.zeros(shape = (Nb*k,k))
        for box_ind in range(Nb):
            V       = Vtot[box_ind*n:(box_ind+1)*n,:]
            [_,_,Ur] = np.linalg.svd(V.T)#svd needed here, otherwise accuracy not guaranteed
            Vr = Ur.T
            Vr = Vr[:,k:]
            W       = np.append(Vr,V,axis=1)
            D       = Dtot[box_ind*n:(box_ind+1)*n,:]
            [Q,R]   = np.linalg.qr(D@W)
            
            Q1[box_ind*n:(box_ind+1)*n,:]           = Q[:,:n-k]
            Q2[box_ind*n:(box_ind+1)*n,:]           = Q[:,n-k:]
            W1[box_ind*n:(box_ind+1)*n,:]           = W[:,:n-k]
            W2[box_ind*n:(box_ind+1)*n,:]           = W[:,n-k:]
            R11[box_ind*(n-k):(box_ind+1)*(n-k),:]  = R[:,:n-k][:n-k,:]
            R12[box_ind*(n-k):(box_ind+1)*(n-k),:]  = R[:,n-k:][:n-k,:]
            R22[box_ind*k:(box_ind+1)*k,:]          = R[:,n-k:][n-k:,:]
    return Q1,Q2,W1,W2,R11,R12,R22,NN
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
                Asub = A[i*na:(i+1)*na,:].copy()
                for j in range(fac):
                    Bsub[j*nb:(j+1)*nb,:][:,j*kb:(j+1)*kb] = B[i*nb+j*nb:i*nb+(j+1)*nb,:].copy()
                C[i*na:(i+1)*na,:] = Asub@Bsub
        else:
            fac = (NbA//NbB)
            startA=0
            for i in range(NbB):
                Asub = np.zeros(shape = (fac*na,fac*ka))
                Bsub = B[i*nb:(i+1)*nb,:].copy()
                for j in range(fac):
                    Asub[j*na:(j+1)*na,:][:,j*ka:(j+1)*ka] = A[startA+j*na:startA+(j+1)*na,:].copy()
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
            Asub = A[i*na:(i+1)*na,:].copy()
            for j in range(fac):
                Bsub[j*nb:(j+1)*nb,:][:,j*kb:(j+1)*kb] = B[i*na+j*nb:i*na+(j+1)*nb,:].copy()
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
        C=A.copy()
        for i in range(NbA):
            for j in range(fac):
                C[i*nA+j*nB:i*nA+(j+1)*nB,:][:,j*kB:(j+1)*kB]+=B[i*nA+j*nB:i*nA+(j+1)*nB,:]
    else:
        raise(ValueError("put smol frist"))
    return C

def update_U(Q1,Q2,U,Umat,NNvec,Nb,fac=4):

    '''
    Update U matrix
    U <- blkdiag(I,Q1,Q2)@U@Umat


    INPUT  
        Q1,Q2   :   Block diagonal matrices (reduced form)
        U       :   cbd matrix
        Umat    :   Block diagonal matrix   (reduced form)
    OUTPUT
        U       :   blkdiag(I,Q1,Q2)@U@Umat
    '''
    
    U_tmp = np.zeros(shape=(U.shape[0],Umat.shape[1]))
    for i in range(len(NNvec)-1):
        U_tmp[NNvec[i]:NNvec[i+1],:] = sparse_block_mult(U[NNvec[i]:NNvec[i+1],:],Umat,fac*Nb,Nb)
    U_tmp[NNvec[-1]:,:] = sparse_block_mult(U[NNvec[-1]:,:],Umat,fac*Nb,Nb)
    U = U_tmp.copy()
    NNtot = NNvec[-1]
    NNu = Q1.shape[1]*Nb
    U[NNtot:NNtot+NNu,:] = sparse_block_mult(Q1,U_tmp[NNtot:,:],Nb,Nb,mode='T')
    U[NNtot+NNu:,:] = sparse_block_mult(Q2,U_tmp[NNtot:,:],Nb,Nb,mode='T')
    
    return U
def compute_Rr(Rprime,R12,R22,W2,NNvec,Nb):
    Rr = np.zeros(shape = (Rprime.shape[0],R12.shape[1]))
    for i in range(len(NNvec)-2):
        Rr[NNvec[i]:NNvec[i+1],:] = sparse_block_mult(Rprime[NNvec[i]:NNvec[i+1],:],W2,Nb,Nb)
    Rr[NNvec[-2]:NNvec[-1],:] = R12
    Rr[NNvec[-1]:,:] = R22
    return Rr
def compute_Rl(Rprime,W1,R11,NNvec,Nb):
    Rl = np.zeros(shape = (Rprime.shape[0],R11.shape[1]))
    for i in range(len(NNvec)-1):
        Rl[NNvec[i]:NNvec[i+1],:] = sparse_block_mult(Rprime[NNvec[i]:NNvec[i+1],:],W1,Nb,Nb)
    Rl[NNvec[-2]:NNvec[-1],:] = R11
    return Rl

def compute_Rprime(U,D,Rr,NNvec,Nb,fac=4):
    Rprime                           = np.zeros(shape = (U.shape[0],D.shape[1]))
    for i in range(len(NNvec)-1):
        Rprime[NNvec[i]:NNvec[i+1],:] = sparse_block_mult(U[NNvec[i]:NNvec[i+1],:],D,fac*Nb,Nb)
        Rprime[NNvec[i]:NNvec[i+1],:] = block_diag_add(Rprime[NNvec[i]:NNvec[i+1],:],Rr[NNvec[i]:NNvec[i+1],:],Nb,fac*Nb)
    
    Rprime[NNvec[-1]:,:]            = sparse_block_mult(U[NNvec[-1]:,:],D,fac*Nb,Nb)
    Rprime[NNvec[-1]:,:]            = block_diag_add(Rprime[NNvec[-1]:,:],Rr[NNvec[-1]:,:],Nb,fac*Nb)

    return Rprime

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
def apply_cbd(Q,A,Nbvec,NNvec,NNQvec,mode='N'):
    '''
    apply compound block diag matrix
    '''
    if mode == 'N':
        B = np.zeros(shape = (Q.shape[0],A.shape[1]))
        for i in range(len(NNQvec)-1):
            print("i = ",i)
            B+= apply_sparse_block(Q[:,NNQvec[i]:NNQvec[i+1]],A[NNvec[i]:NNvec[i+1],:],Nbvec[i])
    elif mode=='T':
        B = np.zeros(shape = (NNvec[-1],A.shape[1]))
        for i in range(len(NNQvec)-1):
            print("i = ",i)
            B[NNvec[i]:NNvec[i+1],:]= apply_sparse_block(Q[:,NNQvec[i]:NNQvec[i+1]],A,Nbvec[i],mode='T')
    else:
        raise ValueError("mode not recognized")
    
    
    return B
def apply_Rblock(R,A,lvl,Nbvec,NNvec,NNRvec):
    Rsub = R[:,NNRvec[lvl]:NNRvec[lvl+1]]
    B =np.zeros(shape = (Rsub.shape[0],A.shape[1]))
    NNsub = NNvec[:lvl+2]
    for i in range(len(NNsub)-2):
        B[NNsub[i]:NNsub[i+1],:] = apply_sparse_block(Rsub[NNsub[i]:NNsub[i+1],:],A,Nbvec[lvl])
    B[NNsub[-2]:NNsub[-1],:] = apply_sparse_block(Rsub[NNsub[-2]:NNsub[-1],:],A,Nbvec[lvl])
    return B

def apply_Rblock_off_diag(R,A,lvl,Nbvec,NNvec,NNRvec,mode='N'):
    if mode=='N':
        Rsub = R[:,NNRvec[lvl]:NNRvec[lvl+1]]
        B =np.zeros(shape = (Rsub.shape[0],A.shape[1]))
        NNsub = NNvec[:lvl+2]
        for i in range(len(NNsub)-2):
            B[NNsub[i]:NNsub[i+1],:] = apply_sparse_block(Rsub[NNsub[i]:NNsub[i+1],:],A,Nbvec[lvl])
    elif mode=='T':
        Rsub = R[:,NNRvec[lvl]:NNRvec[lvl+1]]
        B =np.zeros(shape = (Rsub.shape[1]*Nbvec[lvl],A.shape[1]))
        NNsub = NNvec[:lvl+2]
        for i in range(len(NNsub)-2):
            B += apply_sparse_block(Rsub[NNsub[i]:NNsub[i+1],:],A,Nbvec[lvl],mode='T')
    else:
        raise ValueError("mode not recognized")
    return B
def block_solve(A,B,Nb):
    
    n = A.shape[0]//Nb
    C = np.zeros(shape = (A.shape[0],B.shape[1]))
    for i in range(Nb):
        C[i*n:(i+1)*n,:] = np.linalg.solve(A[i*n:(i+1)*n,:],B[i*n:(i+1)*n,:])
    return C

def solve_R(R,RHS,Nbvec,NNvec,NNRvec,mode='N'):
    if mode=='N':
        if RHS.ndim==1:
            RHS0 = RHS[:,np.newaxis]
        else:
            RHS0 = RHS
        u = np.zeros(shape = (NNvec[-1],RHS.shape[1]))
        u[NNvec[-2]:NNvec[-1],:]=block_solve(R[NNvec[-2]:NNvec[-1],:][:,NNRvec[-2]:NNRvec[-1]],RHS0[NNvec[-2]:NNvec[-1],:],Nbvec[-1])
        lvl = len(NNvec)-2
        if lvl>0:
            RHS00 = (RHS0 - apply_Rblock_off_diag(R,u[NNvec[-2]:NNvec[-1],:],lvl,Nbvec,NNvec,NNRvec)).copy()
            u[:NNvec[-2],:] = solve_R(R[:,:NNRvec[-2]],RHS00,Nbvec[:lvl],NNvec[:lvl+1],NNRvec[:lvl+1])
    elif mode == 'T':
        if RHS.ndim==1:
            RHS0 = RHS[:,np.newaxis]
        else:
            RHS0 = RHS
        
        u = np.zeros(shape = (NNvec[1]-NNvec[0],RHS.shape[1]))
        u[NNvec[0]:NNvec[1],:]=block_solve(R[NNvec[0]:NNvec[1],:][:,NNRvec[0]:NNRvec[1]],RHS0[NNvec[0]:NNvec[1],:],Nbvec[0])
        lvl = len(NNvec)-2
        print("lvl = ",lvl)
        if lvl>0:
            RHS00 = RHS0.copy()
            rhat = apply_Rblock_off_diag(R,u[NNvec[0]:NNvec[1],:],lvl,Nbvec,NNvec,NNRvec,mode='T')
            print("rhat shape = ",rhat.shape)
            print("NNvec = ",NNvec)
            RHS00[NNvec[1]:NNvec[2],:] -= rhat
            u[NNvec[1]:,:] = solve_R(R[:,NNRvec[1]:],RHS00,Nbvec[:lvl],NNvec[:lvl+1],NNRvec[:lvl+1],mode='T')
    else:
        raise ValueError("mode not recognized")
    return u



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
    
    Rtot = np.zeros(shape = (Dmats[0].shape[0],0))
    Qtot = np.zeros(shape=(Dmats[0].shape[0],0))
    Wtot = np.zeros(shape=(Dmats[0].shape[0],0))

    NNvec = np.zeros(shape=(0,),dtype=np.int64)
    NNvec = np.append(NNvec,0)
    NNQvec = np.zeros(shape=(0,),dtype=np.int64)
    NNQvec = np.append(NNQvec,0)
    NNRvec = np.zeros(shape=(0,),dtype=np.int64)
    NNRvec = np.append(NNRvec,0)
    NNWvec = np.zeros(shape=(0,),dtype=np.int64)
    NNWvec = np.append(NNWvec,0)
    fac = Nbvec[-2]//Nbvec[-1]
    print("fac = ",fac)
    for i in range(len(Dmats)):
        if i==0:
            Rprime = Dmats[0]
            Q1,Q2,W1,W2,R11,R12,R22,NN = compute_QRW_sparse(Rprime,Vmats[0],Nbvec[0])
            print("W1 shape = ",W1.shape)
            print("W2 shape = ",W2.shape)
            U = np.zeros(shape = Umats[0].shape)
            U[:NN,:] = sparse_block_mult(Q1,Umats[0],Nbvec[0],Nbvec[0],mode='T')
            U[NN:,:] = sparse_block_mult(Q2,Umats[0],Nbvec[0],Nbvec[0],mode='T')
            QQ = Q1.copy()
            Qprev = Q2.copy()

            WW = W1.copy()
            Wprev = W2.copy()
        else:
            Rprime = compute_Rprime(U,Dmats[i],Rr,NNvec,Nbvec[i],fac)
            if i<len(Dmats)-1:
                Q1,Q2,W1,W2,R11,R12,R22,NN = compute_QRW_sparse(Rprime[NNvec[-1]:,:],Vmats[i],Nbvec[i])
            else:
                Q1,Q2,W1,W2,R11,R12,R22,NN = compute_QRW_sparse(Rprime[NNvec[-1]:,:],np.array(1.),Nbvec[i])
            QQ = sparse_block_mult(Qprev,Q1,Nbvec[i-1],Nbvec[i]).copy()
            WW = sparse_block_mult(Wprev,W1,Nbvec[i-1],Nbvec[i]).copy()
            if i<len(Dmats)-1:
                print("W1 shape = ",W1.shape)
                print("W2 shape = ",W2.shape)
                Qprev = sparse_block_mult(Qprev,Q2,Nbvec[i-1],Nbvec[i]).copy()
                Wprev = sparse_block_mult(Wprev,W2,Nbvec[i-1],Nbvec[i]).copy()
                U = update_U(Q1,Q2,U,Umats[i],NNvec,Nbvec[i],fac)
        
        NNvec = np.append(NNvec,NN+NNvec[-1])
        Qtot = np.append(Qtot,QQ,axis=1)
        NNQvec=np.append(NNQvec,NNQvec[-1]+QQ.shape[1])
        
        Wtot = np.append(Wtot,WW,axis=1)
        NNWvec=np.append(NNWvec,NNWvec[-1]+WW.shape[1])
        
        Rl = compute_Rl(Rprime,W1,R11,NNvec,Nbvec[i])

        Rtot = np.append(Rtot,Rl,axis=1)
        NNRvec=np.append(NNRvec,NNRvec[-1]+Rl.shape[1])
        if i<len(Dmats)-1:
            Rr = compute_Rr(Rprime,R12,R22,W2,NNvec,Nbvec[i])
        
    return Qtot,Rtot,Wtot,NNvec,NNQvec,NNRvec,NNWvec

