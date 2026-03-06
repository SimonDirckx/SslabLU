import numpy as np
import scipy.linalg as splinalg
import time
def block_col(A,rk,Nb):
    B = np.zeros(shape = (A.shape[0],rk))
    n = A.shape[0]//Nb
    for i in range(Nb):
        #[U,_,_] = np.linalg.svd(A[i*n:(i+1)*n,:])
        U,_ = np.linalg.qr(A[i*n:(i+1)*n,:],mode='reduced')
        B[i*n:(i+1)*n,:] = U[:,:rk]
    return B

def block_null(A,rk,Nb):
    nA = A.shape[0]//Nb
    kA = A.shape[1]
    B = np.zeros(shape = (kA*Nb,rk))
    for i in range(Nb):
        Q,_ = np.linalg.qr(A[i*nA:(i+1)*nA,:].T,mode='reduced')
        Om = np.random.standard_normal(size = (Q.shape[0],rk))
        Om-=Q@(Q.T@Om)
        V,_ = np.linalg.qr(Om,mode='reduced')
        nV = V.shape[1]
        B[i*kA:(i+1)*kA,:] = V[:,nV-rk:]
    return B

def block_solve_r(A,B,Nb):
    #compute A_tau/B_tau per block
    kA = A.shape[1]
    nb = B.shape[0]//Nb
    n = A.shape[0]//Nb
    C = np.zeros(shape = (A.shape[0],nb))
    for i in range(Nb):
        [U,s,Vh] = np.linalg.svd(B[i*nb:(i+1)*nb,:],full_matrices=False)
        k = sum(s>s[0]*1e-8)
        Vh = Vh[:k,:].T
        C[i*n:(i+1)*n,:] = ((A[i*n:(i+1)*n,:]@Vh)/s[:k])@U[:,:k].T
    return C

def block_orth_proj(A,B,Nb,compl=True):
    # assumes blocks of A orth
    C = np.zeros(shape = (A.shape[0],B.shape[1]))
    n = A.shape[0]//Nb
    for i in range(Nb):
        U = A[i*n:(i+1)*n,:]
        
        if compl:
            C[i*n:(i+1)*n,:] = B[i*n:(i+1)*n,:]-U@(U.T@B[i*n:(i+1)*n,:])
        else:
            C[i*n:(i+1)*n,:] = U@(U.T@B[i*n:(i+1)*n,:])
    return C


    return 0
def block_transpose(A,Nb):
    
    #note: this is one of ONLY TWO places where uniformity of B ito blocks is assumed!
    kA = A.shape[1]
    nA = A.shape[0]//Nb
    AT = np.zeros(shape = (Nb*kA,nA))
    for i in range(Nb):
        AT[i*kA:(i+1)*kA,:] = A[i*nA:(i+1)*nA,:].T
    return AT
def block_mult(A,B,Nb,mode='N'):
    if mode=='N':
        C = np.zeros(shape=(A.shape[0],B.shape[1]))
        kA = A.shape[1]
        n = A.shape[0]//Nb
        for i in range(Nb):
            C[i*n:(i+1)*n,:]=A[i*n:(i+1)*n,:]@B[i*kA:(i+1)*kA,:]
    elif mode=='T':
        kA = A.shape[1]
        C = np.zeros(shape=(kA*Nb,B.shape[1]))        
        n = A.shape[0]//Nb
        for i in range(Nb):
            C[i*kA:(i+1)*kA,:]=A[i*n:(i+1)*n,:].T@B[i*n:(i+1)*n,:]
    else:
        raise ValueError("mode not recognized")
    return C


class HBSMAT:
    """
    
    HBS mat in new framework

    @init:
            linear operator A
            tree on DOFS (symmetric)
            target rank k

    @constructs: 
            HBS approximation to the source-target map
    @implements:
            matvec (normal/transpose)

    """

    def __init__(self,A,tree):
        self.Umats  =   []
        self.Vmats  =   []
        self.Dmats  =   []
        self.nbytes =   0
        self.A      =   A
        self.shape  =   self.A.shape
        self.tree   = tree
        self.perm   =   np.zeros(shape = (0,),dtype=np.int64)
        for leaf in self.tree.get_leaves():
            self.perm = np.append(self.perm,self.tree.get_box_inds(leaf))
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
    
    def construct(self,rk):
        # we compute an HBS compression of permuted op
        
        tic = time.time()
        Om = np.random.standard_normal(size = (self.shape[1],6*rk+10))
        Psi = np.random.standard_normal(size = (self.shape[0],6*rk+10))
        Omprime = np.zeros(shape = Om.shape)
        Omprime[self.perm,:] = Om
        Psiprime = np.zeros(shape = Psi.shape)
        Psiprime[self.perm,:] = Psi
        Y = self.A@Omprime
        Z = self.A.T@Psiprime
        Y = Y[self.perm,:]
        Z = Z[self.perm,:]
        boxes = self.tree.get_leaves()
        Nb = len(boxes)
        nl = len(self.tree.get_box_inds(boxes[0]))
        self.setupTime+=time.time()-tic
        for lvl in range(self.tree.nlevels-1,-1,-1):

            if lvl == self.tree.nlevels-1:
                Om_ell      = Om
                Psi_ell     = Psi
                Y_ell       = Y
                Z_ell       = Z
                rkm = min(rk,nl)
            else:
                
                Y_ell       -=block_mult(D_ell,Om_ell,Nb)
                Y_ell       = block_mult(U_ell,Y_ell,Nb,mode='T')

                Z_ell       -= block_mult(D_ell,Psi_ell,Nb)
                Z_ell       = block_mult(V_ell,Z_ell,Nb,mode='T')


                Om_ell      = block_mult(V_ell,Om_ell,Nb,mode='T')
                Psi_ell     = block_mult(U_ell,Psi_ell,Nb,mode='T')
                Nb = Nb//4
                rkm = rk
            
            
            if lvl>0:
                tic = time.time()
                P_ell = block_null(Om_ell,rkm,Nb)
                Q_ell = block_null(Psi_ell,rkm,Nb)
                self.nullTime+=time.time()-tic
                YP = block_mult(Y_ell,P_ell,Nb)
                ZQ = block_mult(Z_ell,Q_ell,Nb)
                U_ell = block_col(YP,rkm,Nb)
                V_ell = block_col(ZQ,rkm,Nb)
                tic = time.time()
                YO = block_solve_r(Y_ell,Om_ell,Nb)
                ZP = block_solve_r(Z_ell,Psi_ell,Nb)
                self.blockSolveTime+=time.time()-tic
                tic = time.time()
                D_ell = block_orth_proj(U_ell,YO,Nb)
                Dtemp = block_orth_proj(V_ell,ZP,Nb)
                Dtemp = block_transpose(Dtemp,Nb)
                Dtemp = block_orth_proj(U_ell,Dtemp,Nb,compl=False)
                D_ell += Dtemp
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,Nb)
                self.blockSolveTime+=time.time()-tic
                self.Dmats+=[D_ell]
    def matvec(self,v,mode='N'):
        if v.ndim==1:
            vperm = v[self.perm,np.newaxis]    
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = len(self.tree.get_leaves())
        if mode=='N':
            VV+=[vperm]
            for lvl in range(len(self.Vmats)):
                v_lvl = block_mult(self.Vmats[lvl],VV[lvl],Nb,mode='T')
                VV+=[v_lvl]
                Nb=Nb//4
            uperm = block_mult(self.Dmats[-1],VV[-1],Nb)
            for lvl in range(len(self.Umats)-1,-1,-1):
                uperm = block_mult(self.Umats[lvl],uperm,4*Nb)+ block_mult(self.Dmats[lvl],VV[lvl],4*Nb)
                Nb=Nb*4
            u = np.zeros(shape = uperm.shape)
            u[self.perm,:] = uperm

        elif mode=='T':
            VV+=[vperm]
            for lvl in range(len(self.Umats)):
                v_lvl = block_mult(self.Umats[lvl],VV[lvl],Nb,mode='T')
                VV+=[v_lvl]
                Nb=Nb//4
            uperm = block_mult(self.Dmats[-1],VV[-1],Nb,mode='T')
            for lvl in range(len(self.Vmats)-1,-1,-1):
                uperm = block_mult(self.Vmats[lvl],uperm,4*Nb)+ block_mult(self.Dmats[lvl],VV[lvl],4*Nb,mode='T')
                Nb=Nb*4            
            u = np.zeros(shape = uperm.shape)
            u[self.perm,:] = uperm
        else:
            raise ValueError("mode not recognized")
        if v.ndim==1:
            u = u.flatten()
        return u
            
            
