import numpy as np
import scipy.linalg as splinalg
import time
import matAssembly.HBS.ULVsparse as ULVsparse

#sparse block matrix operations

def block_col(A,rk,Nb):
    B = np.zeros(shape = (A.shape[0],rk))
    n = A.shape[0]//Nb
    for i in range(Nb):
        [U,_,_] = np.linalg.svd(A[i*n:(i+1)*n,:])
        #U,_ = np.linalg.qr(A[i*n:(i+1)*n,:],mode='reduced')
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
        k = sum(s>s[0]*1e-14)
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
def construct_D(U_ell,V_ell,YO,ZP,Nb):
    C = np.zeros(shape = (U_ell.shape[0],YO.shape[1]))
    n = U_ell.shape[0]//Nb
    for i in range(Nb):
        Usub = U_ell[i*n:(i+1)*n,:]
        Vsub = V_ell[i*n:(i+1)*n,:]
        YOsub = YO[i*n:(i+1)*n,:]
        ZPsub = ZP[i*n:(i+1)*n,:]

        C[i*n:(i+1)*n,:] = YOsub-Usub@Usub.T@YOsub\
                            +Usub@(Usub.T@((ZPsub-Vsub@(Vsub.T@ZPsub)).T))
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

    def __init__(self,A,tree,quad=True):
        self.Umats  =   []
        self.Vmats  =   []
        self.Dmats  =   []
        self.nbytes =   0
        self.A      =   A
        self.shape  =   self.A.shape
        self.perm   =   tree.perm_leaf
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
        self.Nb = tree.nleaves
        self.nl = self.A.shape[0]//self.Nb
        self.L = tree.nlevels
        self.Nbvec = []
        if quad:
            self.fac = 4
        else:
            self.fac = 2
    
    def construct(self,rk):
        # we compute an HBS compression of permuted op
        
        tic = time.time()
        Om = np.random.standard_normal(size = (self.shape[1],(self.fac+2)*rk+10))
        Psi = np.random.standard_normal(size = (self.shape[0],(self.fac+2)*rk+10))
        Omprime = np.zeros(shape = Om.shape)
        Omprime[self.perm,:] = Om
        Psiprime = np.zeros(shape = Psi.shape)
        Psiprime[self.perm,:] = Psi
        Y = self.A@Omprime
        Z = self.A.T@Psiprime
        Y = Y[self.perm,:]
        Z = Z[self.perm,:]
        Nb = self.Nb
        nl = self.nl
        self.setupTime+=time.time()-tic
        for lvl in range(self.L-1,-1,-1):
            
            if lvl == self.L-1:
                Om_ell      = Om
                Psi_ell     = Psi
                Y_ell       = Y
                Z_ell       = Z
                rkm = min(rk,nl)
            else:
                
                Y_ell       -=block_mult(D_ell,Om_ell,Nb)
                Y_ell       = block_mult(U_ell,Y_ell,Nb,mode='T')

                Z_ell       -= block_mult(D_ell,Psi_ell,Nb,mode='T')
                Z_ell       = block_mult(V_ell,Z_ell,Nb,mode='T')
                
                Om_ell      = block_mult(V_ell,Om_ell,Nb,mode='T')
                Psi_ell     = block_mult(U_ell,Psi_ell,Nb,mode='T')

                
                Nb = Nb//self.fac
                rkm = rk
            print("lvl//Nb = ",lvl,"//",Nb)
            
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
                D_ell = construct_D(U_ell,V_ell,YO,ZP,Nb)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,Nb)
                self.blockSolveTime+=time.time()-tic
                self.Dmats+=[D_ell]
            self.Nbvec+=[Nb]

    def matvec(self,v,mode='N'):
        if v.ndim==1:
            vperm = v[self.perm,np.newaxis]    
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
        if mode=='N':
            VV+=[vperm]
            for lvl in range(len(self.Vmats)):
                v_lvl = block_mult(self.Vmats[lvl],VV[lvl],Nb,mode='T')
                VV+=[v_lvl]
                Nb=Nb//self.fac
            uperm = block_mult(self.Dmats[-1],VV[-1],Nb)
            for lvl in range(len(self.Umats)-1,-1,-1):
                uperm = block_mult(self.Umats[lvl],uperm,self.fac*Nb)+ block_mult(self.Dmats[lvl],VV[lvl],self.fac*Nb)
                Nb=Nb*self.fac
            u = np.zeros(shape = uperm.shape)
            u[self.perm,:] = uperm

        elif mode=='T':
            VV+=[vperm]
            for lvl in range(len(self.Umats)):
                v_lvl = block_mult(self.Umats[lvl],VV[lvl],Nb,mode='T')
                VV+=[v_lvl]
                Nb=Nb//self.fac
            uperm = block_mult(self.Dmats[-1],VV[-1],Nb,mode='T')
            for lvl in range(len(self.Vmats)-1,-1,-1):
                uperm = block_mult(self.Vmats[lvl],uperm,self.fac*Nb)+ block_mult(self.Dmats[lvl],VV[lvl],self.fac*Nb,mode='T')
                Nb=Nb*self.fac    
            u = np.zeros(shape = uperm.shape)
            u[self.perm,:] = uperm
        else:
            raise ValueError("mode not recognized")
        if v.ndim==1:
            u = u.flatten()
        return u
    
    def compute_ULV(self):
        self.Qtot,self.Rtot,self.Wtot,self.NNvec,self.NNQvec,self.NNRvec,self.NNWvec = ULVsparse.compute_ULV(self.Umats,self.Dmats,self.Vmats,self.Nbvec)

    def solve_ULV(self,b):
        if b.ndim==1:
            bperm = b[self.perm,np.newaxis]    
        else:
            bperm= b[self.perm,:]
        rhs = ULVsparse.apply_cbd(self.Qtot,bperm,self.Nbvec,self.NNvec,self.NNQvec,mode='T')
        uhat = ULVsparse.solve_R(self.Rtot,rhs,self.Nbvec,self.NNvec,self.NNRvec)
        uperm = ULVsparse.apply_cbd(self.Wtot,uhat,self.Nbvec,self.NNvec,self.NNQvec)
        u = np.zeros(shape = uperm.shape)
        u[self.perm,:] = uperm
        if b.ndim==1:
            u = u.flatten()
        return u
