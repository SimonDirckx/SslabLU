import numpy as np
import scipy.linalg as splinalg
import time
import matAssembly.HBS.ULVsparse as ULVsparse
import matAssembly.HBS.ULVsparse_torch as ULVsparse_torch
import torch.linalg as tla
import scipy.linalg as sclinalg
#sparse block matrix operations


def block_solve_r(A,B,Nb):
    #compute A_tau/B_tau per block
    kA = A.shape[1]
    nb = B.shape[0]//Nb
    n = A.shape[0]//Nb
    C = np.zeros(shape = (A.shape[0],nb))
    for i in range(Nb):
        C[i*n:(i+1)*n,:] = A[i*n:(i+1)*n,:]@np.linalg.pinv(B[i*nb:(i+1)*nb,:],rcond=1e-15)
    return C
def compute_UV(Om,Y,rk,Nb):
    nloc = Om.shape[0]//Nb
    U = np.zeros(shape = (Y.shape[0],rk))
    n = Om.shape[0]//Nb
    nU = U.shape[0]//Nb
    for i in range(Nb):
        Q = np.linalg.qr(Om[i*n:(i+1)*n,:].T,mode='complete')[0]
        U[i*nU:(i+1)*nU,:] = (np.linalg.svd(Y[i*nU:(i+1)*nU,:]@Q[:,-nloc:])[0])[:,:rk]
    return U



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
def construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell,Nb):
    C = np.zeros(shape = (U_ell.shape[0],Om_ell.shape[0]//Nb))
    n = U_ell.shape[0]//Nb
    for i in range(Nb):
        Usub = U_ell[i*n:(i+1)*n,:]
        Vsub = V_ell[i*n:(i+1)*n,:]
        Ysub = Y_ell[i*n:(i+1)*n,:]
        Zsub = Z_ell[i*n:(i+1)*n,:]
        Omsub = Om_ell[i*n:(i+1)*n,:]
        Psisub = Psi_ell[i*n:(i+1)*n,:]
        #YO = (Ysub@np.linalg.pinv(Omsub))
        #ZP = (Zsub@np.linalg.pinv(Psisub))
        C[i*n:(i+1)*n,:] = (Ysub-Usub@(Usub.T@Ysub))@np.linalg.pinv(Omsub)\
                            +Usub@(Usub.T@(((Zsub-Vsub@(Vsub.T@Zsub))@np.linalg.pinv(Psisub)).T))
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

    def __init__(self,A=None,tree=None,quad=True):
        self.Umats  =   []
        self.Vmats  =   []
        self.Dmats  =   []
        self.Qlist  =   []
        self.Rlist  =   []
        self.Wlist  =   []
        self.Uulist =   []

        
        self.mode   =   'N'
        if A is not None:
            self.A      =   A
            print("A shape = ",A.shape)
            self.shape  =   self.A.shape
            self.shape = A.shape
            self.dtype = A.dtype

        if tree is not None:
            self.perm   =   tree.perm_leaf
            self.Nb = tree.nleaves
            self.nl = self.A.shape[0]//self.Nb
            self.L = tree.nlevels
            self.tree = tree
        
        self.quad = quad
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
        self.tSample = 0
        self.tConstruct = 0
        self.Nbvec = []
        if quad:
            self.fac = 4
        else:
            self.fac = 2
    def set_Nbvec(self,Nbvec):
        self.Nbvec = Nbvec    
        self.Nb = Nbvec[0]
        self.nl = self.A.shape[0]//self.Nb
        self.L = len(Nbvec)
        self.perm   =   np.arange(self.A.shape[0],dtype=np.int64)
    def set_mats(self,Umats,Dmats,Vmats,Nbvec,fac=4):
        self.Umats = Umats
        self.Dmats = Dmats
        self.Vmats = Vmats
        self.perm = np.arange(Dmats[0].shape[0])
        self.fac = fac
        self.Nb = Nbvec[0]
        self.shape = np.array([Dmats[0].shape[0],Dmats[0].shape[0]],dtype = np.int64)
        self.dtype = Dmats[0][0].dtype
    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view
    def construct(self,rk,compute_ULV=False):
        
        if compute_ULV:
            self.constructHBS_ULV(rk)
        else: 
            self.constructHBS(rk)

    def constructHBS(self,rk):
        # we compute an HBS compression of permuted op
        if self.fac == 4:
            s = 6*rk+4*self.nl+5
        else:
            s = 4*rk+2*self.nl+5
        self.nSamples = s
        rng = np.random.default_rng()
        tic = time.time()
        Om = rng.standard_normal(size = (self.shape[1],s))
        Psi = rng.standard_normal(size = (self.shape[0],s))
        Omprime = np.zeros(shape = Om.shape)
        Omprime[self.perm,:] = Om
        Psiprime = np.zeros(shape = Psi.shape)
        Psiprime[self.perm,:] = Psi
        Y = self.A.matmat(Omprime)
        Z = self.A.rmatmat(Psiprime)
        Y = Y[self.perm,:]
        Z = Z[self.perm,:]
        Nb = self.Nb
        nl = self.nl
        self.tSample+=time.time()-tic
        tic = time.time()
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
                rkm = min(rk,nl*((self.fac)**(self.L-1-lvl)))
            
            
            if lvl>0:
                tic = time.time()
                U_ell = compute_UV(Om_ell,Y_ell,rkm,Nb)
                V_ell = compute_UV(Psi_ell,Z_ell,rkm,Nb)
                self.nullTime+=time.time()-tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell,Nb)
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
        self.tCompress = time.time()-tic
    def constructHBS_ULV(self,rk):
        tic = time.time()
        if self.fac == 4:
            s = 6*rk+4*self.nl+5
        else:
            s = 4*rk+2*self.nl+5
        self.nSamples = s
        Om = np.random.standard_normal(size = (self.shape[1],s))
        Psi = np.random.standard_normal(size = (self.shape[0],s))
        Omprime = np.zeros(shape = Om.shape)
        Omprime[self.perm,:] = Om
        Psiprime = np.zeros(shape = Psi.shape)
        Psiprime[self.perm,:] = Psi
        Y = self.A.matmat(Omprime)
        Z = self.A.rmatmat(Psiprime)
        Y = Y[self.perm,:]
        Z = Z[self.perm,:]
        Nb = self.Nb
        nl = self.nl
        self.tSample+=time.time()-tic
        self.NNvec = np.zeros(shape=(0,),dtype=np.int64)
        self.NNvec = np.append(self.NNvec,0)
        tic = time.time()
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
                rkm = rkm = min(rk,nl*(self.fac**(self.L-1-lvl)))
            self.Nbvec+=[Nb]
            if lvl>0:
                tic = time.time()
                U_ell = compute_UV(Om_ell,Y_ell,rkm,Nb)
                V_ell = compute_UV(Psi_ell,Z_ell,rkm,Nb)
                D_ell = construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell,Nb)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,Nb)
                self.blockSolveTime+=time.time()-tic
                self.Dmats+=[D_ell]
                U_ell = np.identity(D_ell.shape[0])
            
            if lvl==self.L-1:
                    Rhat = D_ell
            else:
                Rhat = ULVsparse.sparse_block_mult(Uhat,D_ell,self.Nbvec[-2],self.Nbvec[-1])
                Rhat = ULVsparse.block_diag_add(Rhat,R_22,self.Nbvec[-1],self.Nbvec[-2])
            
            Q,W,Ru,R_22,NN = ULVsparse.compute_QRW_sparse(Rhat,V_ell,self.Nbvec[-1])
            self.Qlist+=[Q]
            self.Wlist+=[W]
            self.Rlist+=[Ru]
            self.NNvec=np.append(self.NNvec,self.NNvec[-1]+NN)


            if lvl == self.L-1:
                Uhat = U_ell
            else:
                Uhat = ULVsparse.sparse_block_mult(Uhat,U_ell,self.Nbvec[-2],self.Nbvec[-1])
            
            Uu = ULVsparse.sparse_block_mult(Q[:,:-rk],Uhat,self.Nbvec[-1],self.Nbvec[-1],mode='T')
            Ud = ULVsparse.sparse_block_mult(Q[:,-rk:],Uhat,self.Nbvec[-1],self.Nbvec[-1],mode='T')
            self.Uulist+=[Uu]
            Uhat=Ud
        self.tCompress = time.time()-tic
    @property
    def nbytes(self):
        ctr = 0
        ctr+=sum([U.nbytes for U in self.Umats])
        ctr+=sum([V.nbytes for V in self.Vmats])
        ctr+=sum([D.nbytes for D in self.Dmats])
        return ctr
    def matvec(self,v):
        return self.matmat(v)
    def rmatvec(self,v):
        return self.rmatmat(v)
    
    def matmat(self,v):
        if v.ndim==1:
            vperm = v[self.perm,np.newaxis]    
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
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
        if v.ndim==1:
            u = u.flatten()
        return u
    
    def rmatmat(self,v):
        if v.ndim==1:
            vperm = v[self.perm,np.newaxis]    
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
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
        if v.ndim==1:
            u = u.flatten()
        return u
    
    def __matmul__(self,v):
        if v.ndim==1:
            vperm = v[self.perm,np.newaxis]    
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
        if self.mode=='N':
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

        elif self.mode=='T':
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
        self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec = ULVsparse.compute_ULV(self.Umats,self.Dmats,self.Vmats,self.Nbvec)

    def solve(self,b,mode='N'):
        if mode =='N':
            if b.ndim==1:
                bperm = b[self.perm,np.newaxis]    
            else:
                bperm= b[self.perm,:]
            rhs = bperm.copy()
            uhat = ULVsparse.solve(self.Umats,self.Dmats,self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec,self.Nbvec,rhs)
            u = np.zeros(shape = uhat.shape)
            u[self.perm,:] = uhat
        elif mode=='T':
            rhs = np.zeros(shape = b.shape)
            if b.ndim==1:
                rhs = rhs[:,None]
                rhs[self.perm,:] = b[:,None].copy()
            else:
                rhs[self.perm,:] = b.copy()
            uhat = ULVsparse.solve(self.Umats,self.Dmats,self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec,self.Nbvec,rhs,mode='T')
            u = uhat[self.perm,:]
        else:
            raise NotImplementedError("mode not recognized")
        if b.ndim==1:
            u = u.flatten()
        return u