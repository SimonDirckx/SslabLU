import numpy as np
import scipy.linalg as splinalg
import time
import matAssembly.HBS.ULVsparse_torch as ULVsparse
import torch.linalg as tla
import torch
#sparse block matrix operations

def block_col(A,rk,device):
    B = torch.zeros(size = (A.shape[0],A.shape[1],rk),device=device)
    Nb = A.shape[0]
    for i in range(Nb):
        U,_ = tla.qr(A[i,:,:],mode='reduced')
        B[i,:,:] = U[:,:rk]
    return B
def block_col_full(A,device):
    Nb = A.shape[0]
    n = A.shape[1]
    B = torch.zeros(size = (Nb,A.shape[1],n),device=device)
    for i in range(Nb):
        B[i,:,:],_ = tla.qr(A[i,:,:],mode='complete')
    return B

def block_null(A,rk,device):
    Nb = A.shape[0]
    nA = A.shape[1]
    kA = A.shape[2]
    B = torch.zeros(size = (Nb,kA,rk),device=device)
    for i in range(Nb):
        Q,_ = tla.qr(A[i,:,:].T,mode='reduced')
        Om = torch.randn(size = (Q.shape[0],rk),device=device)
        Om-=Q@(Q.T@Om)
        V,_ = tla.qr(Om,mode='reduced')
        nV = V.shape[1]
        B[i,:,:] = V[:,nV-rk:]
    return B

def block_solve_r(A,B,device):
    #compute A_tau/B_tau per block
    Nb = B.shape[0]
    nb = B.shape[1]
    C = torch.zeros(size = (A.shape[0],A.shape[1],nb),device=device)
    for i in range(Nb):
        [U,s,Vh] = tla.svd(B[i,:,:],full_matrices=False)
        k = sum(s>s[0]*1e-14)
        Vh = (Vh[:k,:].T)
        C[i,:,:] = ((A[i,:,:]@Vh)/s[:k])@U[:,:k].T
    return C


def block_mult(A,B,device,mode='N'):
    Nb = A.shape[0]
    if mode=='N':
        C = torch.zeros(size=(A.shape[0],A.shape[1],B.shape[2]),device=device)
        for i in range(Nb):
            C[i,:,:]=A[i,:,:]@B[i,:,:]
    elif mode=='T':
        kA = A.shape[2]
        C = torch.zeros(size=(Nb,kA,B.shape[2]),device=device)        
        for i in range(Nb):
            C[i,:,:]=A[i,:,:].T@B[i,:,:]
    else:
        raise ValueError("mode not recognized")
    return C
def block_mult_and_reduce(A,B,fac,device,mode='N'):
    Nb = A.shape[0]
    if mode=='N':
        C = torch.zeros(size=(Nb//fac,fac*A.shape[1],B.shape[2]),device=device)
        for i in range(Nb//fac):
            M = A[i*fac,:,:]@B[i*fac,:,:]
            for j in range(1,fac):
                M=torch.cat((M,A[i*fac+j,:,:]@B[i*fac+j,:,:]),axis=0)
        C[i,:,:] = M
    elif mode=='T':
        kA = A.shape[2]
        C = torch.zeros(size=(Nb//fac,fac*kA,B.shape[2]),device=device)        
        for i in range(Nb//fac):
            M = A[i*fac,:,:].T@B[i*fac,:,:]
            for j in range(1,fac):
                M=torch.cat((M,A[i*fac+j,:,:].T@B[i*fac+j,:,:]),axis=0)
            C[i,:,:]=M
    else:
        raise ValueError("mode not recognized")
    return C

def construct_D(U_ell,V_ell,YO,ZP,device):
    Nb = U_ell.shape[0]
    C = torch.zeros(size = (U_ell.shape[0],U_ell.shape[1],YO.shape[2]),device=device)
    n = U_ell.shape[1]
    for i in range(Nb):
        Usub = U_ell[i,:,:]
        Vsub = V_ell[i,:,:]
        YOsub = YO[i,:,:]
        ZPsub = ZP[i,:,:]
        C[i,:,:] = YOsub-Usub@Usub.T@YOsub\
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

    def __init__(self,A=None,device=None,tree=None,quad=True):
        self.Umats  =   []
        self.Vmats  =   []
        self.Dmats  =   []
        self.Qlist  =   []
        self.Rlist  =   []
        self.Wlist  =   []
        self.Uulist =   []

        self.nbytes =   0
        if A is not None:
            self.A      =   A
            self.shape  =   self.A.shape
            self.shape  = A.shape
            self.dtype  = A.dtype
            self.device = device
            torch.set_default_dtype(torch.float64)

        if tree is not None:
            self.perm   =   tree.perm_leaf
            self.Nb = tree.nleaves
            self.nl = self.A.shape[0]//self.Nb
            self.L = tree.nlevels
            
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
        
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
        self.perm   =   torch.arange(self.A.shape[0],dtype=torch.int64)
    def set_mats(self,Umats,Dmats,Vmats,Nbvec,fac=4):
        self.Umats = Umats
        self.Dmats = Dmats
        self.Vmats = Vmats
        self.perm = torch.arange(Dmats[0].shape[0])
        self.fac = fac
        self.Nb = Nbvec[0]
        self.shape = torch.array([Dmats[0].shape[0],Dmats[0].shape[0]],dtype = torch.int64)
        self.dtype = Dmats[0][0].dtype
    def construct(self,rk,compute_ULV=False):
        if compute_ULV:
            self.constructHBS_ULV(rk)
        else: 
            self.constructHBS(rk)

    def constructHBS(self,rk):
        # we compute an HBS compression of permuted op
        
        tic = time.time()
        Om = np.random.standard_normal(size = (self.shape[1],(self.fac+2)*rk+5))
        Psi = np.random.standard_normal(size = (self.shape[0],(self.fac+2)*rk+5))
        Omprime = np.zeros(shape = Om.shape)
        Omprime[self.perm,:] = Om
        Psiprime = np.zeros(shape = Psi.shape)
        Psiprime[self.perm,:] = Psi
        Y = self.A.matvec(Omprime)
        Z = self.A.matvec(Psiprime,mode='T')
        Y = torch.from_numpy(Y[self.perm,:])
        Z = torch.from_numpy(Z[self.perm,:])
        Y = ULVsparse.convert_to_torch_tens(Y,self.Nb)
        Z = ULVsparse.convert_to_torch_tens(Z,self.Nb)
        Om = torch.from_numpy(Om)
        Psi = torch.from_numpy(Psi)
        Om = ULVsparse.convert_to_torch_tens(Om,self.Nb)
        Psi = ULVsparse.convert_to_torch_tens(Psi,self.Nb)
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
                
                Y_ell       -=block_mult(D_ell,Om_ell,self.device)
                Y_ell       = block_mult(U_ell,Y_ell,self.device,mode='T')

                Z_ell       -= block_mult(D_ell,Psi_ell,self.device,mode='T')
                Z_ell       = block_mult(V_ell,Z_ell,self.device,mode='T')
                
                Om_ell      = block_mult(V_ell,Om_ell,self.device,mode='T')
                Psi_ell     = block_mult(U_ell,Psi_ell,self.device,mode='T')
                
                Nb = Nb//self.fac
                rkm = rk
            print("lvl//Nb = ",lvl,"//",Nb)
            
            if lvl>0:
                tic = time.time()
                P_ell = block_null(Om_ell,rkm,self.device)
                Q_ell = block_null(Psi_ell,rkm,self.device)
                self.nullTime+=time.time()-tic
                YP = block_mult(Y_ell,P_ell,self.device)
                ZQ = block_mult(Z_ell,Q_ell,self.device)
                U_ell = block_col(YP,rkm,self.device)
                V_ell = block_col(ZQ,rkm,self.device)
                tic = time.time()
                YO = block_solve_r(Y_ell,Om_ell,self.device)
                ZP = block_solve_r(Z_ell,Psi_ell,self.device)
                self.blockSolveTime+=time.time()-tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,YO,ZP,self.device)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device)
                self.blockSolveTime+=time.time()-tic
                self.Dmats+=[D_ell]
            self.Nbvec+=[Nb]
    def constructHBS_ULV(self,rk):
        # we compute an HBS compression of permuted op
        tic = time.time()
        Om0 = np.random.standard_normal(size = (self.shape[1],(self.fac+2)*rk+10))
        Psi0 = np.random.standard_normal(size = (self.shape[0],(self.fac+2)*rk+10))
        Omprime = np.zeros(shape = Om0.shape)
        Omprime[self.perm,:] = Om0
        Psiprime = np.zeros(shape = Psi0.shape)
        Psiprime[self.perm,:] = Psi0
        Y = self.A.matvec(Omprime)
        Z = self.A.matvec(Psiprime,mode='T')
        Y = torch.from_numpy(Y[self.perm,:]).to(self.device)
        Z = torch.from_numpy(Z[self.perm,:]).to(self.device)
        Y = ULVsparse.convert_to_torch_tens(Y,self.Nb)
        Z = ULVsparse.convert_to_torch_tens(Z,self.Nb)

        Om = torch.from_numpy(Om0)
        Psi = torch.from_numpy(Psi0)
        Om = ULVsparse.convert_to_torch_tens(Om,self.Nb).to(self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psi,self.Nb).to(self.device)

        Nb = self.Nb
        nl = self.nl
        self.setupTime+=time.time()-tic
        self.NNvec = np.zeros(shape=(0,),dtype=np.int64)
        self.NNvec = np.append(self.NNvec,0)
        for lvl in range(self.L-1,-1,-1):
            
            if lvl == self.L-1:
                Om_ell      = Om
                Psi_ell     = Psi
                Y_ell       = Y
                Z_ell       = Z
                rkm = min(rk,nl)
            else:
                Y_ell       -=block_mult(D_ell,Om_ell,self.device)
                Y_ell       = block_mult_and_reduce(U_ell,Y_ell,self.fac,self.device,mode='T')

                Z_ell       -= block_mult(D_ell,Psi_ell,self.device,mode='T')
                Z_ell       = block_mult_and_reduce(V_ell,Z_ell,self.fac,self.device,mode='T')
                
                Om_ell      = block_mult_and_reduce(V_ell,Om_ell,self.fac,self.device,mode='T')
                Psi_ell     = block_mult_and_reduce(U_ell,Psi_ell,self.fac,self.device,mode='T')
                
                Nb = Nb//self.fac
                rkm = rk
            print("lvl//Nb = ",lvl,"//",Nb)
            
            if lvl>0:
                tic = time.time()
                P_ell = block_null(Om_ell,rkm,self.device)
                Q_ell = block_null(Psi_ell,rkm,self.device)
                self.nullTime+=time.time()-tic
                YP = block_mult(Y_ell,P_ell,self.device)
                ZQ = block_mult(Z_ell,Q_ell,self.device)
                U_ell = block_col(YP,rkm,self.device)
                W_ell = block_col_full(ZQ,self.device)
                V_ell = W_ell[:,:,:rkm]



                tic = time.time()
                YO = block_solve_r(Y_ell,Om_ell,self.device)
                ZP = block_solve_r(Z_ell,Psi_ell,self.device)
                self.blockSolveTime+=time.time()-tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,YO,ZP,self.device)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
                self.Wlist+=[W_ell]

            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device)
                self.blockSolveTime+=time.time()-tic
                self.Dmats+=[D_ell]
            self.Nbvec+=[Nb]

            
            if lvl==self.L-1:
                Rprime = D_ell
                Q,Ru,R22,NN = ULVsparse.compute_QR_sparse(Rprime,W_ell,rkm,self.device)
                n = U_ell.shape[1]
                k = U_ell.shape[2]
                Uu = ULVsparse.sparse_block_mult_tens(Q[:,:,:n-k],U_ell,self.device,mode='T')
                Ud = ULVsparse.sparse_block_mult_tens(Q[:,:,n-k:],U_ell,self.device,mode='T')
                self.Uulist+=[Uu]
                Uhat=Ud
            else:
                Rhat = ULVsparse.sparse_block_mult_tens(Uhat,D_ell,self.device)
                Rhat = ULVsparse.block_diag_add_tens(Rhat,R22,self.device)
                Q,Ru,R22,NN = ULVsparse.compute_QR_sparse(Rhat,W_ell,rkm,self.device)
                if lvl>0:
                    n = U_ell.shape[1]
                    k = U_ell.shape[2]
                    Uhat = ULVsparse.sparse_block_mult_tens(Uhat,U_ell,self.device)
                    Uu = ULVsparse.sparse_block_mult_tens(Q[:,:,:n-k],Uhat,self.device,mode='T')
                    Ud = ULVsparse.sparse_block_mult_tens(Q[:,:,n-k:],Uhat,self.device,mode='T')
                    self.Uulist+=[Uu]
                    Uhat=Ud           
            self.Qlist+=[Q]
            self.Rlist+=[Ru]
            self.NNvec = np.append(self.NNvec,self.NNvec[-1]+NN)

                

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
        self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec = ULVsparse.compute_ULV(self.Umats,self.Dmats,self.Vmats,self.Nbvec)

    def solve(self,b,mode='N'):
        if b.ndim==1:
            bperm = b[self.perm,np.newaxis]    
        else:
            bperm= b[self.perm,:]
        rhs = bperm.copy()
        for i in range(len(self.Q1list)):
            btemp = rhs.copy()
            btemp[self.NNvec[i]:self.NNvec[i+1],:] = ULVsparse.apply_sparse_block(self.Q1list[i],bperm,self.Nbvec[i],mode='T')
            btemp[self.NNvec[i+1]:,:] = ULVsparse.apply_sparse_block(self.Q2list[i],bperm,self.Nbvec[i],mode='T')
            rhs = btemp.copy()
        uhat = ULVsparse.solve_R(self.Rtot,rhs,self.Nbvec,self.NNvec,self.NNRvec)
        uperm = ULVsparse.apply_cbd(self.Wtot,uhat,self.Nbvec,self.NNvec,self.NNQvec)
        u = np.zeros(shape = uperm.shape)
        u[self.perm,:] = uperm
        if b.ndim==1:
            u = u.flatten()
        return u
