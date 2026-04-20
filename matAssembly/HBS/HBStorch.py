import numpy as np
import scipy.linalg as splinalg
import time
import matAssembly.HBS.ULVsparse_torch as ULVsparse
import torch.linalg as tla
import torch
import matAssembly.HBS.HBSnew as HBSnew
#sparse block matrix operations

def block_col(A,rk,device):
    B = torch.zeros(size = (A.shape[0],A.shape[1],rk),device=device)
    Nb = A.shape[0]
    for i in range(Nb):
        U = tla.svd(A[i,:,:])[0]
        B[i,:,:] = U[:,:rk]
    return B
def block_col_full(A,device):
    Nb = A.shape[0]
    n = A.shape[1]
    B = torch.zeros(size = (Nb,A.shape[1],n),device=device)
    for i in range(Nb):
        B[i,:,:] =   tla.qr(A[i,:,:],mode='complete')[0]
    return B

def block_null(A,rk,device):
    Nb = A.shape[0]
    nA = A.shape[1]
    kA = A.shape[2]
    B = torch.zeros(size = (Nb,kA,rk),device=device)
    for i in range(Nb):
        B[i,:,:] = (tla.svd(A[i,:,:])[2].T)[:,-rk:]
    return B

def block_solve_r(A,B,device):
    #compute A_tau/B_tau per block
    Nb = B.shape[0]
    nb = B.shape[1]
    C = torch.zeros(size = (A.shape[0],A.shape[1],nb),device=device)
    for i in range(Nb):
        #[U,s,Vh] = tla.svd(B[i,:,:],full_matrices=False)
        #k = sum(s>s[0]*1e-14)
        #Vh = (Vh[:k,:].T)
        #U=U[:,:k]
        #s=s[:k]
        C[i,:,:] = A[i,:,:]@tla.pinv(B[i,:,:]);# (A[i,:,:]@Vh)@(U/s).T
        #C[i,:,:] = tla.lstsq(B[i,:,:].T,A[i,:,:].T)[0].T
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
def construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell):
    Nb = Om_ell.shape[0]
    C = torch.zeros(size = (Nb,U_ell.shape[1],Om_ell.shape[1]))
    for i in range(Nb):
        Usub = U_ell[i,:,:]
        Vsub = V_ell[i,:,:]
        Ysub = Y_ell[i,:,:]
        Zsub = Z_ell[i,:,:]
        Omsub = Om_ell[i,:,:]
        Psisub = Psi_ell[i,:,:]
        C[i,:,:] = (Ysub-Usub@(Usub.T@Ysub))@tla.pinv(Omsub)\
                            +Usub@(Usub.T@(((Zsub-Vsub@(Vsub.T@Zsub))@tla.pinv(Psisub)).T))
    return C
def compute_UV(Om,Y,rk):
    Nb = Om.shape[0]
    n = Om.shape[1]
    U = torch.zeros(size = (Nb,Y.shape[1],rk))
    for i in range(Nb):
        Q = tla.qr(Om[i,:,:].T,mode='complete')[0]
        U[i,:,:] = (tla.svd(Y[i,:,:]@Q[:,-n:])[0])[:,:rk]
    return U


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
        torch.set_default_dtype(torch.float64)
        tic = time.time()
        print("self.fac = ",self.fac)
        Om0 = np.random.standard_normal(size = (self.shape[1],(self.fac+4)*rk+5))
        Psi0 = np.random.standard_normal(size = (self.shape[0],(self.fac+4)*rk+5))
        Omprime = np.zeros(shape = Om0.shape)
        Omprime[self.perm,:] = Om0
        Psiprime = np.zeros(shape = Psi0.shape)
        Psiprime[self.perm,:] = Psi0
        Y = self.A.matvec(Omprime)
        Z = self.A.matvec(Psiprime,mode='T')
        Y = torch.from_numpy(Y[self.perm,:]).to(self.device)
        Z = torch.from_numpy(Z[self.perm,:]).to(self.device)
        Y = ULVsparse.convert_to_torch_tens(Y,self.Nb,self.device)
        Z = ULVsparse.convert_to_torch_tens(Z,self.Nb,self.device)

        Om = torch.from_numpy(Om0).to(self.device)
        Psi = torch.from_numpy(Psi0).to(self.device)
        Om = ULVsparse.convert_to_torch_tens(Om,self.Nb,self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psi,self.Nb,self.device)
        
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
            self.Nbvec+=[Nb]
            if lvl>0:
                U_ell = compute_UV(Om_ell,Y_ell,rkm)
                V_ell = compute_UV(Psi_ell,Z_ell,rkm)
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device)
                self.blockSolveTime+=time.time()-tic
                self.Dmats+=[D_ell]
                U_ell = torch.eye(D_ell.shape[1])
                U_ell = U_ell[None,:,:]
            
            if lvl==self.L-1:
                    Rhat = D_ell
            else:
                Rhat = ULVsparse.sparse_block_mult_tens(Uhat,D_ell,device=self.device)
                Rhat = ULVsparse.block_diag_add_tens(Rhat,R_22,device=self.device)
            
            Q,W,Ru,R_22,NN = ULVsparse.compute_QRW_sparse(Rhat,V_ell,self.Nbvec[-1],device=self.device)
            self.Qlist+=[Q]
            self.Wlist+=[W]
            self.Rlist+=[Ru]
            self.NNvec=np.append(self.NNvec,self.NNvec[-1]+NN)


            if lvl == self.L-1:
                Uhat = U_ell
            else:
                Uhat = ULVsparse.sparse_block_mult_tens(Uhat,U_ell,device=self.device)
            
            Uu = ULVsparse.sparse_block_mult_tens(Q[:,:,:-rk],Uhat,device=self.device,mode='T')
            Ud = ULVsparse.sparse_block_mult_tens(Q[:,:,-rk:],Uhat,device=self.device,mode='T')
            self.Uulist+=[Uu]
            Uhat=Ud
        

    def matvec(self,v,mode='N'):
        if v.ndim==1:
            vperm = v[:,None]    
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
        if mode=='N':
            VV+=[vperm]
            for lvl in range(len(self.Vmats)):
                Vloc = ULVsparse.convert_to_blkdiag(self.Vmats[lvl]).detach().clone().cpu().numpy()
                v_lvl = HBSnew.block_mult(Vloc,VV[lvl],Nb,mode='T')
                VV+=[v_lvl]
                Nb=Nb//self.fac
            Dloc = ULVsparse.convert_to_blkdiag(self.Dmats[-1]).detach().clone().cpu().numpy()
            uperm = HBSnew.block_mult(Dloc,VV[-1],Nb)
            for lvl in range(len(self.Umats)-1,-1,-1):
                Uloc = ULVsparse.convert_to_blkdiag(self.Umats[lvl]).detach().clone().cpu().numpy()
                Dloc = ULVsparse.convert_to_blkdiag(self.Dmats[lvl]).detach().clone().cpu().numpy()
                uperm = HBSnew.block_mult(Uloc,uperm,self.fac*Nb)+ HBSnew.block_mult(Dloc,VV[lvl],self.fac*Nb)
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

    def solve(self,b,device,mode='N'):
        if mode =='N':
            if b.ndim==1:
                bperm = b[self.perm,None]    
            else:
                bperm= b[self.perm,:]
            rhs = bperm.detach().clone()
            uhat = ULVsparse.solve(self.Umats,self.Dmats,self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec,rhs,device=device)
            u = torch.zeros(size = uhat.shape)
            u[self.perm,:] = uhat
        elif mode=='T':
            rhs = np.zeros(shape = b.shape)
            if b.ndim==1:
                rhs = rhs[:,None]
                rhs[self.perm,:] = b[:,None].copy()
            else:
                rhs[self.perm,:] = b.copy()
            uhat = ULVsparse.solve(self.Umats,self.Dmats,self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec,rhs,mode='T')
            u = uhat[self.perm,:]
        else:
            raise NotImplementedError("mode not recognized")
        if b.ndim==1:
            u = u.flatten()
        return u
