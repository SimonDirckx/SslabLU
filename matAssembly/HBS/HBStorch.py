import numpy as np
import scipy.linalg as splinalg
import time
import matAssembly.HBS.ULVsparse_torch as ULVsparse
import torch.linalg as tla
import torch
import matAssembly.HBS.HBSnew as HBSnew
#sparse block matrix operations



def to_block_tensor(M, n, b):
    """(n*b, s) -> (n, b, s) block tensor (analogue of convert_to_torch_tens)."""
    s = M.shape[1]
    return M.reshape(n, b, s)

def _rsolve(P, B, fast=False):
    # batched min-residual solution of  X @ B = P  (i.e. X = P @ pinv(B)).
    # fast=False: SVD-based pseudoinverse (handles rank deficiency via rcond).
    # fast=True : QR-based batched least squares; assumes full column rank.
    if fast:
        return tla.lstsq(B.mT, P.mT).solution.mT
    return torch.bmm(P, tla.pinv(B))

def block_solve_r(A,B,device,fast=False):
    # batched: solves X[i] = A[i] @ pinv(B[i]) over the block dim
    return _rsolve(A, B, fast=fast)

def block_mult(A,B,device,mode='N'):
    # A: (Nb, n, k), B: (Nb, k, m)
    if mode=='N':
        return torch.bmm(A, B)
    elif mode=='T':
        return torch.bmm(A.mT, B)
    else:
        raise ValueError("mode not recognized")

def block_matvec(A,B,device,mode='N'):
    # A: (Nb, n, k), B: (Nb*nB, col) flat
    Nb  = A.shape[0]
    k   = B.shape[1]
    nB  = B.shape[0] // Nb
    Bm  = B.reshape(Nb, nB, k)
    if mode=='N':
        return torch.bmm(A, Bm).reshape(Nb * A.shape[1], k)
    elif mode=='T':
        return torch.bmm(A.mT, Bm).reshape(Nb * A.shape[2], k)
    else:
        raise ValueError("mode not recognized")

def block_mult_and_reduce(A,B,fac,device,mode='N'):
    # A: (Nb, n, rk), B: (Nb, rk, s) or (Nb, n, s)
    # After bmm: (Nb, *, s); reshape to group fac blocks together
    Nb = A.shape[0]
    if mode=='N':
        C = torch.bmm(A, B)                             # (Nb, n, s)
        return C.reshape(Nb//fac, fac*A.shape[1], B.shape[2])
    elif mode=='T':
        C = torch.bmm(A.mT, B)                         # (Nb, rk, s)
        return C.reshape(Nb//fac, fac*A.shape[2], B.shape[2])
    else:
        raise ValueError("mode not recognized")

def construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell,device,fast=False):
    # batched over the block dim; same per-block algebra as before
    Yperp = Y_ell - torch.bmm(U_ell, torch.bmm(U_ell.mT, Y_ell))
    Zperp = Z_ell - torch.bmm(V_ell, torch.bmm(V_ell.mT, Z_ell))
    term1 = _rsolve(Yperp, Om_ell, fast=fast)
    inner = _rsolve(Zperp, Psi_ell, fast=fast).mT
    term2 = torch.bmm(U_ell, torch.bmm(U_ell.mT, inner))
    return term1 + term2

def compute_UV(Om,Y,rk,device):
    # batched over the block dim; same complete QR + SVD per block as before
    n = Om.shape[1]
    Q = tla.qr(Om.mT, mode='complete').Q          # (Nb, s, s)
    # Null space of Om_i = trailing s-n columns of Q (indices n..s-1).
    # Projecting Y onto these annihilates the diagonal-block term D@Om.
    # NB: must use Q[:,:,n:], NOT Q[:,:,-n:]; the two coincide only when
    # s >= 2n, which fails once the per-level block size n grows past s/2.
    M = torch.bmm(Y, Q[:,:,n:])                    # (Nb, ny, s-n)
    return tla.svd(M, full_matrices=False).U[:,:,:rk]


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

    def __init__(self,A=None,device=None,tree=None,quad=False):
        self.Umats  =   []
        self.Vmats  =   []
        self.Dmats  =   []
        self.Qlist  =   []
        self.Rlist  =   []
        self.Wlist  =   []
        self.Uulist =   []
        torch.set_default_dtype(torch.float64)

        self.mode   =   'N'
        self._tree  =   None
        self.device = device
        if A is not None:
            self.A      =   A
            self.shape  =   self.A.shape
            self.dtype  = A.dtype

        if tree is not None:
            self.tree   =   tree
            self.perm   =   tree.perm_leaf
            self.Nb = tree.nleaves
            self.nl = len(self.perm)//self.Nb
            self.L = tree.nlevels
            self.shape = (len(self.perm),len(self.perm))
            
        self.blockSolveTime = 0
        self.nullTime = 0
        self.setupTime = 0
        self.DTime = 0
        self.tSample = 0
        self.tConstruct = 0
        self.Nbvec = []
        self.quad = quad
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
        self.shape = torch.tensor([Dmats[0].shape[0],Dmats[0].shape[0]],dtype = torch.int64)
        self.dtype = Dmats[0][0].dtype
    @property
    def nbytes(self):
        ctr = 0
        ctr+=sum([U.nbytes for U in self.Umats])
        ctr+=sum([V.nbytes for V in self.Vmats])
        ctr+=sum([D.nbytes for D in self.Dmats])
        return ctr
    def construct(self,rk,Om,Psi,Y,Z,compute_ULV=False,fast=False):
        if compute_ULV:
            self.constructHBS_ULV(rk,fast=fast)
        else: 
            self.constructHBS(rk,Om,Psi,Y,Z,fast=fast)

    def constructHBS(self,rk,Om0,Psi0,Y0,Z0,fast=False):
        s = Om0.shape[1]
        Nb = self.Nb
        self.Nbvec = [Nb]
        nl = self.nl
        self.nSamples = s
        tic = time.time()
        
        Ompr  = torch.from_numpy(Om0 ).to(device=self.device)[self.perm, :]
        Psipr = torch.from_numpy(Psi0).to(device=self.device)[self.perm, :]
        Ypr   = torch.from_numpy(Y0  ).to(device=self.device)[self.perm, :]
        Zpr   = torch.from_numpy(Z0  ).to(device=self.device)[self.perm, :]

        Y = ULVsparse.convert_to_torch_tens(Ypr,self.Nb,device=self.device)
        Z = ULVsparse.convert_to_torch_tens(Zpr,self.Nb,device=self.device)
        Om  = ULVsparse.convert_to_torch_tens(Ompr, self.Nb, device=self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psipr,self.Nb, device=self.device)


        #Om  = to_block_tensor(Ompr,  Nb, nl)
        #Psi = to_block_tensor(Psipr, Nb, nl)
        #Y   = to_block_tensor(Ypr,   Nb, nl)
        #Z   = to_block_tensor(Zpr,   Nb, nl)
        
        
        nl = self.nl
        self.setupTime+=time.time()-tic
        self.tSample+=time.time()-tic
        tic_compress = time.time()
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
                rkm = min(rk,nl*((self.fac)**(self.L-1-lvl)))
            print("lvl//Nb = ",lvl,"//",Nb)
            
            if lvl>0:
                tic = time.time()
                U_ell = compute_UV(Om_ell,Y_ell,rkm,self.device)
                V_ell = compute_UV(Psi_ell,Z_ell,rkm,self.device)
                self.nullTime+=time.time()-tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell,self.device,fast=fast)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device,fast=fast)
                self.blockSolveTime+=time.time()-tic
                # pin leaf-level matrices to CPU to save VRAM
                if self.device == 'cpu':
                    self.Dmats+=[D_ell]
                else:
                    self.Dmats+=[D_ell.cpu().pin_memory()]
            self.Nbvec+=[Nb]
        self.tCompress = time.time()-tic_compress
    def constructHBS_ULV(self,rk,fast=False):
        torch.set_default_dtype(torch.float64)
        if self.fac == 4:
            s = 6*rk+4*self.nl+5
        else:
            s = 4*rk+2*self.nl+5
        self.nSamples = s
        tic = time.time()
        print("self.fac = ",self.fac)
        # generate Om/Psi directly as torch on device
        Om_flat  = torch.randn(self.shape[1], s, dtype=torch.float64, device=self.device)
        Psi_flat = torch.randn(self.shape[0], s, dtype=torch.float64, device=self.device)
        Omprime  = torch.zeros_like(Om_flat);  Omprime[self.perm,:]  = Om_flat
        Psiprime = torch.zeros_like(Psi_flat); Psiprime[self.perm,:] = Psi_flat
        Omprime_np  = Omprime.cpu().numpy()
        Psiprime_np = Psiprime.cpu().numpy()
        Y = self.A.matvec(Omprime_np)
        Z = self.A.matvec(Psiprime_np,mode='T')
        Y = torch.from_numpy(Y[self.perm,:]).to(self.device)
        Z = torch.from_numpy(Z[self.perm,:]).to(self.device)
        Y = ULVsparse.convert_to_torch_tens(Y,self.Nb,self.device)
        Z = ULVsparse.convert_to_torch_tens(Z,self.Nb,self.device)
        Om  = ULVsparse.convert_to_torch_tens(Om_flat, self.Nb, self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psi_flat,self.Nb, self.device)
        
        Nb = self.Nb
        nl = self.nl
        self.setupTime+=time.time()-tic
        self.tSample+=time.time()-tic
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
                rkm = min(rk,nl*(self.fac**(self.L-1-lvl)))
            print("lvl//Nb = ",lvl,"//",Nb)
            self.Nbvec+=[Nb]
            if lvl>0:
                U_ell = compute_UV(Om_ell,Y_ell,rkm,self.device)
                V_ell = compute_UV(Psi_ell,Z_ell,rkm,self.device)
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,Y_ell,Z_ell,Om_ell,Psi_ell,self.device,fast=fast)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device,fast=fast)
                self.blockSolveTime+=time.time()-tic
                # pin leaf-level matrix to CPU to save VRAM
                if self.device == 'cpu':
                    self.Dmats+=[D_ell]                  
                else:
                    self.Dmats+=[D_ell.cpu().pin_memory()]
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
        self.tCompress = time.time()-tic

    @property
    def T(self):
        view = object.__new__(self.__class__)
        view.__dict__ = self.__dict__.copy()
        view.mode = 'T'
        return view

    def matvec(self,v):
        return self.matmat(v)
    def rmatvec(self,v):
        return self.rmatmat(v)

    def matmat(self,v):
        numpy_input = isinstance(v, np.ndarray)
        if numpy_input:
            v = torch.from_numpy(v).to(self.device)
        v = v.to(self.Dmats[0].dtype)
        if v.ndim==1:
            vperm = v[self.perm,None]
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
        VV+=[vperm]
        for lvl in range(len(self.Vmats)):
            v_lvl = block_matvec(self.Vmats[lvl],VV[lvl],self.device,mode='T')
            VV+=[v_lvl]
            Nb=Nb//self.fac
        # stream pinned leaf D to device non-blocking
        D_leaf = self.Dmats[-1].to(self.device, non_blocking=True)
        uperm = block_matvec(D_leaf,VV[-1],self.device)
        for lvl in range(len(self.Umats)-1,-1,-1):
            uperm = block_matvec(self.Umats[lvl],uperm,self.device)+ block_matvec(self.Dmats[lvl],VV[lvl],self.device)
            Nb=Nb*self.fac
        u = torch.zeros(size=uperm.shape,device=self.device)
        u[self.perm,:] = uperm
        if v.ndim==1:
            u = u.flatten()
        if numpy_input:
            u = u.cpu().numpy()
        return u

    def rmatmat(self,v):
        numpy_input = isinstance(v, np.ndarray)
        if numpy_input:
            v = torch.from_numpy(v).to(self.device)
        v = v.to(self.Dmats[0].dtype)
        if v.ndim==1:
            vperm = v[self.perm,None]
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
        VV+=[vperm]
        for lvl in range(len(self.Umats)):
            v_lvl = block_matvec(self.Umats[lvl],VV[lvl],self.device,mode='T')
            VV+=[v_lvl]
            Nb=Nb//self.fac
        # stream pinned leaf D to device non-blocking
        D_leaf = self.Dmats[-1].to(self.device, non_blocking=True)
        uperm = block_matvec(D_leaf,VV[-1],self.device,mode='T')
        for lvl in range(len(self.Vmats)-1,-1,-1):
            uperm = block_matvec(self.Vmats[lvl],uperm,self.device)+ block_matvec(self.Dmats[lvl],VV[lvl],self.device,mode='T')
            Nb=Nb*self.fac
        u = torch.zeros(size=uperm.shape,device=self.device)
        u[self.perm,:] = uperm
        if v.ndim==1:
            u = u.flatten()
        if numpy_input:
            u = u.cpu().numpy()
        return u

    def __matmul__(self,v):
        numpy_input = isinstance(v, np.ndarray)
        if numpy_input:
            v = torch.from_numpy(v).to(self.device)
        v = v.to(self.Dmats[0].dtype)
        if v.ndim==1:
            vperm = v[self.perm,None]
        else:
            vperm= v[self.perm,:]
        VV = []
        Nb = self.Nb
        # stream pinned leaf D to device once, shared by both branches
        D_leaf = self.Dmats[-1].to(self.device, non_blocking=True)
        if self.mode=='N':
            VV+=[vperm]
            for lvl in range(len(self.Vmats)):
                v_lvl = block_matvec(self.Vmats[lvl],VV[lvl],self.device,mode='T')
                VV+=[v_lvl]
                Nb=Nb//self.fac
            uperm = block_matvec(D_leaf,VV[-1],self.device)
            for lvl in range(len(self.Umats)-1,-1,-1):
                uperm = block_matvec(self.Umats[lvl],uperm,self.device)+ block_matvec(self.Dmats[lvl],VV[lvl],self.device)
                Nb=Nb*self.fac
            u = torch.zeros(size=uperm.shape,device=self.device)
            u[self.perm,:] = uperm
        elif self.mode=='T':
            VV+=[vperm]
            for lvl in range(len(self.Umats)):
                v_lvl = block_matvec(self.Umats[lvl],VV[lvl],self.device,mode='T')
                VV+=[v_lvl]
                Nb=Nb//self.fac
            uperm = block_matvec(D_leaf,VV[-1],self.device,mode='T')
            for lvl in range(len(self.Vmats)-1,-1,-1):
                uperm = block_matvec(self.Vmats[lvl],uperm,self.device)+ block_matvec(self.Dmats[lvl],VV[lvl],self.device,mode='T')
                Nb=Nb*self.fac
            u = torch.zeros(size=uperm.shape,device=self.device)
            u[self.perm,:] = uperm
        else:
            raise ValueError("mode not recognized")
        if v.ndim==1:
            u = u.flatten()
        if numpy_input:
            u = u.cpu().numpy()
        return u

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, t):
        self._tree = t
    
    def compute_ULV(self):
        self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec = ULVsparse.compute_ULV(self.Umats,self.Dmats,self.Vmats,self.Nbvec)

    def solve(self,b,device,mode='N'):
        if mode =='N':
            if b.ndim==1:
                bperm = b[self.perm,None]    
            else:
                bperm= b[self.perm,:]
            rhs = bperm.detach().clone()
            uhat = ULVsparse.solve(self.Umats,self.Dmats,self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec,self.Nbvec,rhs,device=device)
            u = torch.zeros(size = uhat.shape)
            u[self.perm,:] = uhat
        elif mode=='T':
            rhs = torch.zeros(size = b.shape)
            if b.ndim==1:
                rhs = rhs[:,None]
                rhs[self.perm,:] = b[:,None].detach().clone()
            else:
                rhs[self.perm,:] = b.detach().clone()
            uhat = ULVsparse.solve(self.Umats,self.Dmats,self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec,self.Nbvec,rhs,device=device,mode='T')
            u = uhat[self.perm,:]
        else:
            raise NotImplementedError("mode not recognized")
        if b.ndim==1:
            u = u.flatten()
        return u