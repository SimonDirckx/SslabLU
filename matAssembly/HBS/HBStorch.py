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

def _rsolve(P, B, fast=False, rtol=None, QR=None):
    """
    Batched min-residual solution of  X @ B = P   (X = P B^+).
    B: (Nb, n, s) with s >= n.  P: (Nb, ny, s).  Returns (Nb, ny, n).
    Solves the transposed system B^T X^T = P^T, which is tall/overdetermined.
    fast=True : QR + triangular solve (assumes full column rank of B^T).
    fast=False: SVD applied factor-by-factor with an explicit relative cutoff.
    QR        : optional precomputed (Q, R) of B.mT, see note below.
    """
    assert B.shape[-1] >= B.shape[-2], "undersampled: s < n, this is a different problem"

    #if fast:
    #    if QR is None:
    #        Q, R = tla.qr(B.mT, mode='reduced')      # Q:(Nb,s,n)  R:(Nb,n,n)
    #    else:
    #        Q, R = QR
    #    PQ = torch.bmm(P, Q)                          # (Nb, ny, n)
    #    # X R^T = PQ ;  R^T is lower triangular, solve from the right
    #    return torch.linalg.solve_triangular(R.mT, PQ, upper=False, left=False)

    U, S, Vh = tla.svd(B, full_matrices=False)        # U:(Nb,n,n) S:(Nb,n) Vh:(Nb,n,s)
    if rtol is None:
        rtol = max(B.shape[-1], B.shape[-2]) * torch.finfo(B.dtype).eps
    Sinv = torch.where(S > rtol * S[..., :1], 1.0 / S, torch.zeros_like(S))
    # X = P V S^+ U^T, applied right-to-left; the (s,n) pinv is never built
    return torch.bmm(torch.bmm(P, Vh.mT) * Sinv.unsqueeze(-2), U.mT)

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


def _small_pinv_factors(R, s=None, rtol=None):
    """B = R^T Q^T  =>  pinv(R^T) = Vh^T diag(Sinv) Uc^T, all n x n."""
    n = R.shape[-1]
    Uc, S, Vhc = tla.svd(R.mT, full_matrices=False)
    if rtol is None:
        rtol = max(s if s is not None else n, n) * torch.finfo(R.dtype).eps
    Sinv = torch.where(S > rtol*S[...,:1], S.reciprocal(), torch.zeros_like(S))
    return Uc, Sinv, Vhc


def _rsolve_qr(P, QR, s=None, fast=False):
    """P B^+ where B = R^T Q^T. Replaces _rsolve(P, B)."""
    Q, R = QR
    PQ = torch.bmm(P, Q)                                  # (Nb, ny, n)
    #if fast:
    #    return torch.linalg.solve_triangular(R, PQ, upper=True, left=False)
    Uc, Sinv, Vhc = _small_pinv_factors(R, s=s)
    return torch.bmm(torch.bmm(PQ, Vhc.mT) * Sinv.unsqueeze(-2), Uc.mT)


def _pinv_apply_left(QR, U, s=None, fast=False):
    """B^+ U = Q pinv(R^T) U, applied to k columns."""
    Q, R = QR
    if fast:
        T = torch.linalg.solve_triangular(R.mT, U, upper=False, left=True)
    else:
        Uc, Sinv, Vhc = _small_pinv_factors(R, s=s)
        T = torch.bmm(Vhc.mT, Sinv.unsqueeze(-1) * torch.bmm(Uc.mT, U))
    return torch.bmm(Q, T)                                # (Nb, s, k)

def construct_D(U_ell, V_ell, Y_ell, Z_ell, om_qr, psi_qr,
                device=None, s=None, fast=False):
    """
    om_qr  = (Q_om,  R_om)   from compute_UV(Om_ell,  Y_ell, ...)
    psi_qr = (Q_psi, R_psi)  from compute_UV(Psi_ell, Z_ell, ...)
    s : column count of the original Om/Psi, needed only to keep the rtol
        cutoff identical to the old max(s, n) * eps.
    """
    # --- term1: (I - U U^T) Y  Om^+ ---------------------------------------
    # formed explicitly: this term needs all n columns of the result
    Yperp = Y_ell - torch.bmm(U_ell, torch.bmm(U_ell.mT, Y_ell))
    term1 = _rsolve_qr(Yperp, om_qr, s=s, fast=fast)

    # --- term2: U U^T (Zperp Psi^+)^T  ==  U ( Zperp (Psi^+ U) )^T ---------
    X  = _pinv_apply_left(psi_qr, U_ell, s=s, fast=fast)   # (Nb, s, k) = Psi^+ U
    ZX = torch.bmm(Z_ell, X)                               # (Nb, n, k)
    ZX -= torch.bmm(V_ell, torch.bmm(V_ell.mT, ZX))        # = Zperp X, Zperp never formed
    term2 = torch.bmm(U_ell, ZX.mT)

    return term1 + term2

def compute_UV(Om, Y, rk, device,fast=False,work=None):
    """
    Orthonormal basis for the dominant rk-dim left singular subspace of
    Y projected off the row space of Om.

    Om : (Nb, n,  s)        Y : (Nb, ny, s)        s >= n
    Returns (U, (Q_om, R_om)) with U : (Nb, ny, k), k = min(rk, ny).
    The QR pair is the factorization of Om that _rsolve needs in
    construct_D -- handing it back is what removes the duplicate factor.
    """
    Nb, ny, s = Y.shape
    n = Om.shape[1]
    k = min(rk, ny)
    assert s >= n + k, "undersampled: not enough columns left after projecting off Om"

    # ---- one QR does the projection and the Om factorization at once -------
    if work is None:
        W = torch.empty((Nb, s, n + ny), dtype=Y.dtype, device=Y.device)
    else:
        W = work[:Nb, :s, :n + ny]          # caller-owned arena, see below
    W[:, :, :n].copy_(Om.mT)
    W[:, :, n:].copy_(Y.mT)

    Q, R = tla.qr(W, mode='reduced')        # Q:(Nb,s,r) R:(Nb,r,n+ny), r=min(s,n+ny)

    # .contiguous() drops the reference to the full Q so the (Nb,s,ny) tail
    # is freed here rather than living as long as the returned view.
    Q_om = Q[:, :, :n]        # == old Qhead
    R_om = R[:, :n, :n]
    L    = R[:, n:, n:].mT                  # (Nb, ny, r-n) -- this is the L of M

    # ---- tiny factorization on L instead of a wide one on M ----------------
    if fast:
        G = torch.bmm(L, L.mT)
        G = 0.5 * (G + G.mT)
        U = tla.eigh(G).eigenvectors[..., -k:].flip(-1)
    else:
        U = tla.svd(L.contiguous(), full_matrices=False).U[..., :k]

    return U, (Q_om, R_om)


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

    Device policy
    -------------
    Every tensor the object stores lives on self.device.  There is no per-level
    exception: Dmats is handled exactly like Umats and Vmats, so any consumer
    that takes the whole list (ULVsparse.solve, ULVsparse.compute_ULV) sees a
    list on one device.  Use to()/cpu() to relocate the whole object; that is
    the only supported way to trade VRAM for host memory.

    Note on level ordering: the construction loop runs L-1 down to 0, so
    Dmats[0] is the leaf level and Dmats[-1] is the ROOT.  Earlier versions
    parked Dmats[-1] in pinned host memory under the name "leaf D"; that block
    is in fact the smallest in the list (one block of side ~2*rk), so the
    saving was negligible while the mixed-device list silently broke solve().

    """

    # every attribute holding a list of tensors; to() walks these.
    # NNvec is numpy and Nbvec is a list of ints, so both stay put.
    _GROUPS = {
        'core': ('Umats', 'Vmats', 'Dmats'),
        'ulv' : ('Qlist', 'Wlist', 'Rlist', 'Uulist'),
    }
    _tensor_lists = ('Umats', 'Vmats', 'Dmats',
                     'Qlist', 'Wlist', 'Rlist', 'Uulist')

    def __init__(self,A=None,device=None,tree=None,quad=False):
        self.Umats  =   []
        self.Vmats  =   []
        self.Dmats  =   []
        self.Qlist  =   []
        self.Rlist  =   []
        self.Wlist  =   []
        self.Uulist =   []
        torch.set_default_dtype(torch.float64)
        self.dtype = np.float64
        self.dtype_t = torch.float64

        self.mode   =   'N'
        self._tree  =   None

        dev = torch.device(device) if device is not None else torch.device('cpu')
        self.compute_device = dev
        self.home           = torch.device('cpu')
        self._resident = {'core': dev, 'ulv': dev}
        self._dirty    = {'core': dev != self.home, 'ulv': dev != self.home}
        self.strict    = False
        self.nFill = self.nSpill = 0
        self.bytesH2D = self.bytesD2H = 0

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
        self.tULV = 0
        self.tCompress = 0
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
    @property
    def device(self):
        return str(self.compute_device)
    def _as_local(self,X):
        dev = self.compute_device
        if torch.is_tensor(X):
            return X.to(device=dev,dtype=self.dtype_t,non_blocking=True)
        return torch.from_numpy(X).to(device=dev,dtype=self.dtype_t)
    def construct(self,rk,Om=None,Psi=None,Y=None,Z=None,compute_ULV=False,fast=False):
        if Om is None:
            if self.A is None:
                raise ValueError("Samples and LinOP cannot both be None")
            else:
                s = 2*max(rk,self.tree._min_leaf_size)+rk+10

                Om = np.random.standard_normal(size = (self.A.shape[1],s))
                Psi= np.random.standard_normal(size = (self.A.shape[0],s))
                Y = self.A@Om
                Z = self.A.T@Psi

        if compute_ULV:
            self.constructHBS_ULV(rk,Om,Psi,Y,Z,fast=fast)
        else: 
            self.constructHBS(rk,Om,Psi,Y,Z,fast=fast)

    def constructHBS(self,rk,Om0,Psi0,Y0,Z0,fast=False):
        s = Om0.shape[1]
        Nb = self.Nb
        self.Nbvec = [Nb]
        nl = self.nl
        self.nSamples = s
        tic = time.time()
        if torch.is_tensor(self.perm):
            self.perm = self.perm.to(self.compute_device)
        else:
            self.perm = torch.as_tensor(self.perm, dtype=torch.int64,
                                        device=self.compute_device)        
        Ompr  = self._as_local(Om0 )[self.perm, :]
        Psipr = self._as_local(Psi0)[self.perm, :]
        Ypr   = self._as_local(Y0  )[self.perm, :]
        Zpr   = self._as_local(Z0  )[self.perm, :]

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
            
            if lvl>0:
                tic = time.time()
                U_ell,om_qr = compute_UV(Om_ell,Y_ell,rkm,self.device,fast=fast)
                V_ell,psi_qr = compute_UV(Psi_ell,Z_ell,rkm,self.device,fast=fast)
                self.nullTime+=time.time()-tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,Y_ell,Z_ell,om_qr,psi_qr,self.device,fast=fast)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device,fast=fast)
                self.blockSolveTime+=time.time()-tic
                # stored on self.device like every other level; use to()/cpu()
                # to relocate the whole object rather than one block of it
                self.Dmats+=[D_ell]
            self.Nbvec+=[Nb]
        self.tCompress = time.time()-tic_compress
    def constructHBS_ULV(self,rk,Om0,Psi0,Y0,Z0,fast=True):
        s = Om0.shape[1]
        Nb = self.Nb
        self.Nbvec = [Nb]
        nl = self.nl
        self.nSamples = s
        tic = time.time()
        if torch.is_tensor(self.perm):
            self.perm = self.perm.to(self.compute_device)
        else:
            self.perm = torch.as_tensor(self.perm, dtype=torch.int64,
                                        device=self.compute_device)
        Ompr  = self._as_local(Om0 )[self.perm, :]
        Psipr = self._as_local(Psi0)[self.perm, :]
        Ypr   = self._as_local(Y0  )[self.perm, :]
        Zpr   = self._as_local(Z0  )[self.perm, :]

        Y = ULVsparse.convert_to_torch_tens(Ypr,self.Nb,device=self.device)
        Z = ULVsparse.convert_to_torch_tens(Zpr,self.Nb,device=self.device)
        Om  = ULVsparse.convert_to_torch_tens(Ompr, self.Nb, device=self.device)
        Psi = ULVsparse.convert_to_torch_tens(Psipr,self.Nb, device=self.device)
        
        Nb = self.Nb
        nl = self.nl
        self.setupTime+=time.time()-tic
        self.tSample+=time.time()-tic
        self.NNvec = np.zeros(shape=(0,),dtype=np.int64)
        self.NNvec = np.append(self.NNvec,0)
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
                rkm = min(rk,nl*(self.fac**(self.L-1-lvl)))
            #print("lvl//Nb = ",lvl,"//",Nb)
            self.Nbvec+=[Nb]
            if lvl>0:
                tic = time.time()
                U_ell,om_qr = compute_UV(Om_ell,Y_ell,rkm,self.device,fast=fast)
                V_ell,psi_qr = compute_UV(Psi_ell,Z_ell,rkm,self.device,fast=fast)
                self.nullTime+=time.time()-tic
                tic = time.time()
                D_ell = construct_D(U_ell,V_ell,Y_ell,Z_ell,om_qr,psi_qr,self.device,fast=fast)
                self.DTime+= time.time()-tic
                self.Dmats+=[D_ell]
                self.Umats+=[U_ell]
                self.Vmats+=[V_ell]
            else:
                tic = time.time()
                D_ell = block_solve_r(Y_ell,Om_ell,self.device,fast=fast)
                self.blockSolveTime+=time.time()-tic
                # stored on self.device like every other level; use to()/cpu()
                # to relocate the whole object rather than one block of it
                self.Dmats+=[D_ell]
                U_ell = torch.eye(D_ell.shape[1], dtype=D_ell.dtype,device=self.device)[None, :, :]
            
            tic = time.time()
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

            tic = time.time()
            if lvl == self.L-1:
                Uhat = U_ell
            else:
                Uhat = ULVsparse.sparse_block_mult_tens(Uhat,U_ell,device=self.device)
            
            Uu = ULVsparse.sparse_block_mult_tens(Q[:,:,:-rkm],Uhat,device=self.device,mode='T')
            Ud = ULVsparse.sparse_block_mult_tens(Q[:,:,-rkm:],Uhat,device=self.device,mode='T')
            self.Uulist+=[Uu]
            Uhat=Ud
            self.tULV +=time.time()-tic
        if self.compute_device == 'cuda':
            torch.cuda.synchronize()
        self.tCompress = time.time()-tic_compress

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
        self._require_resident('matmat')
        dev = self.compute_device
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
        uperm = block_matvec(self.Dmats[-1],VV[-1],self.device)
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
        uperm = block_matvec(self.Dmats[-1],VV[-1],self.device,mode='T')
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
        if self.mode=='N':
            VV+=[vperm]
            for lvl in range(len(self.Vmats)):
                v_lvl = block_matvec(self.Vmats[lvl],VV[lvl],self.device,mode='T')
                VV+=[v_lvl]
                Nb=Nb//self.fac
            uperm = block_matvec(self.Dmats[-1],VV[-1],self.device)
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
            uperm = block_matvec(self.Dmats[-1],VV[-1],self.device,mode='T')
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

    # ------------------------------------------------------------------
    # Device placement
    # ------------------------------------------------------------------

    def _lists(self, groups):
        for g in groups:
            for name in self._GROUPS[g]:
                yield g, getattr(self, name)

    def _move(self, target, groups, non_blocking=False):
        target = torch.device(target)
        moved = 0
        for g, lst in self._lists(groups):
            for i, t in enumerate(lst):
                if torch.is_tensor(t) and t.device != target:
                    moved += t.nbytes
                    lst[i] = t.to(target, non_blocking=non_blocking)
        if 'core' in groups and torch.is_tensor(self.perm) \
                and self.perm.device != target:
            moved += self.perm.nbytes
            self.perm = self.perm.to(target, non_blocking=non_blocking)
        return moved

    def _group_resident(self, g):
        # A group with no tensors (e.g. ulv on a compute_ULV=False block)
        # is vacuously resident -- there is nothing to stage.
        if not any(len(getattr(self, n)) for n in self._GROUPS[g]):
            return True
        r = self._resident[g]
        return r is not None and torch.device(r) == self.compute_device

    def prefetch(self, groups=('core',), non_blocking=True):
        """Stage a device mirror for the named groups. No-op if resident."""
        need = [g for g in groups if not self._group_resident(g)]
        if not need:
            return self
        n = self._move(self.compute_device, need, non_blocking=non_blocking)
        for g in need:
            self._resident[g] = self.compute_device
        self.nFill += 1
        self.bytesH2D += n
        return self

    def evict(self, groups=('core', 'ulv')):
        """Drop the device mirror for the named groups.

        A *dirty* group was built on device and has no home copy: it is
        written back.  A *clean* group already has an identical home copy
        -- it was staged in for read-only use -- so the tensors are rebound
        to `home` and the device memory returns to the caching allocator
        with no copy at all.  That clean path is what makes a
        consume-before-evict schedule cost one bus crossing per block.

        compute_device is NOT touched.  An evicted block that is applied
        again must be re-staged, never silently demoted to host arithmetic;
        that is the whole point of this method existing instead of to('cpu').
        """
        if self.compute_device == self.home:
            return self 
        for g in groups:
            if self._resident[g] is None:
                continue
            n = self._move(self.home, (g,))
            if self._dirty[g]:
                self.bytesD2H += n
                self._dirty[g] = False
            self._resident[g] = None
            self.nSpill += 1
        return self

    def release_ulv(self):
        """Send the ULV factors home, keep the apply factors on device."""
        return self.evict(groups=('ulv',))
    def to(self, device, non_blocking=False):
        device = torch.device(device)

        moved = 0
        for g in self._GROUPS:
            moved += self._move(device, (g,), non_blocking=non_blocking)
            self._resident[g] = device
            self._dirty[g]    = False

        # perm rides with 'core' inside _move, but normalize it here so a
        # tree-supplied numpy perm becomes a tensor on the target rather
        # than staying host-side and forcing an implicit H2D on every
        # fancy-index in matmat/constructHBS.
        if not torch.is_tensor(self.perm):
            self.perm = torch.as_tensor(self.perm, dtype=torch.int64,
                                        device=device)

        self.home           = device
        self.compute_device = device

        # Counted separately from bytesH2D/bytesD2H so residency_report()
        # keeps measuring the spill/fill schedule and is not polluted by
        # one-off relocations.
        self.bytesRelocated = getattr(self, 'bytesRelocated', 0) + moved
        return self

    def cpu(self):
        """Relocate to host: this becomes a host object that computes on the
        host.  NOT the same as evict(), which keeps compute_device on the
        accelerator and merely drops the mirror."""
        return self.to('cpu')

    def cuda(self, index=None):
        """Relocate to a CUDA device."""
        return self.to('cuda' if index is None else f'cuda:{index}')

    def device_nbytes(self, include_ulv=True):
        groups = ('core', 'ulv') if include_ulv else ('core',)
        tot = sum(t.nbytes for _, lst in self._lists(groups)
                  for t in lst if torch.is_tensor(t))
        if torch.is_tensor(self.perm):
            tot += self.perm.nbytes
        return tot

    def _require_resident(self, what, need_ulv=False):
        groups = ('core', 'ulv') if need_ulv else ('core',)
        missing = [g for g in groups if not self._group_resident(g)]
        if not missing:
            return
        if self.strict:
            raise RuntimeError(
                f"{what}() on an HBS block whose {'+'.join(missing)} factors "
                f"are not resident on {self.compute_device} (resident="
                f"{self._resident}). The block was evicted and nothing staged "
                "it back. Under strict=True this is an error instead of a "
                "silent fallback to host arithmetic."
            )
        self.prefetch(groups=missing)

    def compute_ULV(self):
        self._require_resident('compute_ULV')
        tic=time.time()
        self.Qlist,self.Wlist,self.Uulist,self.Rlist,self.NNvec = ULVsparse.compute_ULV(self.Umats,self.Dmats,self.Vmats,self.Nbvec)
        self.tULV = time.time() - tic
        self._resident['ulv'] = self.compute_device
        self._dirty['ulv']    = (self.compute_device != self.home)

    def solve(self, b, mode='N'):

        # ------------------------------------------------------------
        # Input handling
        # ------------------------------------------------------------
        if not self.Qlist:
            raise RuntimeError(
                "solve() requires the ULV factorization, but this HBSMAT has "
                "an empty Qlist.  Build it with construct(..., compute_ULV=True) "
                "or call compute_ULV() first."
            )

        self._require_resident('solve',need_ulv=True)
        input_is_numpy = isinstance(b, np.ndarray)
        input_is_torch = torch.is_tensor(b)

        if not (input_is_numpy or input_is_torch):
            raise TypeError(
                "b must be either a numpy.ndarray or a torch.Tensor"
            )

        # Convert NumPy input to torch. Keep torch input unchanged.
        if input_is_numpy:
            b_torch = torch.as_tensor(
                b,
                dtype=self.Umats[0].dtype,
                device=self.device
            )
        else:
            b_torch = b.to(device=self.compute_device,dtype=self.Umats[0].dtype)

        was_vector = (b_torch.ndim == 1)

        if b_torch.ndim not in (1, 2):
            raise ValueError(
                f"b must have ndim 1 or 2, got ndim={b_torch.ndim}"
            )

        # Always work internally with a 2D RHS
        if was_vector:
            b_torch = b_torch[:, None]

        # ------------------------------------------------------------
        # Normal solve
        # ------------------------------------------------------------
        if mode == 'N':

            # Apply permutation
            bperm = b_torch[self.perm, :].clone()

            uhat = ULVsparse.solve(
                self.Umats,
                self.Dmats,
                self.Qlist,
                self.Wlist,
                self.Uulist,
                self.Rlist,
                self.NNvec,
                bperm,
                device=self.device
            )

            # Undo permutation
            u = torch.empty_like(uhat)
            u[self.perm, :] = uhat

        # ------------------------------------------------------------
        # Transpose solve
        # ------------------------------------------------------------
        elif mode == 'T':

            # Here the permutation is applied in the opposite direction
            rhs = torch.empty_like(b_torch)
            rhs[self.perm, :] = b_torch

            uhat = ULVsparse.solve(
                self.Umats,
                self.Dmats,
                self.Qlist,
                self.Wlist,
                self.Uulist,
                self.Rlist,
                self.NNvec,
                rhs,
                device=self.device,
                mode='T'
            )

            u = uhat[self.perm, :]

        else:
            raise NotImplementedError(
                f"mode '{mode}' not recognized. Use 'N' or 'T'."
            )

        # Restore vector shape
        if was_vector:
            u = u[:, 0]

        # Return in the same array type as the input
        if input_is_numpy:
            return u.detach().cpu().numpy()

        return u
