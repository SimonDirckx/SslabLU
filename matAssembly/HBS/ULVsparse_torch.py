import time
import torch
import torch.linalg as tla
import matplotlib.pyplot as plt

'''
Factorize Q^* S W = R
with R upper triangular 

Q,R,W given in reduced format

'''

# ---------------------------------------------------------------------------
# module knobs
# ---------------------------------------------------------------------------
_QR_CHUNK = [0]   # blocks per batched QR call; 0 means "the whole batch at once"


def _qr_batches(Nb):
    """[(j0, j1), ...] covering range(Nb), honouring _QR_CHUNK.

    A single (0, Nb) pair is the default, and both QR helpers below take a
    fast path for it that returns the factors directly instead of copying
    them into a preallocated buffer.  The knob exists because the batched
    form holds every block's Q at once where the old per-block loop held
    one: at the coarse red-black levels, where n is large, that peak can be
    the binding constraint even though the throughput is better.
    """
    c = _QR_CHUNK[0] or Nb
    c = max(int(c), 1)
    return [(j, min(j + c, Nb)) for j in range(0, Nb, c)]


def convert_to_torch_tens(A,Nb,device):
    """(Nb*n, k) -> (Nb, n, k).

    A VIEW when A is contiguous, which the permuted sample gathers in
    constructHBS / constructHBS_ULV always are (advanced indexing returns a
    fresh contiguous tensor).  The old loop copied into a zeroed buffer,
    which held a second full copy of Om/Psi/Y/Z on the device -- 4H at the
    compression peak -- and issued 4*Nb small kernels per construction.

    Consequence: callers may modify the result in place, and that modifies A.
    constructHBS* relies on exactly that (Y_ell -= ... writes into Ypr, which
    is private to the call and never read again).  Do not pass a tensor that
    is still needed afterwards.
    """
    if A.shape[0] % Nb:
        raise ValueError(f"convert_to_torch_tens: {A.shape[0]} rows not divisible by Nb={Nb}")
    n = A.shape[0]//Nb
    return A.to(device).reshape(Nb, n, A.shape[1])
def convert_to_blkdiag(A):
    n = A.shape[1]
    k = A.shape[2]
    Nb = A.shape[0]
    B = torch.zeros(size = (n*Nb,k))
    for i in range(Nb):
        B[i*n:(i+1)*n,:] = A[i,:,:]
    return B

def block_qr_tens(A, device=None):
    """Orthonormal basis for the complement of each block's column space.

    A: (Nb, n, k) with k <= n.  Returns (Nb, n, n-k) whose columns span
    range(A[i])^perp, from the trailing columns of a complete QR.

    One batched QR replaces a Python loop of Nb single-matrix cuSOLVER
    calls, each of which also copied its result into a slice of a zeroed
    buffer -- 2*Nb kernel launches per call on matrices far too small to
    fill the device.  `device` is accepted for call-site compatibility and
    ignored: the tensor carries its own, as does its dtype (the old buffer
    was allocated at the default dtype regardless of A's).

    The result is a fresh contiguous tensor, not a view into Q, so callers
    may write into it and the (Nb, n, n) Q is freed on return.
    """
    Nb, n, k = A.shape
    if k > n:
        raise ValueError(f"block_qr_tens: need k <= n, got n={n}, k={k}")

    batches = _qr_batches(Nb)
    if len(batches) <= 1:
        return tla.qr(A, mode='complete').Q[:, :, k:].contiguous()

    C = A.new_empty((Nb, n, n - k))
    for j0, j1 in batches:
        Q = tla.qr(A[j0:j1], mode='complete').Q
        C[j0:j1] = Q[:, :, k:]
        del Q
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
def block_Q_and_R_tens(W12, Dtot, device=None):
    """Reduced QR of D[i] @ W12[i], batched over the block dim.

    Dtot: (Nb, n, p), W12: (Nb, p, m).  Returns Q (Nb, n, min(n,m)) and
    R (Nb, min(n,m), m).  The loop form issued 2*Nb launches for the
    products alone and then Nb separate cuSOLVER QRs; one bmm and one
    batched QR replace both.  `device` is accepted for call-site
    compatibility and ignored.
    """
    if W12.shape[0] != Dtot.shape[0] or W12.shape[1] != Dtot.shape[2]:
        raise ValueError(
            f"block_Q_and_R_tens: shape mismatch, Dtot {tuple(Dtot.shape)} "
            f"@ W12 {tuple(W12.shape)}")

    Nb, n = Dtot.shape[0], Dtot.shape[1]
    m = W12.shape[2]

    batches = _qr_batches(Nb)
    if len(batches) <= 1:
        return tla.qr(torch.bmm(Dtot, W12))

    r = min(n, m)
    Q = Dtot.new_empty((Nb, n, r))
    R = Dtot.new_empty((Nb, r, m))
    for j0, j1 in batches:
        Qc, Rc = tla.qr(torch.bmm(Dtot[j0:j1], W12[j0:j1]))
        Q[j0:j1] = Qc
        R[j0:j1] = Rc
        del Qc, Rc
    return Q, R
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
def sparse_block_mult_tens(A, B, device=None, mode='N'):
    """Multiply block diagonal matrices, both in reduced form.

    mode='N': A (NbA, na, ka), B (NbB, fac*ka, kb) with fac = NbA//NbB.
              Returns (NbB, fac*na, kb), the product of B with the block
              diagonal whose fac consecutive A blocks sit on the diagonal of
              output block i.
    mode='T': A (Nb, n, k), B (Nb, n, kb) -> (Nb, k, kb), C[i] = A[i]^T B[i].

    The 'N' branch used to materialize that (fac*na, fac*ka) block diagonal
    explicitly, one output block at a time, and multiply through its zeros:
    fac times the necessary flops (2x normally, 4x with quad=True), a zeroed
    buffer of fac^2 the useful size per block, and 1 + fac launches per
    block on top.  Block i's j-th row stripe is just A[fac*i+j] @ B[i]'s
    j-th row stripe, so the whole thing is one bmm over a reshape -- the
    same transformation HBStorch.block_mult_and_reduce already applies to
    the identical pattern in the compression sweep.

    `device` is accepted for call-site compatibility and ignored; the result
    carries A's device and dtype rather than the default dtype.
    """
    NbA, na, ka = A.shape
    NbB, nb, kb = B.shape

    if mode == 'N':
        if NbB == 0 or NbA % NbB:
            raise ValueError(
                f"sparse_block_mult_tens: NbA={NbA} is not a multiple of "
                f"NbB={NbB}")
        fac = NbA // NbB
        if nb != fac * ka:
            raise ValueError(
                f"sparse_block_mult_tens: mode='N' expects B with {fac*ka} "
                f"rows (fac={fac} x ka={ka}), got {nb}")
        # B (NbB, fac*ka, kb) -> (NbB*fac, ka, kb): the row stripes of each
        # output block, in the same order as A's blocks.
        return torch.bmm(A, B.reshape(NbA, ka, kb)).reshape(NbB, fac * na, kb)

    if mode == 'T':
        if NbA != NbB:
            raise ValueError(
                f"sparse_block_mult_tens: mode='T' needs NbA == NbB, got "
                f"{NbA} and {NbB}")
        if na != nb:
            raise ValueError(
                f"sparse_block_mult_tens: mode='T' expects B with {na} rows, "
                f"got {nb}")
        return torch.bmm(A.mT, B)

    raise ValueError("mode not recognized")


def block_diag_add_tens(A, B, device=None, inplace=False):
    """Add the finer block diagonal B into the coarser one A.

    A: (NbA, fac*nB, fac*kB), B: (NbA*fac, nB, kB).  B's blocks
    fac*i .. fac*i+fac-1 land on the diagonal of A's block i.  Returns the
    sum in reduced form.

    The nested Python loop wrote NbA*fac tiny in-place adds through chained
    slices.  Viewing A as (NbA, fac, nB, fac, kB) exposes those targets as
    the (dim 1, dim 3) diagonal, which `torch.diagonal` gives as a strided
    VIEW, so one add_ covers the whole level.

    inplace
    -------
    The old version bound C = A and returned A mutated, while reading as a
    pure function.  Both call sites pass a tensor they have just built and
    do not read again, so the mutation was harmless there, but nothing in
    the signature said so.  The default is now a copy; pass inplace=True
    where the caller owns A and the copy of a (NbA, fac*nB, fac*kB) tensor
    is worth avoiding, which at the coarse levels it is.

    `device` is accepted for call-site compatibility and ignored.
    """
    NbA, nA, kA = A.shape
    NbB, nB, kB = B.shape

    if kA < kB:
        raise ValueError(
            "block_diag_add_tens: A must be the COARSER operand (kA >= kB), "
            f"got kA={kA}, kB={kB}; swap the arguments")
    if kB == 0 or kA % kB:
        raise ValueError(
            f"block_diag_add_tens: kA={kA} is not a multiple of kB={kB}")
    fac = kA // kB
    if nA != fac * nB:
        raise ValueError(
            f"block_diag_add_tens: expected A with {fac*nB} rows per block "
            f"(fac={fac} x nB={nB}), got {nA}")
    if NbB != NbA * fac:
        raise ValueError(
            f"block_diag_add_tens: expected {NbA*fac} B blocks "
            f"(NbA={NbA} x fac={fac}), got {NbB}")

    if inplace:
        if not A.is_contiguous():
            raise ValueError(
                "block_diag_add_tens: inplace=True needs a contiguous A; the "
                "diagonal view below cannot be taken otherwise")
        C = A
    else:
        C = A.clone()
        if not C.is_contiguous():
            C = C.contiguous()

    # (NbA, fac, nB, fac, kB); the diagonal over the two fac axes is
    # (NbA, nB, kB, fac), with the block index j last.
    C.view(NbA, fac, nB, fac, kB).diagonal(dim1=1, dim2=3).add_(
        B.reshape(NbA, fac, nB, kB).permute(0, 2, 3, 1))
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
            # Rhat was built one line up and is read nowhere else, so the add
            # stays in place; the default would clone it.
            Rhat = block_diag_add_tens(Rhat,R_22,device,inplace=True)
            
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
