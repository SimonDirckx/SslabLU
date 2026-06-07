import numpy as np
import matplotlib.pyplot as plt

import SOMS3D_csr
import torch
import matAssembly.HBS.slabTree as slabTree
import matAssembly.HBS.HBStorch_strong as HBStorch
from scipy.sparse.linalg import LinearOperator
import solver.stencil.stencilSolver as stencil
import solver.stencil.geom as geom
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
import mumps
import scipy.sparse as sparse
import time
import os
from scipy.sparse.linalg import gmres

def rss_gb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6   # kB -> GB
    return -1.0

def _enable_blr(ctx, blr_tol):
    ctx.mumps_instance.icntl[35] = 1
    ctx.mumps_instance.cntl[7] = blr_tol
    # inst.icntl[36] = 0      # BLR variant (UFSC); leave default unless tuning


def setup_mumps(Sii, blr=False, blr_tol=1e-8):
    ctx = mumps.Context()
    ctx.analyze(Sii)
    if blr:
        _enable_blr(ctx, blr_tol)   # set BEFORE analyze so estimates account for BLR
    ctx.analyze(Sii)                # symbolic factorization (sparsity pattern only)
    ctx.factor(Sii)                 # numeric factorization (BLR-compressed if enabled)
    return ctx


def setup_mumps_transpose(Sii, ctx=None, sym=False, blr=False, blr_tol=1e-8):
    ctxT = mumps.Context()
    ctxT.analyze(Sii.T)
    if blr:
        _enable_blr(ctxT, blr_tol)
    ctxT.analyze(Sii.T)
    ctxT.factor(Sii.T)
    return ctxT


def bc_laplace(p):
    """Free-space Green's function with source at (-0.5, -0.5, -0.5)."""
    r = np.sqrt((p[:,0]+.5)**2+(p[:,1]+.5)**2+(p[:,2]+.5)**2)
    return 1./(4*np.pi*r)
def bc_helmholtz(p,kh):
    """Free-space Green's function with source at (-0.5, -0.5, -0.5)."""
    r = np.sqrt((p[:,0]+.5)**2+(p[:,1]+.5)**2+(p[:,2]+.5)**2)
    return np.real(np.exp(1j*kh*r)/(4*np.pi*r))


def match_rows(A, B, decimals=9):
    """For each row of A return the index of the matching row in B (stencil path).

    Used to line up DOF orderings by coordinate, e.g. the physical x = 0 / x = Lx
    faces against the interface plane, or the strictly-interior interface points
    against their positions inside the full plane.  A and B must contain the same
    set of points (matched to `decimals` places)."""
    lut = {}
    for j in range(B.shape[0]):
        lut[tuple(np.round(B[j], decimals))] = j
    idx = np.empty(A.shape[0], dtype=np.int64)
    for i in range(A.shape[0]):
        idx[i] = lut[tuple(np.round(A[i], decimals))]
    return idx


class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.resList=[]
    def __call__(self, rk=None):
        self.niter += 1
        self.resList+=[rk]
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


import argparse
import re

def _ints3(tokens):
    """Parse a triple of ints given either space- or comma-separated."""
    flat = [t for t in re.split(r"[,\s]+", " ".join(tokens).strip()) if t]
    if len(flat) != 3:
        raise SystemExit(f"order expects 3 values (got {flat})")
    return [int(t) for t in flat]

def _nums3(tokens):
    """Parse a triple of floats; each entry may be a fraction like '1/16'."""
    flat = [t for t in re.split(r"[,\s]+", " ".join(tokens).strip()) if t]
    if len(flat) != 3:
        raise SystemExit(f"shape expects 3 values (got {flat})")
    out = []
    for t in flat:
        if "/" in t:
            a, b = t.split("/")
            out.append(float(a) / float(b))
        else:
            out.append(float(t))
    return out

_parser = argparse.ArgumentParser(
    description="Thin-slab interface-map GMRES solve (LU and HBS-compressed).")
_parser.add_argument("--type", choices=["SOMS", "stencil"], default="stencil",
                     help="discretization / solver type")
_parser.add_argument("--order", nargs="+", default=None,
                     help="(nx ny nz) for stencil [nx is overridden = int(ny*Lx)+1] "
                          "or (px py pz) for SOMS; space- or comma-separated")
_parser.add_argument("--shape", nargs="+", default=["1/16", "1", "1"],
                     help="domain extents Lx Ly Lz (fractions like 1/16 allowed)")
_parser.add_argument("--admissibility", choices=["full", "partial"], default="full",
                     help="HBS tree adjacency / admissibility")
_parser.add_argument("--gmres-iters", dest="gmres_iters", type=int, default=100,
                     help="max GMRES iterations (sets maxiter & restart); 0 skips the GMRES solve")
args = _parser.parse_args()

solve_method = args.type
# type-appropriate default order if none supplied
if args.order is None:
    order = [9, 128, 128] if solve_method == "stencil" else [6, 6, 6]
else:
    order = _ints3(args.order)
Lx, Ly, Lz = _nums3(args.shape)
admissibility = args.admissibility
gmres_iters = args.gmres_iters
cx = Lx/2
slabGeom = geom.BoxGeometry(np.array([[0,0,0],[Lx,Ly,Lz]]))


kh = 5.2

def  c11(p):
    return np.ones_like(p[:,0])
def  c22(p):
    return np.ones_like(p[:,0])
def  c33(p):
    return np.ones_like(p[:,0])
def  c(p):
    return kh*kh*np.ones_like(p[:,0])
HH = pdo.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)


coeffs = {'c11': 1., 'c22': 1., 'c33': 1.,'c':kh**2}
print("============BUILDING SOLVER============")
print(f"type={solve_method}  order={order}  shape=({Lx},{Ly},{Lz})  "
      f"admissibility={admissibility}")

# ###########################################################################
# #############################   SOMS PATH   ###############################
# ###########################################################################
if solve_method == 'SOMS':
    px, py, pz = order            # polynomial order per block (CLI --order)
    nbx = 2                       # 2 blocks in x -> interface at the centre x = cx
    nby = 16
    nbz = 16
    Sii, Sib, ftild, XYtot, Ii, Ib, wi,wb = SOMS3D_csr.SOMS_solver_sparse(
         px, py, pz, nbx, nby, nbz, Lx, Ly, Lz,
         coeffs, True, None, weighted=False)

    XXi = XYtot[Ii,:]
    XXb = XYtot[Ib,:]
    Jc = np.where(XXi[:,0]==cx)[0]
    XXc = XXi[Jc,:]
    Jl = np.where(XXb[:,0]==0)[0]
    XXl = XXb[Jl,:]
    Jr = np.where(XXb[:,0]==Lx)[0]

    print("|Jl| = ",len(Jl))
    print("|Jr| = ",len(Jr))
    print("|Jc| = ",len(Jc))


    XXr = XXb[Jr,:]
    Jb = np.array([i for i in range(XXb.shape[0]) if i not in Jl and i not in Jr],dtype=np.int64)

    uc = bc_helmholtz(XXc,kh)
    ul = bc_helmholtz(XXl,kh)
    ur = bc_helmholtz(XXr,kh)
    ub = bc_helmholtz(XXb[Jb,:],kh)

    tic_lu = time.time()
    ctx = setup_mumps(Sii,blr=False)
    ctxT = setup_mumps_transpose(Sii,blr=False)
    print("LU decomposition total time = ", time.time()-tic_lu)


    BLK = 32                                   # tune; see note below
    ctx.mumps_instance.icntl[27]  = BLK        # one wide BLAS-3 block per chunk
    ctxT.mumps_instance.icntl[27] = BLK

    def smatmat(v,J , transpose=False):
        v_tmp = v[:, None] if v.ndim == 1 else v
        k = v_tmp.shape[1]


        if not transpose:
            Sib_J = Sib[:,J].tocsc()
            out = np.zeros((len(Jc), k))
            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                rhs = (Sib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
                sol = ctx._solve_sparse(rhs)              # dense (len(Ii) x BLK) — bounded
                out[:, c] = -sol[Jc, :]
                del rhs, sol
                ctx.mumps_instance.icntl[20]=0
        else:
            Sib_J_T = Sib[:,J].T.tocsr()
            out = np.zeros((len(Jr), k))

            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                w  = v_tmp[:, c]; bw = w.shape[1]; m = len(Jc)
                rhs = sparse.csc_matrix(
                    (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                    shape=(len(Ii), bw))
                sol = ctxT._solve_sparse(rhs)
                out[:, c] = -Sib_J_T @ sol
                del rhs, sol
                ctx.mumps_instance.icntl[20]=0
        return out.flatten() if v.ndim == 1 else out

    LinOp_r = LinearOperator(shape=(len(Jc),len(Jr)),\
        matvec = lambda v:smatmat(v,Jr), rmatvec = lambda v:smatmat(v,Jr,transpose=True),\
        matmat = lambda v:smatmat(v,Jr), rmatmat = lambda v:smatmat(v,Jr,transpose=True))
    LinOp_l = LinearOperator(shape=(len(Jc),len(Jl)),\
        matvec = lambda v:smatmat(v,Jl), rmatvec = lambda v:smatmat(v,Jl,transpose=True),\
        matmat = lambda v:smatmat(v,Jl), rmatmat = lambda v:smatmat(v,Jl,transpose=True))

    err = uc-LinOp_l@ul-LinOp_r@ur-(ctx.solve(-(Sib[:,Jb]@ub)))[Jc]
    print("err = ",np.linalg.norm(err))

    ndofs_if = len(Jc)
    ndslab = int(1/cx)-1
    XXif = np.zeros((ndslab*ndofs_if,3))
    rhs = np.zeros((ndslab*ndofs_if,))
    for i in range(ndslab):
        XXc = XXi[Jc] + np.array([i*cx, 0., 0.])
        XXif[i*ndofs_if:(i+1)*ndofs_if, :] = XXc

        XXb_loc = XXb + np.array([i*cx, 0., 0.])
        XXl = XXb[Jl] + np.array([i*cx, 0., 0.])

        XXr = XXb[Jr] + np.array([i*cx, 0., 0.])
        uj = bc_helmholtz(XXb_loc, kh)
        if i == 0:
            XXr = XXb[Jr]
            XXc = XXi[Jc]

            br = bc_helmholtz(XXr,kh)
            uc = bc_helmholtz(XXc,kh)
            ur = LinOp_r@br
            Jb0 = np.array([k for k in range(XXb.shape[0]) if k not in Jr])
            blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
            print("err slab ",i," = ",np.linalg.norm(uc-blk-ur))
        elif i == ndslab - 1:             # physical far face is known -> keep Jr, drop Jl
            Jb0 = np.array([k for k in range(XXb.shape[0]) if k not in Jl])
            blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
            XXl = XXb[Jl] + np.array([i*cx, 0., 0.])
            XXc = XXi[Jc] + np.array([i*cx, 0., 0.])
            bl = bc_helmholtz(XXl,kh)
            uc = bc_helmholtz(XXc,kh)
            ul = LinOp_l@bl
            print("err slab ",i," = ",np.linalg.norm(uc-blk-ul))
        else:                             # interior slab: only the (y,z) faces are known
            blk = -ctx.solve(Sib[:, Jb] @ uj[Jb])[Jc]
            bl = bc_helmholtz(XXl,kh)
            br = bc_helmholtz(XXr,kh)
            uc = bc_helmholtz(XXc,kh)
            ul = LinOp_l@bl
            ur = LinOp_r@br
            print("err slab ",i," = ",np.linalg.norm(uc-blk-ul-ur))
        rhs[i*ndofs_if:(i+1)*ndofs_if] = blk

    ui = bc_helmholtz(XXif,kh)

    def apply_balance(u):
        if u.ndim == 1:
            utmp = u[:,None]
        else:
            utmp = u
        out = utmp.copy()
        for j in range(ndslab):
            if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_l(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
            if j < ndslab-1:      out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_r(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
        if u.ndim == 1:
            out = out.flatten()
        return out

    A_balance = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                               matvec=apply_balance, dtype=float)

    N = A_balance.shape[0]
    v = np.random.standard_normal((N,))
    tic =time.time()
    bb = A_balance@v
    print("matvec time = ",time.time()-tic)
    gInfo = gmres_info()
    u = bc_helmholtz(XXif,kh)
    res = A_balance@u-rhs
    print("res = ",np.linalg.norm(res))
    if gmres_iters > 0:
        tic = time.time()
        uhat,_   = gmres(A_balance,rhs,rtol=1e-8,callback=gInfo,maxiter=gmres_iters,restart=gmres_iters)
        niter = gInfo.niter
        print("time = ",time.time()-tic)
        print("niter = ",niter)
        print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))
    else:
        print("GMRES solve skipped (gmres_iters = 0)")

    print("===============  HBS version  ===============")

    if torch.cuda.is_available():
        device = 'cuda'          # or 'cuda:0' to pin a specific GPU
    else:
        device = 'cpu'

    tree = slabTree.slabTree(XXc,False,py*pz,adjacency=admissibility)

    SSr = HBStorch.HBSMAT(device=device,tree=tree)
    SSl = HBStorch.HBSMAT(device=device,tree=tree)

    print("Sl shape = ",LinOp_l.shape)
    print("Sr shape = ",LinOp_r.shape)
    nl = len(tree.get_box_inds(tree.get_leaves()[0]))
    rk = 50
    s = 10*max(2*rk,nl)+rk+10
    N = LinOp_r.shape[0]
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Nb = N//nl
    Y = LinOp_r@Om
    Z = LinOp_r.T@Psi
    tic = time.time()
    SSr.construct(rk,Om,Psi,Y,Z,fast=True)
    print("HBS done in :",time.time()-tic)
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Nb = N//nl
    Y = LinOp_l@Om
    Z = LinOp_l.T@Psi
    tic = time.time()
    SSl.construct(rk,Om,Psi,Y,Z,fast=True)
    print("HBS done in :",time.time()-tic)

    def apply_balance_HBS(u):
        if u.ndim == 1:
            utmp = u[:,None]
        else:
            utmp = u
        out = utmp.copy()
        for j in range(ndslab):
            if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= SSl@(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
            if j < ndslab-1:      out[j*ndofs_if:(j+1)*ndofs_if,:] -= SSr@(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
        if u.ndim == 1:
            out = out.flatten()
        return out

    A_balance_HBS = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                               matvec=apply_balance_HBS, dtype=float)
    N = A_balance.shape[0]
    v = np.random.standard_normal((N,))
    tic =time.time()
    bb = A_balance_HBS@v
    print("matvec time = ",time.time()-tic)
    gInfo = gmres_info()
    u = bc_helmholtz(XXif,kh)
    res = A_balance_HBS@u-rhs
    print("res = ",np.linalg.norm(res))
    if gmres_iters > 0:
        tic = time.time()
        uhat,_   = gmres(A_balance_HBS,rhs,rtol=1e-8,callback=gInfo,maxiter=gmres_iters,restart=gmres_iters)
        niter = gInfo.niter
        print("time = ",time.time()-tic)
        print("niter = ",niter)
        print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))
    else:
        print("GMRES solve skipped (gmres_iters = 0)")

# ###########################################################################
# ###########################   STENCIL PATH   ##############################
# ###########################################################################
elif solve_method == 'stencil':
    nx, ny, nz = order            # ny, nz from CLI --order (nx is overridden below)
    nx = int(ny*Lx) + 1           # derive nx so the centre plane x = cx lands on-grid
    print("stencil nx (derived from ny, Lx) = ", nx)
    ord = [nx,ny,nz]
    solver = stencil.stencilSolver(HH,slabGeom,ord)
    Sii = solver.Aii
    Sib = solver.Aix
    XYtot = solver.XX
    Ii = np.asarray(solver.Ji)
    Ib = np.asarray(solver.Jx)
    wi = np.ones((Sii.shape[0],))
    wb = np.ones((Sib.shape[1],))
    tree_leaf = 8*8            # max leaf size for the HBS tree on the interface plane

    # -----------------------------------------------------------------------
    # Interface / face index machinery.
    #   Jc       : strictly-interior interface DOFs (subset of the interior Ii)
    #              on the plane x = cx.
    #   Jc_large : the FULL plane x = cx, INCLUDING its (y,z)-boundary ring.
    #              We carry the whole plane so the HBS tree (built on Jc_large)
    #              is clean / uniform.
    #   Jc_inJc  : positions of Jc inside Jc_large.  The interior solve produces
    #              len(Jc) values which we scatter into these rows; the ring rows
    #              carry KNOWN Dirichlet data (handled in the RHS, not as zeros).
    # -----------------------------------------------------------------------
    tol = 1e-9
    XXi = XYtot[Ii,:]
    XXb = XYtot[Ib,:]

    Jc = np.where(np.abs(XXi[:,0]-cx) < tol)[0]               # interior interface DOFs
    Jl = np.where(np.abs(XXb[:,0]-0.) < tol)[0]               # physical x = 0  face
    Jr = np.where(np.abs(XXb[:,0]-Lx) < tol)[0]               # physical x = Lx face
    Jb = np.setdiff1d(np.arange(XXb.shape[0]),
                      np.concatenate([Jl, Jr])).astype(np.int64)   # the four (y,z) faces

    Jc_large = np.where(np.abs(XYtot[:,0]-cx) < tol)[0]       # full interface plane
    Jc_inJc =  np.where((XYtot[Jc_large,1] > tol) &\
                    (XYtot[Jc_large,1] < Ly-tol) &\
                    (XYtot[Jc_large,2] > tol) &\
                    (XYtot[Jc_large,2] < Lz-tol))[0]   # x = Lx/2
    ndofs_if = len(Jc_large)                                  # size of one interface block

    print("|Jl| = ", len(Jl))
    print("|Jr| = ", len(Jr))
    print("|Jc| = ", len(Jc))
    print("|Jc_large| = ", ndofs_if)
    print("|Jc_inJc|  = ", len(Jc_inJc))

    assert len(Jl) == ndofs_if and len(Jr) == ndofs_if, "face/plane size mismatch"
    assert np.allclose(XXb[Jr][:,1:3], XYtot[Jc_large][:,1:3]), "Jr not aligned to Jc_large"
    assert np.allclose(XXb[Jl][:,1:3], XYtot[Jc_large][:,1:3]), "Jl not aligned to Jc_large"

    def scatter(vec_Jc):
        """Scatter a length-len(Jc) interior result into a length-ndofs_if block."""
        out = np.zeros(ndofs_if)
        out[Jc_inJc] = vec_Jc
        return out

    tic_lu = time.time()
    ctx = setup_mumps(Sii,blr=False)
    ctxT = setup_mumps_transpose(Sii,blr=False)
    print("LU decomposition total time = ", time.time()-tic_lu)

    BLK = 32                                   # tune; see note below
    ctx.mumps_instance.icntl[27]  = BLK        # one wide BLAS-3 block per chunk
    ctxT.mumps_instance.icntl[27] = BLK

    def smatmat(v, J, transpose=False):
        """Apply the interface map  -(Sii^{-1} Sib_J)  (or its transpose).

        Forward : boundary data on face J (Jc_large-ordered)  ->  interface block.
                  The interior solve fills the Jc_inJc rows; the (y,z)-boundary
                  rows of the block stay 0.
        Transpose: interface block -> face J.  Only the Jc_inJc rows of the input
                  feed the (transposed) interior solve.
        """
        v_tmp = v[:, None] if v.ndim == 1 else v
        k = v_tmp.shape[1]

        if not transpose:
            Sib_J = Sib[:, J].tocsc()
            out = np.zeros((ndofs_if, k))
            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                rhs = (Sib_J @ sparse.csc_matrix(v_tmp[:, c])).tocsc()
                sol = ctx._solve_sparse(rhs)              # dense (len(Ii) x BLK) — bounded
                out[Jc_inJc, c] = -sol[Jc, :]
                del rhs, sol
                ctx.mumps_instance.icntl[20] = 0
        else:
            Sib_J_T = Sib[:, J].T.tocsr()
            out = np.zeros((ndofs_if, k))
            for s in range(0, k, BLK):
                c = slice(s, min(s + BLK, k))
                w  = v_tmp[Jc_inJc, c]; bw = w.shape[1]; m = len(Jc)
                rhs = sparse.csc_matrix(
                    (w.ravel(order="F"), np.tile(Jc, bw), np.arange(0, m*bw+1, m)),
                    shape=(len(Ii), bw))
                sol = ctxT._solve_sparse(rhs)
                out[:, c] = -Sib_J_T @ sol
                del rhs, sol
                ctx.mumps_instance.icntl[20] = 0
        return out.flatten() if v.ndim == 1 else out

    LinOp_r = LinearOperator(shape=(ndofs_if,len(Jr)),\
        matvec = lambda v:smatmat(v,Jr), rmatvec = lambda v:smatmat(v,Jr,transpose=True),\
        matmat = lambda v:smatmat(v,Jr), rmatmat = lambda v:smatmat(v,Jr,transpose=True))
    LinOp_l = LinearOperator(shape=(ndofs_if,len(Jl)),\
        matvec = lambda v:smatmat(v,Jl), rmatvec = lambda v:smatmat(v,Jl,transpose=True),\
        matmat = lambda v:smatmat(v,Jl), rmatmat = lambda v:smatmat(v,Jl,transpose=True))

    # single-box sanity check: the interface map should reproduce the trace at x = cx
    ul = bc_helmholtz(XXb[Jl], kh)
    ur = bc_helmholtz(XXb[Jr], kh)
    ub = bc_helmholtz(XXb[Jb], kh)
    uc_full = scatter(bc_helmholtz(XXi[Jc], kh))
    pred = LinOp_l@ul + LinOp_r@ur + scatter((ctx.solve(-(Sib[:,Jb]@ub)))[Jc])
    err = uc_full - pred
    print("err = ", np.linalg.norm(err))

    # -----------------------------------------------------------------------
    # Slab RHS + manufactured (Green's function) reference solution.
    # Interfaces are spaced cx apart; LinOp_l / LinOp_r map a neighbour interface
    # (cx away on either side) onto the current interface centerline.
    # The (y,z)-boundary ring DOFs of each interface block are GLOBAL Dirichlet
    # unknowns: their balance-operator row is the identity (LinOp produces 0
    # there), so we pin them to the known trace in the RHS, which makes them
    # couple correctly into the neighbouring slabs through Sib[:,Jl/Jr].
    # -----------------------------------------------------------------------
    ndslab = int(round(1./cx)) - 1
    XXif = np.zeros((ndslab*ndofs_if,3))
    rhs  = np.zeros((ndslab*ndofs_if,))
    u_true = np.zeros((ndslab*ndofs_if,))
    for i in range(ndslab):
        shift = np.array([i*cx, 0., 0.])

        XXif[i*ndofs_if:(i+1)*ndofs_if, :] = XYtot[Jc_large] + shift
        trace_full = bc_helmholtz(XYtot[Jc_large] + shift, kh)   # interior AND ring
        u_true[i*ndofs_if:(i+1)*ndofs_if]  = trace_full

        XXb_loc = XXb + shift
        uj = bc_helmholtz(XXb_loc, kh)

        if i == 0:                        # physical x=0 face known -> keep Jl, drop Jr
            Jb0 = np.setdiff1d(np.arange(XXb.shape[0]), Jr).astype(np.int64)
            blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
            br = bc_helmholtz(XXb[Jr] + shift, kh)
            uc = bc_helmholtz(XXi[Jc] + shift, kh)
            ur = (LinOp_r @ br)[Jc_inJc]
            print("err slab ", i, " = ", np.linalg.norm(uc - blk - ur))
        elif i == ndslab - 1:             # physical far face known -> keep Jr, drop Jl
            Jb0 = np.setdiff1d(np.arange(XXb.shape[0]), Jl).astype(np.int64)
            blk = -ctx.solve(Sib[:, Jb0] @ uj[Jb0])[Jc]
            bl = bc_helmholtz(XXb[Jl] + shift, kh)
            uc = bc_helmholtz(XXi[Jc] + shift, kh)
            ul = (LinOp_l @ bl)[Jc_inJc]
            print("err slab ", i, " = ", np.linalg.norm(uc - blk - ul))
        else:                             # interior slab: only the (y,z) faces are known
            blk = -ctx.solve(Sib[:, Jb] @ uj[Jb])[Jc]
            bl = bc_helmholtz(XXb[Jl] + shift, kh)
            br = bc_helmholtz(XXb[Jr] + shift, kh)
            uc = bc_helmholtz(XXi[Jc] + shift, kh)
            ul = (LinOp_l @ bl)[Jc_inJc]
            ur = (LinOp_r @ br)[Jc_inJc]
            print("err slab ", i, " = ", np.linalg.norm(uc - blk - ul - ur))

        # interior rows -> centerline solve; ring rows -> known Dirichlet trace
        blk_full = trace_full.copy()
        blk_full[Jc_inJc] = blk
        rhs[i*ndofs_if:(i+1)*ndofs_if] = blk_full

    def apply_balance(u):
        if u.ndim == 1:
            utmp = u[:,None]
        else:
            utmp = u
        out = utmp.copy()
        for j in range(ndslab):
            if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_l@(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
            if j < ndslab-1:   out[j*ndofs_if:(j+1)*ndofs_if,:] -= LinOp_r@(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
        if u.ndim == 1:
            out = out.flatten()
        return out

    A_balance = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                               matvec=apply_balance, dtype=float)

    N = A_balance.shape[0]
    v = np.random.standard_normal((N,))
    tic =time.time()
    bb = A_balance@v
    print("matvec time = ",time.time()-tic)
    gInfo = gmres_info()
    u = u_true
    res = A_balance@u-rhs
    print("res = ",np.linalg.norm(res))
    if gmres_iters > 0:
        tic = time.time()
        uhat,_   = gmres(A_balance,rhs,rtol=1e-8,callback=gInfo,maxiter=gmres_iters,restart=gmres_iters)
        niter = gInfo.niter
        print("time = ",time.time()-tic)
        print("niter = ",niter)
        print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))
    else:
        print("GMRES solve skipped (gmres_iters = 0)")

    print("===============  HBS version  ===============")

    if torch.cuda.is_available():
        device = 'cuda'          # or 'cuda:0' to pin a specific GPU
    else:
        device = 'cpu'

    # tree on the FULL interface plane Jc_large (clean / uniform leaves)
    tree = slabTree.slabTree(XYtot[Jc_large],False,tree_leaf,adjacency=admissibility)

    SSr = HBStorch.HBSMAT(device=device,tree=tree)
    SSl = HBStorch.HBSMAT(device=device,tree=tree)

    print("Sl shape = ",LinOp_l.shape)
    print("Sr shape = ",LinOp_r.shape)
    nl = len(tree.get_box_inds(tree.get_leaves()[0]))
    rk = 20
    s = 10*max(2*rk,nl)+rk+10
    N = LinOp_r.shape[0]
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Nb = N//nl
    Y = LinOp_r@Om
    Z = LinOp_r.T@Psi
    tic = time.time()
    SSr.construct(rk,Om,Psi,Y,Z,fast=True)
    print("HBS done in :",time.time()-tic)
    Om = np.random.standard_normal((N,s))
    Psi = np.random.standard_normal((N,s))
    Nb = N//nl
    Y = LinOp_l@Om
    Z = LinOp_l.T@Psi
    tic = time.time()
    SSl.construct(rk,Om,Psi,Y,Z,fast=True)
    print("HBS done in :",time.time()-tic)

    def apply_balance_HBS(u):
        if u.ndim == 1:
            utmp = u[:,None]
        else:
            utmp = u
        out = utmp.copy()
        for j in range(ndslab):
            if j > 0:          out[j*ndofs_if:(j+1)*ndofs_if,:] -= SSl@(utmp[(j-1)*ndofs_if:j*ndofs_if,:])
            if j < ndslab-1:   out[j*ndofs_if:(j+1)*ndofs_if,:] -= SSr@(utmp[(j+1)*ndofs_if:(j+2)*ndofs_if,:])
        if u.ndim == 1:
            out = out.flatten()
        return out

    A_balance_HBS = LinearOperator(shape=(ndslab*ndofs_if, ndslab*ndofs_if),
                               matvec=apply_balance_HBS, dtype=float)
    N = A_balance.shape[0]
    v = np.random.standard_normal((N,))
    tic =time.time()
    bb = A_balance_HBS@v
    print("matvec time = ",time.time()-tic)
    gInfo = gmres_info()
    u = u_true
    res = A_balance_HBS@u-rhs
    print("res = ",np.linalg.norm(res))
    if gmres_iters > 0:
        tic = time.time()
        uhat,_   = gmres(A_balance_HBS,rhs,rtol=1e-8,callback=gInfo,maxiter=gmres_iters,restart=gmres_iters)
        niter = gInfo.niter
        print("time = ",time.time()-tic)
        print("niter = ",niter)
        print("u err = ",np.linalg.norm(uhat-u)/np.linalg.norm(u))
    else:
        print("GMRES solve skipped (gmres_iters = 0)")

else:
    raise ValueError("solve method not recognized")