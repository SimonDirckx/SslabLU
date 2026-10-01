import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg   import LinearOperator
from solver.stencil.stencilSolver import stencilSolver as stencil
from solver.spectral.spectralSolver import spectralSolver as spectral
import solver.stencil.geom as stencilGeom
import solver.spectral.geom as spectralGeom
import solver.HPSInterp as interp
import mumps

# Things we need to add:
from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsaltGeom


from time import time


# =========================================================================== #
#  MUMPS layer
#
#  Ported from thinSlab3D.py.  The central point: A^{-T} does NOT need its own
#  factorization.  MUMPS solves A^T x = b against the *existing* A factors when
#  ICNTL(9) != 1, which halves both LU memory and factorization time relative
#  to building a second context for A^T.  The explicit-A^T path is kept as an
#  opt-in fallback (`use_ctxT`) for when the transposed-solve path needs to be
#  ruled out as a source of error.
#
#  INDEXING NOTE.  python-mumps wraps icntl/cntl/info/infog with
#  `__getitem__ -> array[key - 1]`, i.e. genuine 1-based Fortran indexing, so
#  `icntl[9]` really is MUMPS ICNTL(9).  Index 0 is NOT a no-op: it decrements
#  a raw C pointer and writes to `comm_fortran`, the struct field sitting
#  immediately before icntl[60].  thinSlab3D.py sets icntl[0] in four places;
#  those writes corrupt the communicator and silence nothing.  Output control
#  is ICNTL(1..4) -- see _silence_mumps below.
# =========================================================================== #

# ICNTL(1) error stream, (2) diagnostic/warning, (3) global info, (4) verbosity
_ICNTL_OUT_STREAMS = (1, 2, 3, 4)

# ICNTL(9):  1 -> solve A x = b   (default)
#            0 -> solve A^T x = b against the same factors
_ICNTL_TRANSPOSE = 9
_ICNTL_SPARSE_RHS = 20      # ICNTL(20): 1 = sparse RHS
_ICNTL_BLOCK_SIZE = 27      # ICNTL(27): blocking size for multiple RHS
_ICNTL_BLR = 35             # ICNTL(35): block low-rank
_CNTL_BLR_TOL = 7           # CNTL(7):  BLR dropping parameter


def _silence_mumps(ctx):
    """Mute MUMPS' internal output streams (ICNTL(1..4))."""
    for i in _ICNTL_OUT_STREAMS:
        ctx.mumps_instance.icntl[i] = 0


def _enable_blr(ctx, blr_tol):
    """Block-low-rank factorization with dropping tolerance `blr_tol`."""
    ctx.mumps_instance.icntl[_ICNTL_BLR] = 1
    ctx.mumps_instance.cntl[_CNTL_BLR_TOL] = blr_tol
    # ICNTL(36) selects the BLR variant (UFSC); leave at default unless tuning.


def setup_mumps(A, ordering="metis", blr_tol=0.0, block_size=None, verbose=0):
    """Analyze + factorize A in one MUMPS context.

    Returns (ctx, time_analysis, time_factor).

    The context is left configured for forward solves; use `mumps_solve(ctx, b,
    transpose=True)` for A^{-T} applies against the same factors.

    Parameters
    ----------
    ordering : MUMPS ordering for the analysis ('metis', 'auto', 'amd', ...).
        Falls back to 'auto' if the requested ordering is not in this MUMPS
        build.
    blr_tol : if > 0, enable BLR compression with this dropping tolerance.
    block_size : ICNTL(27), the blocking size for multiple right-hand sides.
        Setting this to the number of columns you sample with gives one wide
        BLAS-3 block per chunk.
    verbose : >1 leaves MUMPS' own diagnostics on.
    """
    ctx = mumps.Context(verbose=bool(verbose > 1))
    # symmetric=False throughout: the symmetric/Cholesky path is deliberately
    # not used here.
    ctx.set_matrix(A, symmetric=False)

    tic = time()
    try:
        ctx.analyze(ordering=ordering)
    except Exception:
        if ordering == "auto":
            raise
        # e.g. METIS not compiled into this MUMPS build
        ctx.analyze(ordering="auto")
    time_analysis = time() - tic

    if verbose < 2:
        _silence_mumps(ctx)
    if blr_tol and blr_tol > 0:
        _enable_blr(ctx, blr_tol)

    tic = time()
    # reuse_analysis=True is the whole point of having called analyze():
    # factor() re-runs the analysis by default, so without this flag the
    # symbolic step is paid twice.
    ctx.factor(reuse_analysis=True)
    time_factor = time() - tic

    if verbose < 2:
        _silence_mumps(ctx)
    if block_size is not None:
        ctx.mumps_instance.icntl[_ICNTL_BLOCK_SIZE] = int(block_size)

    return ctx, time_analysis, time_factor


def setup_mumps_transpose(A, ordering="metis", blr_tol=0.0, block_size=None,
                          verbose=0):
    """Factor A^T explicitly into its own context (opt-in; see `use_ctxT`).

    Costs a second analysis + factorization and ~2x LU memory.  Only worth it
    to rule out MUMPS' transposed-solve path as a source of error.
    """
    # .T of a CSR matrix is a CSC view; convert so MUMPS receives the same
    # storage type as on the forward path (values unchanged).
    AT = sp.csr_matrix(A.T) if sp.issparse(A) else np.asarray(A).T
    return setup_mumps(AT, ordering=ordering, blr_tol=blr_tol,
                       block_size=block_size, verbose=verbose)


def mumps_solve(ctx, b, transpose=False):
    """Solve A x = b, or A^T x = b, against an existing factorization.

    No context manager: ICNTL(9) is flipped, the solve runs, and the flag is
    restored in a finally block.  Reentrancy is the same as the original
    contextmanager version (i.e. do not interleave two transposed solves on
    one context from different threads).
    """
    if not transpose:
        return ctx.solve(b)

    inst = ctx.mumps_instance
    prev = inst.icntl[_ICNTL_TRANSPOSE]
    inst.icntl[_ICNTL_TRANSPOSE] = 0          # A^T x = b
    try:
        return ctx.solve(b)
    finally:
        inst.icntl[_ICNTL_TRANSPOSE] = prev   # restore A x = b


def mumps_solve_sparse(ctx, b, transpose=False):
    """As `mumps_solve` but for a sparse (csc) right-hand side.

    python-mumps' _solve_dense already resets ICNTL(20), so no manual reset is
    needed between sparse and dense solves.
    """
    if not transpose:
        return ctx._solve_sparse(b)

    inst = ctx.mumps_instance
    prev = inst.icntl[_ICNTL_TRANSPOSE]
    inst.icntl[_ICNTL_TRANSPOSE] = 0
    try:
        return ctx._solve_sparse(b)
    finally:
        inst.icntl[_ICNTL_TRANSPOSE] = prev


def setup_solver_Aii_local(ctx, N, dtype, ctxT=None):
    """LinearOperator applying A^{-1} and A^{-T}.

    ctxT is None (default): A^{-T} reuses the A factors via ICNTL(9)=0.
    ctxT given:             A^{-T} is a forward solve against factored A^T.

    NOTE: the argument order changed from the previous
    (ctx, ctxT, N, dtype) -- ctxT is now an optional trailing argument.
    """
    def _fwd(x):
        return ctx.solve(x)

    def _adj(x):
        if ctxT is not None:
            return ctxT.solve(x)
        return mumps_solve(ctx, x, transpose=True)

    return LinearOperator(
        shape=(N, N),
        dtype=dtype,
        matvec=_fwd,
        rmatvec=_adj,
        matmat=_fwd,
        rmatmat=_adj,
    )


def check_adjoint_consistency(op, k=4, seed=0, verbose=True, name="A^-1"):
    """Verify <op x, y> == <x, op^T y> for the solve operator.

    Worth running once per new MUMPS build / problem class.  Randomized
    compression samples the corange through op^T, so if the transposed solve is
    silently inexact this catches it before it pollutes the compression.  A
    clean LU gives ~1e-14 relative; it is independent of any compression rank.
    """
    rng = np.random.default_rng(seed)
    n = op.shape[0]
    X = rng.standard_normal((n, k))
    Y = rng.standard_normal((n, k))
    lhs = np.einsum("ij,ij->j", op @ X, Y)
    rhs = np.einsum("ij,ij->j", X, op.H @ Y if np.iscomplexobj(X) else op.T @ Y)
    rel = np.abs(lhs - rhs) / np.maximum(np.abs(lhs), np.finfo(float).tiny)
    if verbose:
        print("adjoint consistency %s: max rel. gap = %.3e" % (name, rel.max()))
    return rel.max()


"""
    This header takes care of the Solver Wrapper class
    Recipe:
    - user has some external solver (e.g. 'mySolver') in folder 'mySolverFolder'
    - places mySolverFolder in folder 'solver'
    - add 'from solver.mySolverFolder.mySolver import mySolver' (or variant thereof)
    - add to class solverOptions: 'type==mySolver' and then set order//nyz//...
    - add geometry conversion if needed to 'convertGeom'
    - add class init ( if self.type=='mySolver'...self.solver=mySolver(...) )to solverWrapper
    REQUIREMENTS FOR SOLVER:
    Solver must inherit from AbstractPDESolver or be compatible with it
"""

class stMap:
    def __init__(self,A:LinearOperator,XXI,XXJ,m_large = 0,n_large=0):
        self.XXI = XXI
        self.XXJ = XXJ
        self.A = A
        self.m_large = m_large
        self.n_large = n_large


# =========================================================================== #
#  Interpolation / reconstruction state
#
#  What a local solver must keep, beyond its factorization, so that a
#  boundary trace can be turned into a volumetric field and that field be
#  interpolated -- after the solver itself has been slimmed down or
#  off-loaded by oms.  Two layers:
#
#    HPSInterpGrid     : evaluation only (grid + panel layout).  Light; what an
#                        OMSSolution keeps per slab.
#    HPSaltInterpState : + leaf reconstruction (skeleton -> full grid).  Holds
#                        the leaf discretization (HPS_Multidomain), not the
#                        sparse matrix and not the factorization.
#
#  The two hooks the OMS layer relies on are
#      state.full_solution(u_i, u_b, ...)  -> values on grid._XXfull
#      state.grid.interp(pts, f)           -> values at pts
#  Supporting another discretization means providing an object with these.
# =========================================================================== #

def _to_numpy(x):
    """torch tensor (any device) or array-like -> numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class HPSInterpGrid:
    """
    Duck-types the attributes HPSInterp.interp reads from a solver:
    ndim, npan_dim, p, geom (with .box_geom), _XXfull.
    """
    def __init__(self, typestr, ndim, npan_dim, p, geom, XXfull):
        self.typestr  = typestr
        self.ndim     = int(ndim)
        self.npan_dim = np.asarray(_to_numpy(npan_dim)).astype(np.int64)
        self.p        = np.asarray(_to_numpy(p)).astype(np.int64)
        self.geom     = geom
        self._XXfull  = np.asarray(_to_numpy(XXfull), dtype=np.float64)

    @property
    def XXfull(self):
        return self._XXfull

    def interp(self, pts, f):
        """Values f on _XXfull -> values at pts (points inside this grid)."""
        return interp.interp(self, np.asarray(pts, dtype=np.float64),
                             np.asarray(f), self.typestr)


class HPSaltInterpState:
    """
    Reconstruction + interpolation state of one hpsalt (Domain_Driver) slab.

    Coded against the statically condensed DtN path of hpslib/hpsmultidomain:
    the skeleton solution (on XX = xx_active, split into Ji / Jx) is lifted to
    the leaf interiors by HPS_Multidomain.solve, i.e. batched local leaf
    solves.  That step does NOT touch the slab-level sparse factorization.
    """
    typestr = "hpsalt"

    def __init__(self, driver, sparse_assembly):
        if getattr(driver, "use_iti_maps", False):
            raise NotImplementedError(
                "interpolation: ItI leaf maps are not supported")
        if not getattr(driver, "statically_condense", True):
            raise NotImplementedError(
                "interpolation: statically_condense=False is not supported")
        hps = driver.hps
        self.hps      = hps
        self.n_active = int(len(hps.I_unique))
        self.Ii       = np.asarray(_to_numpy(driver._Ji)).astype(np.int64)
        self.Ib       = np.asarray(_to_numpy(driver._Jx)).astype(np.int64)
        self.device   = "cuda" if sparse_assembly == "reduced_gpu" else "cpu"
        self.grid     = HPSInterpGrid(self.typestr, hps.d, hps.n, hps.p,
                                      driver.geom, driver._XXfull)

    def interp(self, pts, f):
        return self.grid.interp(pts, f)

    def full_solution(self, u_i, u_b, body_load=None, offset=None):
        """
        Skeleton values (u_i on Ii, u_b on Ib) -> values on grid._XXfull.

        body_load : optional callable f(xx) (torch, global coordinates), the
                    body load the local problem was solved with.
        offset    : translation from this solver's own coordinates to the
                    global ones (stiff_mat_const); body_load is evaluated at
                    the shifted points.
        """
        import torch

        u_i = np.ascontiguousarray(np.asarray(u_i).reshape(-1))
        u_b = np.ascontiguousarray(np.asarray(u_b).reshape(-1))
        if len(u_i) != len(self.Ii) or len(u_b) != len(self.Ib):
            raise ValueError(
                "full_solution: got %d interior / %d boundary values, expected "
                "%d / %d" % (len(u_i), len(u_b), len(self.Ii), len(self.Ib)))
        dt = np.result_type(u_i.dtype, u_b.dtype, np.float64)
        uu = torch.zeros((self.n_active, 1),
                         dtype=torch.from_numpy(np.zeros(0, dtype=dt)).dtype)
        uu[torch.from_numpy(self.Ii), 0] = torch.from_numpy(u_i.astype(dt))
        uu[torch.from_numpy(self.Ib), 0] = torch.from_numpy(u_b.astype(dt))

        ff = body_load
        if body_load is not None and offset is not None and np.any(offset):
            shift = torch.as_tensor(np.asarray(offset, dtype=np.float64))

            def ff(xx, _f=body_load, _s=shift):
                return _f(xx + _s.to(device=xx.device, dtype=xx.dtype))

        sol, _ = self.hps.solve(torch.device(self.device), uu,
                                ff_body_func=ff)
        return _to_numpy(sol[:, 0])


# =========================================================================== #
#  Body-load reduction
#
#  A local slab problem with body load f reads, after the discretization has
#  eliminated whatever it eliminates,
#
#      Aii u_i + Aib u_b = f_i        (rows Ii)
#      Abi u_i + Abb u_b = f_b        (rows Ib; used by flux / Neumann rows)
#
#  (f_i, f_b) is the REDUCED load: in the row convention of Aii / Aib, and in
#  general not the body load sampled at points.  How it is obtained depends on
#  the discretization (static condensation for HPS; plain sampling for a
#  collocation scheme that keeps every point), so it lives here, next to the
#  discretization, behind one interface:
#
#      reducer.reduce(body_load, offset=None) -> (f_i on Ii, f_b on Ib)
#
#  body_load : callable f(xx), torch points in GLOBAL coordinates, f of the
#              same PDO the slab was built with (L u = f).
#  offset    : translation from the solver's own coordinates to the global
#              ones (stiff_mat_const), as in full_solution.
#  Returns host numpy arrays; real when the load is real.
#
#  Supporting another discretization means one BodyLoadReducer subclass and
#  one dispatch line in solverWrapper.body_reducer.
# =========================================================================== #

def _shifted(body_load, offset):
    """body_load evaluated at xx + offset (torch points); body_load if no shift."""
    if offset is None or not np.any(offset):
        return body_load
    import torch
    shift = torch.as_tensor(np.asarray(offset, dtype=np.float64))

    def ff(xx, _f=body_load, _s=shift):
        return _f(xx + _s.to(device=xx.device, dtype=xx.dtype))
    return ff


class BodyLoadReducer:
    """Interface: reduce(body_load, offset=None) -> (f_i on Ii, f_b on Ib)."""
    typestr = None

    def reduce(self, body_load, offset=None):
        raise NotImplementedError


class HPSaltBodyReducer(BodyLoadReducer):
    """
    Static condensation of a body load for one hpsalt (Domain_Driver) slab.

    Uses HPS_Multidomain.reduce_body -- batched leaf solves, duplicated face
    copies summed -- which returns the condensed load on the active
    (I_unique) ordering, the ordering XX / Ji / Jx index into.  Same
    convention as Domain_Driver.get_rhs:
        A_CC u_i = -A_CX u_b + reduce_body(...)[Ji].
    Holds only the leaf discretization (shared with HPSaltInterpState), not
    the sparse matrix and not the factorization.
    """
    typestr = "hpsalt"

    def __init__(self, driver, sparse_assembly):
        if getattr(driver, "use_iti_maps", False):
            raise NotImplementedError(
                "body load reduction: ItI leaf maps are not supported")
        if not getattr(driver, "statically_condense", True):
            raise NotImplementedError(
                "body load reduction: statically_condense=False is not "
                "supported (the uncondensed driver takes no body load)")
        hps = driver.hps
        self.hps      = hps
        self.n_active = int(len(hps.I_unique))
        self.Ii       = np.asarray(_to_numpy(driver._Ji)).astype(np.int64)
        self.Ib       = np.asarray(_to_numpy(driver._Jx)).astype(np.int64)
        self.device   = "cuda" if sparse_assembly == "reduced_gpu" else "cpu"

    def reduce(self, body_load, offset=None):
        import torch
        if not callable(body_load):
            raise TypeError(
                "body_load must be a callable f(xx) on torch points in global "
                "coordinates; vector body loads are not supported yet")
        red = self.hps.reduce_body(torch.device(self.device),
                                   _shifted(body_load, offset), None)
        red = _to_numpy(red).reshape(-1)
        if red.shape[0] != self.n_active:
            raise ValueError(
                "reduce_body returned %d values, expected %d (one per active "
                "dof); is body_load returning one value per point?"
                % (red.shape[0], self.n_active))
        # reduce_body allocates a complex buffer for function loads; keep a
        # real problem real
        if np.iscomplexobj(red) and not np.any(red.imag):
            red = np.ascontiguousarray(red.real)
        return red[self.Ii], red[self.Ib]


class solverOptions:
    """
    Class that encodes the options for a local slab Solver
    @param:
    type:       type of discretization (HPS/cheb/stencil/HPSalt)
    ordx,ordy:  order in x and y directions
    a:          characteristic scale in case of HPS
    problem_type: 'Dirichlet' or 'mixed'
                    for mixed, the assumption (for now) is  that we have Dirichlet on vertical bdry sections, Neumann on rest
    MUMPS options below apply to the local factorization of both problem
    types (Dirichlet: Aii; mixed: M) for type 'hpsalt'.
    mumps_ordering: analysis ordering ('metis', 'auto', 'amd', 'scotch', ...)
    blr_tol:    if > 0, BLR-compressed factorization with this tolerance
    use_ctxT:   factor A^T into a second context instead of reusing the A
                factors via ICNTL(9)=0.  ~2x memory and factor time; only for
                cross-checking the transposed-solve path.
    mumps_block_size: ICNTL(27) blocking size for multiple right-hand sides
    """
    def __init__(self,type:str,ord,a=None,problem_type='Dirichlet',
                 mumps_ordering='metis',blr_tol=0.0,use_ctxT=False,
                 mumps_block_size=None,reduced_gpu=False):
        self.type   =   type
        self.ord    =   ord
        self.a      =   a
        self.problem_type = problem_type
        self.mumps_ordering   = mumps_ordering
        self.blr_tol          = blr_tol
        self.use_ctxT         = use_ctxT
        self.mumps_block_size = mumps_block_size
        self.reduced_gpu        = reduced_gpu

def convertGeom(opts,geom):
    if opts.type=='hpsalt':
        return hpsaltGeom.BoxGeometry(np.array(geom))
    if opts.type=='hps':
        from solver.spectralmultidomain.hps import geom as hpsGeom
        import jax.numpy as jnp
        return hpsGeom.BoxGeometry(jnp.array(geom))
    if opts.type=='stencil':
        return stencilGeom.BoxGeometry(np.array(geom))
    if opts.type=='spectral':
        return spectralGeom.BoxGeometry(np.array(geom))
    # previously fell through returning None, which surfaced far downstream as
    # an UnboundLocalError on `solver`
    raise ValueError("unknown solver type %r" % (opts.type,))


class solverWrapper:
    """
    Wrapper class for local Solver
    @param:
    opts:       slab options
    """
    def __init__(self,opts:solverOptions):
        self.ord   = opts.ord
        self.type   = opts.type
        self.a      = opts.a
        self.constructed = False
        self.opts=opts
        # MUMPS handles / timings, populated by construct() on the mixed path
        self.ctx  = None
        self.ctxT = None
        self.time_analysis = 0.0
        self.time_factor   = 0.0

    def construct(self,geom,PDE,verbose=False,compute_inverse=True):
        """
        Actual construction of the local solver
        """
        self.ndim = geom.shape[1]
        if self.type=='stencil':
            geomStencil = convertGeom(self.opts,geom)
            solver = stencil(PDE, geomStencil, self.ord)
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = solver.XX
            self.Ii = solver._Ji
            self.Ib = solver._Jx
            
            self.Aib = solver.Aix
            self.Abi = solver.Axi
            self.Abb = solver.Axx
            self.solver_ii = solver.solver_Aii
        elif self.type=='hps':
            from solver.spectralmultidomain.hps import hps_multidomain as hps
            geomHPS = convertGeom(self.opts,geom)
            solver = hps.HPSMultidomain(PDE, geomHPS,self.a, self.ord[0],verbose=verbose)
            self.solver=solver
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = solver.XX
            self.XXfull = solver._XXfull
            self.Ii = solver._Ji
            self.Ib = solver._Jx
            self.Aib = solver.Aix
            self.Abi = solver.Axi
            self.Abb = solver.Axx
            self.Aii = solver.Aii
            tic      = time()
            
            self.solver_ii = solver.solver_Aii
            toc      = time() - tic
            print("\t Toc construct Aii inverse %5.2f s" % toc) if verbose else None
        elif self.type=='hpsalt':
            geomHPS = convertGeom(self.opts,geom)
            solver = hpsalt.Domain_Driver(geomHPS, PDE, 0, self.a, p=self.ord, d=len(self.ord)) #verbose=verbose)
            self.solver=solver
            if self.opts.reduced_gpu:
                self.solver.build("reduced_gpu", "MUMPS", verbose=verbose)
            else:
                self.solver.build("reduced_cpu", "MUMPS", verbose=verbose)
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = solver.XX
            self.XXfull = solver._XXfull
            self.Ii = solver._Ji
            self.Ib = solver._Jx
            self.Aib = solver.Aix
            self.Abi = solver.Axi
            self.Abb = solver.Axx
            self.Aii = solver.Aii
            if compute_inverse:
                if self.opts.problem_type == 'Dirichlet':
                    # OPT: factor Aii through the same tuned MUMPS layer as
                    #      the mixed path (ordering, BLR, reuse_analysis,
                    #      block size, A^{-T} on the same factors, blocked
                    #      matmat/rmatmat).  It used to go through the driver's
                    #      SparseSolver, which ignored all of those options.
                    tic      = time()
                    self.solver_ii = self._factor_mumps(self.Aii, verbose)
                    # hand the same operator to the driver, so its own solve
                    # paths never trigger a second (lazy) factorization
                    solver.setup_solver_Aii(solve_op=self.solver_ii)
                    toc      = time() - tic
                    print("\t Toc construct Aii inverse %5.2f s "
                          "(analysis %5.2f s, factor %5.2f s)"
                          % (toc, self.time_analysis, self.time_factor)) if verbose else None
                elif self.opts.problem_type == 'mixed':
                    tic      = time()
                    # scale the face-detection tolerance with the geometry;
                    # a bare 1e-10 silently returns an empty JD on a domain
                    # whose coordinates are not O(1)
                    bounds = geomHPS.bounds
                    xlo, xhi = bounds[0][0], bounds[1][0]
                    tol = 1e-10 * max(1.0, abs(xlo), abs(xhi), abs(xhi - xlo))
                    Xb = self.XX[self.Ib, 0]
                    JD = np.flatnonzero((np.abs(Xb - xlo) < tol)
                                        | (np.abs(Xb - xhi) < tol))
                    mask = np.ones(len(self.Ib), dtype=bool)
                    mask[JD] = False
                    JN = np.flatnonzero(mask).astype(np.int64)

                    M = sp.block_array([[self.Aii,self.Aib[:,JN]],[self.Abi[JN,:],self.Abb[JN,:][:,JN]]]).tocsc()
                    E = sp.vstack([self.Aib[:,JD],self.Abb[JN,:][:,JD]]).tocsr()
                    self.M = M
                    self.E = E
                    self.JD = JD
                    self.JN = JN

                    # one factorization; A^{-T} reuses it
                    self.solver_ii = self._factor_mumps(M, verbose)
                    toc      = time() - tic
                    print("\t Toc construct Aii inverse %5.2f s "
                          "(analysis %5.2f s, factor %5.2f s)"
                          % (toc, self.time_analysis, self.time_factor)) if verbose else None
                else:
                    raise ValueError(
                        "problem_type must be 'Dirichlet' or 'mixed', got %r"
                        % (self.opts.problem_type,))

        elif self.type=='spectral':
            geomSpectral = convertGeom(self.opts,geom)
            solver = spectral(PDE, geomSpectral, self.ord)
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = solver.XX
            self.Ii = solver._Ji
            self.Ib = solver._Jx
            
            self.Aib = solver.Aix
            self.Abi = solver.Axi
            self.Abb = solver.Axx
            self.solver_ii = solver.solver_Aii
        else:
            raise ValueError("unknown solver type %r" % (self.type,))
        
        self.XXi = solver.XX[self.Ii,:]
        self.XXb = solver.XX[self.Ib,:]
        self.ndofs = solver.XX.shape[0]

    def _factor_mumps(self, A, verbose=False):
        """
        Factor the local system matrix A with the tuned MUMPS layer and
        return the LinearOperator applying A^{-1} (matvec / matmat) and
        A^{-T} (rmatvec / rmatmat, on the same factors via ICNTL(9) unless
        opts.use_ctxT).  Honours opts.mumps_ordering, blr_tol,
        mumps_block_size.  Sets self.ctx / ctxT / time_analysis / time_factor.
        Used by both the Dirichlet (A = Aii) and the mixed (A = M) path.
        """
        A = A.tocsc() if sp.issparse(A) else A
        ctx, t_an, t_fa = setup_mumps(
            A,
            ordering=self.opts.mumps_ordering,
            blr_tol=self.opts.blr_tol,
            block_size=self.opts.mumps_block_size,
            verbose=2 if verbose else 0,
        )
        self.ctx = ctx
        self.time_analysis = t_an
        self.time_factor   = t_fa

        if self.opts.use_ctxT:
            ctxT, t_anT, t_faT = setup_mumps_transpose(
                A,
                ordering=self.opts.mumps_ordering,
                blr_tol=self.opts.blr_tol,
                block_size=self.opts.mumps_block_size,
                verbose=2 if verbose else 0,
            )
            self.ctxT = ctxT
            self.time_analysis += t_anT
            self.time_factor   += t_faT
            print("\t A^-T applies: dedicated A^T factorization "
                  "(use_ctxT)") if verbose else None
        else:
            self.ctxT = None
            print("\t A^-T applies: reusing A factorization with "
                  "ICNTL(9)=0") if verbose else None

        return setup_solver_Aii_local(ctx, A.shape[0], A.dtype, ctxT=self.ctxT)

    def check_adjoint(self, k=4, seed=0, verbose=True):
        """Adjoint-consistency check on this slab's solve operator."""
        return check_adjoint_consistency(self.solver_ii, k=k, seed=seed,
                                         verbose=verbose, name="Aii^-1")

    @property
    def interp_state(self):
        """
        Reconstruction + interpolation state (see HPSaltInterpState), built
        on first access and cached.  None if this solver type does not support
        it yet; the reason is then in `interp_unavailable`.  Never raises, so
        it is safe to probe with getattr / hasattr.
        """
        st = self.__dict__.get("_interp_state")
        if st is not None or not self.constructed:
            return st
        if self.type == "hpsalt":
            try:
                st = HPSaltInterpState(
                    self.solver,
                    "reduced_gpu" if self.opts.reduced_gpu else "reduced_cpu")
            except NotImplementedError as exc:
                self.interp_unavailable = str(exc)
                return None
            self._interp_state = st
            return st
        self.interp_unavailable = (
            "volumetric reconstruction is not implemented for solver type %r"
            % (self.type,))
        return None

    @property
    def body_reducer(self):
        """
        Body-load reducer (see BodyLoadReducer), built on first access and
        cached.  None if this solver type does not support it yet; the reason
        is then in `body_reduce_unavailable`.  Never raises, so it is safe to
        probe with getattr / hasattr (oms keeps it on slimmed solvers).
        """
        red = self.__dict__.get("_body_reducer")
        if red is not None or not self.constructed:
            return red
        if self.type == "hpsalt":
            try:
                red = HPSaltBodyReducer(
                    self.solver,
                    "reduced_gpu" if self.opts.reduced_gpu else "reduced_cpu")
            except NotImplementedError as exc:
                self.body_reduce_unavailable = str(exc)
                return None
            self._body_reducer = red
            return red
        self.body_reduce_unavailable = (
            "body load reduction is not implemented for solver type %r"
            % (self.type,))
        return None

    def reduce_body_load(self, body_load, offset=None):
        """
        Reduced load (f_i on Ii, f_b on Ib) of `body_load` for this slab, in
        the row convention of Aii / Aib:  Aii u_i + Aib u_b = f_i.

        body_load : callable f(xx), torch points in global coordinates.
        offset    : solver -> global translation (stiff_mat_const), or None.
        """
        red = self.body_reducer
        if red is None:
            raise NotImplementedError(
                getattr(self, "body_reduce_unavailable", None)
                or "body load reduction is not available for this solver")
        return red.reduce(body_load, offset)

    def check_body_reduction(self, v, Lv, offset=None, verbose=True):
        """
        Self-test of reduce_body_load on this slab.  For a smooth v and its
        image under the PDO, Lv = L v (both callables on torch points in global
        coordinates), static condensation gives exactly

            f_i = Aii v_i + Aib v_b      on Ii

        up to the spectral differentiation error of v.  Returns the relative
        error.  A sign or scaling mismatch shows up as O(1).
        """
        import torch
        f_i, _ = self.reduce_body_load(Lv, offset)
        XX = self.XX if hasattr(self.XX, "detach") else torch.as_tensor(np.asarray(self.XX))
        if offset is not None and np.any(offset):
            XX = XX + torch.as_tensor(np.asarray(offset, dtype=np.float64)).to(
                device=XX.device, dtype=XX.dtype)
        vv = _to_numpy(v(XX)).reshape(-1)
        Ii = np.asarray(_to_numpy(self.Ii)).astype(np.int64)
        Ib = np.asarray(_to_numpy(self.Ib)).astype(np.int64)
        ref = np.asarray(self.Aii @ vv[Ii]).reshape(-1) \
            + np.asarray(self.Aib @ vv[Ib]).reshape(-1)
        nrm = np.linalg.norm(ref)
        err = np.linalg.norm(f_i - ref) / (nrm if nrm > 0 else 1.0)
        if verbose:
            print("check_body_reduction: ||f_i - (Aii v_i + Aib v_b)|| / ||.|| "
                  "= %.3e" % err)
        return err

    #given values f on the full solver grid, interpolate f to the points x
    def interp(self,pts,f):
        if self.type=='hps':
            return interp.interp(self.solver,pts,f,'hps')
        elif self.type == 'hpsalt':
            return interp.interp(self.solver,pts,f,'hpsalt')
        else:
            raise ValueError("interp not implemented yet")