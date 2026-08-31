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


class solverOptions:
    """
    Class that encodes the options for a local slab Solver
    @param:
    type:       type of discretization (HPS/cheb/stencil/HPSalt)
    ordx,ordy:  order in x and y directions
    a:          characteristic scale in case of HPS
    problem_type: 'Dirichlet' or 'mixed'
                    for mixed, the assumption (for now) is  that we have Dirichlet on vertical bdry sections, Neumann on rest
    mumps_ordering: analysis ordering ('metis', 'auto', 'amd', 'scotch', ...)
    blr_tol:    if > 0, BLR-compressed factorization with this tolerance
    use_ctxT:   factor A^T into a second context instead of reusing the A
                factors via ICNTL(9)=0.  ~2x memory and factor time; only for
                cross-checking the transposed-solve path.
    mumps_block_size: ICNTL(27) blocking size for multiple right-hand sides
    """
    def __init__(self,type:str,ord,a=None,problem_type='Dirichlet',
                 mumps_ordering='metis',blr_tol=0.0,use_ctxT=False,
                 mumps_block_size=None):
        self.type   =   type
        self.ord    =   ord
        self.a      =   a
        self.problem_type = problem_type
        self.mumps_ordering   = mumps_ordering
        self.blr_tol          = blr_tol
        self.use_ctxT         = use_ctxT
        self.mumps_block_size = mumps_block_size

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

    def construct(self,geom,PDE,verbose=False,compute_inverse=True,reduced_gpu = False):
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
            if reduced_gpu:
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
                    tic      = time()
                    solver.setup_solver_Aii()
                    self.solver_ii = solver.solver_Aii
                    toc      = time() - tic
                    print("\t Toc construct Aii inverse %5.2f s" % toc) if verbose else None
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

                    # ---- one factorization; A^{-T} reuses it -------------- #
                    ctx, t_an, t_fa = setup_mumps(
                        M,
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
                            M,
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

                    self.solver_ii = setup_solver_Aii_local(
                        ctx, M.shape[0], M.dtype, ctxT=self.ctxT)
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

    def check_adjoint(self, k=4, seed=0, verbose=True):
        """Adjoint-consistency check on this slab's solve operator."""
        return check_adjoint_consistency(self.solver_ii, k=k, seed=seed,
                                         verbose=verbose, name="Aii^-1")

    #given values f on the full solver grid, interpolate f to the points x
    def interp(self,pts,f):
        if self.type=='hps':
            return interp.interp(self.solver,pts,f,'hps')
        elif self.type == 'hpsalt':
            return interp.interp(self.solver,pts,f,'hpsalt')
        else:
            raise ValueError("interp not implemented yet")