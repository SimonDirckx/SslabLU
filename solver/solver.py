
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


def setup_mumps(A):
    ctx = mumps.Context()
    ctx.analyze(A)
    ctx.analyze(A)
    ctx.factor(A)
    return ctx


def setup_mumps_transpose(A):
    ctxT = mumps.Context()
    ctxT.analyze(A.T)
    ctxT.factor(A.T)
    return ctxT


def setup_solver_Aii_local(ctx,ctxT,N,dtype):
    
        return LinearOperator(
            shape=(N, N),
            dtype=dtype,
            matvec=lambda x: ctx.solve(x),
            rmatvec=lambda x: ctxT.solve(x),
            matmat=lambda X: ctx.solve(X),
            rmatmat=lambda X: ctxT.solve(X),
        )

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
    """
    def __init__(self,type:str,ord,a=None,problem_type='Dirichlet'):
        self.type   =   type
        self.ord    =   ord
        self.a      =   a
        self.problem_type = problem_type

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
        self.type = opts.type
        self.constructed = False
        self.opts=opts

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
        if self.type=='hps':
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
        if self.type=='hpsalt':
            geomHPS = convertGeom(self.opts,geom)
            solver = hpsalt.Domain_Driver(geomHPS, PDE, 0, self.a, p=self.ord, d=len(self.ord)) #verbose=verbose)
            self.solver=solver
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
                    tol = 1e-10
                    JD = np.where((np.abs(self.XX[self.Ib,0]-geomHPS.bounds[0][0])<tol) | (np.abs(self.XX[self.Ib,0]-geomHPS.bounds[1][0])<tol))[0]
                    JN = np.array( [i for i in range(len(self.Ib)) if not i in JD] , dtype = np.int64)

                    M = sp.block_array([[self.Aii,self.Aib[:,JN]],[self.Abi[JN,:],self.Abb[JN,:][:,JN]]]).tocsc()
                    E = sp.vstack([self.Aib[:,JD],self.Abb[JN,:][:,JD]]).tocsr()
                    self.M = M
                    self.E = E
                    self.JD = JD
                    self.JN = JN
                    ctx = setup_mumps(M)
                    ctxT = setup_mumps_transpose(M)
                    self.solver_ii = setup_solver_Aii_local(ctx,ctxT,M.shape[0],M.dtype)
                    toc      = time() - tic
                    print("\t Toc construct Aii inverse %5.2f s" % toc) if verbose else None

        if self.type=='spectral':
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
        
        self.XXi = solver.XX[self.Ii,:]
        self.XXb = solver.XX[self.Ib,:]
        self.ndofs = solver.XX.shape[0]
    
    #given values f on the full solver grid, interpolate f to the points x
    def interp(self,pts,f):
        if self.type=='hps':
            return interp.interp(self.solver,pts,f,'hps')
        elif self.type == 'hpsalt':
            return interp.interp(self.solver,pts,f,'hpsalt')
        else:
            raise ValueError("interp not implemented yet")
