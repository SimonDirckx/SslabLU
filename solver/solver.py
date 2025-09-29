
import pdo.pdo as pdo
import numpy as np
from scipy.sparse.linalg   import LinearOperator
from solver.stencil.stencilSolver import stencilSolver as stencil
from solver.spectral.spectralSolver import spectralSolver as spectral
from solver.spectralmultidomain.hps import hps_multidomain as hps
import solver.spectralmultidomain.hps.geom as hpsGeom
import solver.stencil.geom as stencilGeom
import solver.spectral.geom as spectralGeom
import jax.numpy as jnp
import solver.HPSInterp as interp

# Things we need to add:
#from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
#import solver.hpsmultidomain.hpsmultidomain.geom as hpsaltGeom


from time import time
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
    def __init__(self,A:LinearOperator,XXI,XXJ):
        self.XXI = XXI
        self.XXJ = XXJ
        self.A = A


class solverOptions:
    """
    Class that encodes the options for a local slab Solver
    @param:
    type:       type of discretization (HPS/cheb/stencil/HPSalt)
    ordx,ordy:  order in x and y directions
    a:          characteristic scale in case of HPS
    """
    def __init__(self,type:str,ord,a=None):
        self.type   =   type
        self.ord    =   ord
        self.a      =   a

def convertGeom(opts,geom):
    #if opts.type=='hpsalt':
    #    return hpsaltGeom.BoxGeometry(np.array(geom))
    if opts.type=='hps':
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

    def construct(self,geom,PDE:pdo,verbose=False):
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
            print("start solver")
            self.solver_ii = solver.solver_Aii
            print("solver done")
            toc      = time() - tic
            print("\t Toc construct Aii inverse %5.2f s" % toc) if verbose else None
        if self.type=='hpsalt':
            geomHPS = convertGeom(self.opts,geom)
            #solver = hpsalt.Domain_Driver(geomHPS, PDE, 0, self.a, p=self.ord[0], d=len(self.ord)) #verbose=verbose)
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
            tic      = time()
            print("start solver")
            solver.setup_solver_Aii()
            self.solver_ii = solver.solver_Aii
            print("solver done")
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
            return interp.interp(self.solver,pts,f)
        else:
            raise ValueError("interp not implemented yet")

