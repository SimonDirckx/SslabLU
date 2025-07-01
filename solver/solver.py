import geometry.slabGeometry as slabGeom
import pdo.pdo as pdo
import numpy as np
from scipy.sparse.linalg   import LinearOperator
from solver.stencil.stencilSolver import stencilSolver as stencil
from solver.spectral.spectralSolver import spectralSolver as spectral
from solver.spectralmultidomain.hps import hps_multidomain as hps
import solver.spectralmultidomain.hps.geom as hpsGeom
import jax.numpy as jnp
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
    type:       type of discretization (HPS/cheb/stencil)
    ordx,ordy:  order in x and y directions
    a:          characteristic scale in case of HPS
    """
    def __init__(self,type:str,ord,a=None):
        self.type   =   type
        self.ord    =   ord
        self.a      =   a

def convertGeom(opts,geom):
    if opts.type=='hps':
        return hpsGeom.BoxGeometry(jnp.array(geom))


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
    def __del__(self):
        print("solverWrap deleted")

    def construct(self,geom,PDE:pdo):
        """
        Actual construction of the local solver
        """
        if self.type=='stencil':
            self.solver = stencil(PDE, geom, self.ord)
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = self.solver.XX
            self.Ii = self.solver._Ji
            self.Ib = self.solver._Jb
            
            self.Aib = self.solver.Aib
            self.Abi = self.solver.Abi
            self.Abb = self.solver.Abb
            self.solver_ii = self.solver.solver_ii
        if self.type=='hps':
            geomHPS = convertGeom(self.opts,geom)
            solver = hps.HPSMultidomain(PDE, geomHPS,self.a, self.ord[0])
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
        if self.type=='spectral':
            self.solver = spectral(PDE, geom, self.ord)
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = self.solver.XX
            self.Ii = self.solver._Ji
            self.Ib = self.solver._Jb
            
            self.Aib = self.solver.Aib
            self.Abi = self.solver.Abi
            self.Abb = self.solver.Abb
            self.solver_ii = self.solver.solver_ii
        
        self.XXi = solver.XX[self.Ii,:]
        self.XXb = solver.XX[self.Ib,:]
        self.ndofs = solver.XX.shape[0]
        #del solver
        #self.constructMapIdxs()