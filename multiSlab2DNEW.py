import numpy as np
import scipy as sc
import spectralmultidomain.hps as hps
import spectralmultidomain.hps.hps_multidomain as hpsMult
import spectralmultidomain.hps.pdo as pdo
from   spectralmultidomain.hps.pde_solver import AbstractPDESolver
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import spectralDisc as spectral
import stencilDisc as stencil
from geometry.slabGeometry import slabGeometry
import solver.solver as slabSolver
import matAssembly.matAssembler as mA







class Slab:

    """
    Class representing slab geometry

    @param:
    geom        : geometry (l2g maps,bounds,...)
    PDE         : PDE to be solved
    solver      : the solver to be used (HPS/Spectral/Stencil/...)
    lrmaps      : left-right maps (encode type of slab decomp)
    matMethod   : method for the construction of the system blocks (dense/HBS/...)    
    """

    
    def __init__(   self,
                    geom:slabGeometry,
                    PDE:pdo.PDO2d,
                    solver:slabSolver.localSolver,
                    lrmaps:slabSolver.localMapper,
                    matAssembler:mA.matAssembler
                ):
        
        self.solver         =   solver
        self.solver.construct(geom,PDE)
        self.XX             =   solver.XX
        self.ndofs          =   self.XX.shape[0]
        self.lrmaps         =   lrmaps
        self.N              =   solver.ndofs
        self.geom           =   geom
        self.matAssembler   =   matAssembler
        
    #########################
    #   geometric methods   #
    #########################
    
    ######################################################################

    def l2g(self,i):
        """maps the ith local dof to the corresponding global coordinates"""
        x,y=self.XX[i,:]
        return self.geom.l2g(x,y)
    
    def eval_global_func(self,f,i):
        """helper function to evaluate globally defined f @ dof i"""
        x,y=self.l2g(i)
        return f(x,y)
    
    def getGlobalXX(self):
        """return global coordinates of vectorized dofs"""
        return np.array([self.l2g(i) for i in range(self.ndofs)])
    
    def setAsEdge(self,edgeType:int):
        """set slab as left and/or right edge in global geometry"""
        if edgeType==1:
            self.leftEdge   = True
        if edgeType == 2:
            self.rightEdge  = True
        self.edge = self.leftEdge or self.rightEdge
    
    
    ######################################################################
    

    
    #########################
    #   Block constructors  #
    #########################
    # note: what about the 'central map' (e.g. T11)? Should this be local or global?
    ######################################################################

    def computeBlock_l(self):
        self.BL = self.matAssembler(self.lrmaps.leftMap(self.solver))
        self.BLComputed = True
        
    
    def computeBlock_r(self):
        self.BR = self.matAssembler(self.lrmaps.rightMap(self.solver))
        self.BRComputed = True
    
    def getLeft(self):
        if self.BLComputed:
            return self.BL
        else:
            self.computeBlock_l()
            return self.BL
    
    def getRight(self):
        if self.BRComputed:
            return self.BR
        else:
            self.computeBlock_r()
            return self.BR


            


         

