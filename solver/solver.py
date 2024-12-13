import geometry.slabGeometry as slabGeom
import pdo.pdo as pdo
import numpy as np
from scipy.sparse.linalg   import LinearOperator
from solver.stencil.stencilSolver import stencilSolver as stencil


class solverOptions:
    """
    Class that encodes the options for a local slab Solver
    @param:
    type:       type of discretization (HPS/cheb/stencil)
    ordx,ordy:  order in x and y directions
    a:          characteristic scale in case of HPS
    """
    def __init__(self,type:str,ord,a=1):
        self.type   =   type
        self.ord    =   ord
        self.a      =   a
        self.nyz    =   1
        if type=='stencil':
            for i in range(1,len(ord)):
                self.nyz *= (ord[i]-2)


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
        self.solver = None
        self.type = opts.type
        self.constructed = False
    def construct(self,geom:slabGeom.slabGeometry,PDE:pdo):
        """
        Actual construction of the local solver
        """
        self.geom   = geom
        self.PDE    = PDE
        if self.type=='stencil':
            self.solver = stencil(PDE, geom, self.ord)
            self.constructed=True
        
        self.XX = self.solver.XX
        self.XXi = self.solver.XXi
        self.XXb = self.solver.XXb
        self.ndofs = self.XX.shape[0]
        self.constructMapIdxs()
    
    def constructMapIdxs(self):
        Il,Ic,Ir,IGB=self.geom.getIlIcIr(self.XXi,self.XXb)
        self.leftIdxs=Il
        self.rightIdxs=Ir
        self.middleIdxs=Ic
        self.IGB=IGB
            

    def stMap(self,J:list[int],I:list[int],stType):
        #for now:   in case of DtD, I should be subset indices in Ii, J should be subset indices in Ib
        #           in case of DtN, both should be subset indices in Ib
        if stType=='DtD':
            if not I==J:
                AiJ  = self.solver.Aib[:,J]
            LUii  = self.solver.solver_ii

            def matmat(v,transpose=False):
                if I==J:
                    return v 
                if (v.ndim == 1):
                    v_tmp = v[:,np.newaxis]
                else:
                    v_tmp = v

                if (not transpose):
                    result = LUii.matmat(AiJ@v_tmp)[I]
                else:
                    result      = np.zeros(shape=(len(self.solver._Ji,)))
                    result[I]   = v_tmp
                    result      = AiJ.T @ LUii.rmatmat(v_tmp)

                if (v.ndim == 1):
                    result = result.flatten()
                return result

            return LinearOperator(shape=(len(I),len(J)),\
                matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
                matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))
        if stType=='DtN':
            AIJ = self.solver.Abb[I][:,J]
            AIi  = self.solver.Abi[I]
            AiJ  = self.solver.Aib[:,J]
            LUii  = self.solver.solver_ii

            def matmat(v,transpose=False):

                if (v.ndim == 1):
                    v_tmp = v[:,np.newaxis]
                else:
                    v_tmp = v

                if (not transpose):

                    result = AIJ @ v_tmp
                    result -= AIi @ LUii.matmat(AiJ @ v_tmp)
                else:
                    result = AIJ.T @ v_tmp
                    result -= AiJ.T @ LUii.rmatmat(AIi.T @ v_tmp)

                if (v.ndim == 1):
                    result = result.flatten()
                return result

            return LinearOperator(shape=AIJ.shape,\
                matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
                matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))