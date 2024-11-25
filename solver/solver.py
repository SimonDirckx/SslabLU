import hps.hps_multidomain as hps
import geometry.slabGeometry as slabGeom
import hps.pdo as pdo



class solverOptions:
    """
    Class that encodes the options for a local slab Solver
    @param:
    type:       type of discretization (HPS/cheb/stencil)
    ordx,ordy:  order in x and y directions
    a:          characteristic scale in case of HPS
    """
    def __init__(self,type:str,ordx,ordy,a=1):
        self.type   =   type
        self.ordx   =   ordx
        self.ordy   =   ordy
        self.a      =   a


class localSolver:
    """
    Class for local Solver
    @param:
    opts:       slab options
    """
    def __init__(self,opts:solverOptions):
        self.opts=opts

    def construct(self,geom:slabGeom,PDE:pdo.PDO2d):
        """
        Actual construction of the local solver
        """
        self.geom   = geom
        self.PDE    = PDE
        if self.opts.type=='HPS':
            self.solver = hps(PDE, geom, self.opts.a, self.opts.ordx)
        
        self.XX = self.solver.XX
        self.ndofs = self.XX.shape[0]
    def get_T(self,I,J):
        """DtN map from I to J"""
        return 0
    def get_S(self,I,J):
        """DtD map from I to J"""
        return 0

class localMapper:
    """
    Class for local Mapper
    @param:
    typeS2T:    type of source-to-target map
    locSl:      location of left sources
    locSr:      location of right sources
    locTl:      location of left targets
    locTr:      location of right targets

    @leftMap:   maps left sources to right targets
    @rightMap:  maps right sources to left targets

    """
    def __init__(self,locSl,locSr,locTl,locTr,typeS2T:str='DtD'):
        self.typeS2T    = typeS2T
        self.locSl      = locSl
        self.locSr      = locSr
        self.locTl      = locTl
        self.locTr      = locTr


    def leftMap(self,solver:localSolver):
        if(self.typeS2T == 'DtN'):
            return solver.get_T(self.locSl,self.locTr)
        if(self.typeS2T == 'DtD'):
            return solver.get_S(self.locSl,self.locTr)

    def rightMap(self,solver:localSolver):
        if(self.typeS2T == 'DtN'):
            return solver.get_T(self.locSr,self.locTl)
        if(self.typeS2T == 'DtD'):
            return solver.get_S(self.locSr,self.locTl)