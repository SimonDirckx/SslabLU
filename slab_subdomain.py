from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from scipy.sparse.linalg   import LinearOperator
import numpy as np

def find_subindices_location(Jsub,Jlong):
    return np.intersect1d(Jsub,Jlong,return_indices=True)[2]

class SlabSubdomain:

    def __init__(self,pdo,geom,a,p):

        if (p > 2):
            solver    = HPSMultidomain(pdo,geom,a,p)
        else:
            solver    = FDDiscretization(pdo,geom,a)
        self._solver  = solver
        self._solver.setup_solver_Aii()

    @property
    def XX(self):
        return self.solver.XX

    @property
    def solver(self):
        return self._solver


    ##########################################################################
    # index vectors

    # exterior points
    @property
    def Jx(self):

        return self.solver.Jx

    # interior points
    @property
    def Ji(self):
        return self.solver.Ji

    # the left exterior points
    @property
    def Jl(self):

        hmin    = np.max(self.XX[1] - self.XX[0])
        Lbool   = self.XX[self.Jx,0] < self.solver.geom.bounds[0,0] + 0.1*hmin
        return self.Jx[np.where(Lbool)[0]]

    # the right exterior points
    @property
    def Jr(self):

        hmin    = np.max(self.XX[1] - self.XX[0])
        Lbool   = self.XX[self.Jx,0] > self.solver.geom.bounds[1,0] - 0.1*hmin
        return self.Jx[np.where(Lbool)[0]]

    ############################################################################
    # linear algebra sparse operators

    @property
    def T_LL(self):
        I1 = find_subindices_location(self.Jl,self.Jx)
        return self.get_TX1X2_op(I1,I1)
    
    @property
    def T_LR(self):
        I1 = find_subindices_location(self.Jl,self.Jx)
        I2 = find_subindices_location(self.Jr,self.Jx)
        return self.get_TX1X2_op(I1,I2)

    @property
    def T_RL(self):
        I1 = find_subindices_location(self.Jr,self.Jx)
        I2 = find_subindices_location(self.Jl,self.Jx)
        return self.get_TX1X2_op(I1,I2)

    @property
    def T_RR(self):
        I1 = find_subindices_location(self.Jr,self.Jx)
        return self.get_TX1X2_op(I1,I1)
    

    def get_TX1X2_op(self,I1,I2):
        nX = self.solver.Jx.shape[0]

        Ax1x2 = self.solver.Axx[I1][:,I2]
        Ax1i  = self.solver.Axi[I1]
        Aix2  = self.solver.Aix[:,I2]
        LUii  = self.solver.solver_Aii

        def matmat(v,transpose=False):

            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):

                result = Ax1x2 @ v_tmp
                result -= Ax1i @ LUii.matmat(Aix2 @ v_tmp)
            else:
                result = Ax1x2.T @ v_tmp
                result -= Aix2.T @ LUii.rmatmat(Ax1i.T @ v_tmp)

            if (v.ndim == 1):
                result = result.flatten()
            return result

        return LinearOperator(shape=Ax1x2.shape,\
            matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
            matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))