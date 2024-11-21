from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from scipy.sparse.linalg   import LinearOperator
import numpy as np

class SlabSubdomain:

    def __init__(self,pdo,geom,a,p):

        if (p > 2):
            solver    = HPSMultidomain(pdo,geom,a,p)
        else:
            solver    = FDDiscretization(pdo,geom,a)
        self._solver  = solver

        self._solver.setup_solver_CC()

    @property
    def XX(self):
        return self.solver.XX

    @property
    def I_X(self):
        return self.solver.I_X

    @property
    def I_C(self):
        return self.solver.I_C

    @property
    def solver(self):
        return self._solver

    @property
    def I_L(self):

        hmin    = np.max(self.XX[1] - self.XX[0])
        Lbool   = self.XX[self.I_X,0] < self.solver.geom.bounds[0,0] + 0.1*hmin
        return self.I_X[np.where(Lbool)[0]]

    @property
    def I_R(self):

        hmin    = np.max(self.XX[1] - self.XX[0])
        Lbool   = self.XX[self.I_X,0] > self.solver.geom.bounds[1,0] - 0.1*hmin
        return self.I_X[np.where(Lbool)[0]]

    @property
    def T_LL(self):
        I1 = np.intersect1d(self.I_L,self.I_X,return_indices=True)[2]
        return self.get_TX1X2_op(I1,I1)
    
    @property
    def T_LR(self):
        I1 = np.intersect1d(self.I_L,self.I_X,return_indices=True)[2]
        I2 = np.intersect1d(self.I_R,self.I_X,return_indices=True)[2]
        return self.get_TX1X2_op(I1,I2)

    @property
    def T_RL(self):
        I1 = np.intersect1d(self.I_R,self.I_X,return_indices=True)[2]
        I2 = np.intersect1d(self.I_L,self.I_X,return_indices=True)[2]
        return self.get_TX1X2_op(I1,I2)

    @property
    def T_RR(self):
        I1 = np.intersect1d(self.I_R,self.I_X,return_indices=True)[2]
        return self.get_TX1X2_op(I1,I1)
    

    def get_TX1X2_op(self,I1,I2):
        nX = self.solver.I_X.shape[0]

        A_X1X2 = self.solver.A_XX[I1][:,I2]
        A_X1C = self.solver.A_XC[I1]
        A_CX2 = self.solver.A_CX[:,I2]
        LU_CC= self.solver.solver_CC

        def matmat(v,transpose=False):

            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):

                result = A_X1X2 @ v_tmp
                result -= A_X1C @ LU_CC.matmat(A_CX2 @ v_tmp)
            else:
                result = A_X1X2.T @ v_tmp
                result -= A_CX2.T @ LU_CC.rmatmat(A_X1C.T @ v_tmp)

            if (v.ndim == 1):
                result = result.flatten()
            return result

        return LinearOperator(shape=A_X1X2.shape,\
            matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
            matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))