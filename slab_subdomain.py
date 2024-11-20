from hps.hps_multidomain   import HPSMultidomain
from hps.fd_discretization import FDDiscretization
from scipy.sparse.linalg   import LinearOperator

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
        return self.solver.I_X

    @property
    def solver(self):
        return self._solver

    @property
    def T_XX_op(self):
        nX = self.solver.I_X.shape[0]

        A_XX = self.solver.A_XX
        A_XC = self.solver.A_XC
        A_CX = self.solver.A_CX
        LU_CC= self.solver.solver_CC

        def matmat(v,transpose=False):

            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):

                result = A_XX @ v_tmp
                result -= A_XC @ LU_CC.matmat(A_CX @ v_tmp)
            else:
                result = A_XX.T @ v_tmp
                result -= A_CX.T @ LU_CC.rmatmat(A_XC.T @ v_tmp)

            if (v.ndim == 1):
                result = result.flatten()
            return result

        return LinearOperator(shape=(nX,nX),\
            matvec = matmat, rmatvec = lambda x: matmat(v,transpose=True),\
            matmat = matmat, rmatmat = lambda x: matmat(v,transpose=True))