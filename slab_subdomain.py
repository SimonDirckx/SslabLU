from spectralmultidomain.hps.hps_multidomain   import HPSMultidomain
from spectralmultidomain.hps.fd_discretization import FDDiscretization
from scipy.sparse.linalg   import LinearOperator
import numpy as np

def find_subindices_location(Jsub,Jlong):
    return np.intersect1d(Jsub,Jlong,return_indices=True)[2]

class SlabSubdomain:

    def __init__(self,pdo,geom,a,p,double_slab=True):

        if (p > 2):
            solver    = HPSMultidomain(pdo,geom,a,p)

            npan_x    = solver.npan_dim[0]
            if (double_slab and np.mod(npan_x,2) == 1 ):
                raise ValueError("expected an even number of panels in x direction for double_slab")
        else:
            solver    = FDDiscretization(pdo,geom,a)

            npoints_x = solver.npoints_dim[0]
            if (double_slab and np.mod(npoints_x,2)== 0):
                raise ValueError("expected an odd number of points in x direction for double_slab")

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

    # the `cut points' on the interior of a double slab
    @property
    def Jc(self):

        len0         = self.solver.geom.bounds[1,0] - self.solver.geom.bounds[0,0]
        center_bound = self.solver.geom.bounds[0,0] + 0.5*len0
        hmin         = np.max(self.XX[1] - self.XX[0])

        Lbool   = self.XX[self.Ji,0] < center_bound + 0.1*hmin
        Rbool   = self.XX[self.Ji,0] > center_bound - 0.1*hmin
        return self.Ji[np.where(np.logical_and(Lbool,Rbool))[0]]

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
    def Tll(self):
        I1 = find_subindices_location(self.Jl,self.Jx)
        return self.get_TX1X2_op(I1,I1)
    
    @property
    def Tlr(self):
        I1 = find_subindices_location(self.Jl,self.Jx)
        I2 = find_subindices_location(self.Jr,self.Jx)
        return self.get_TX1X2_op(I1,I2)

    @property
    def Trl(self):
        I1 = find_subindices_location(self.Jr,self.Jx)
        I2 = find_subindices_location(self.Jl,self.Jx)
        return self.get_TX1X2_op(I1,I2)

    @property
    def Trr(self):
        I1 = find_subindices_location(self.Jr,self.Jx)
        return self.get_TX1X2_op(I1,I1)
    
    # given Dirichlet data on left boundary,
    # return consistent solution on cut boundary
    @property
    def Scl(self):
        I1 = find_subindices_location(self.Jc,self.Ji)
        I2 = find_subindices_location(self.Jl,self.Jx)
        return self.get_SI1X2_op(I1,I2)

    # given Dirichlet data on right boundary,
    # return consistent solution on cut boundary
    @property
    def Scr(self):
        I1 = find_subindices_location(self.Jc,self.Ji)
        I2 = find_subindices_location(self.Jr,self.Jx)
        return self.get_SI1X2_op(I1,I2)

    def get_TX1X2_op(self,I1,I2):

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

    def get_SI1X2_op(self,I1,I2):

        Aix2  = self.solver.Aix[:,I2]
        LUii  = self.solver.solver_Aii

        def matmat(v,transpose=False):

            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):

                tmp    = - LUii.matmat(Aix2 @ v_tmp)
                result = tmp[I1]

            else:

                tmp    = np.zeros((self.Ji.shape[0],v_tmp.shape[-1]))
                tmp[I1]= v_tmp 

                result = - LUii.rmatmat(tmp)
                result = Aix2.T @ result

            if (v.ndim == 1):
                result = result.flatten()
            return result

        return LinearOperator(shape=(I1.shape[0],I2.shape[0]),\
            matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
            matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))