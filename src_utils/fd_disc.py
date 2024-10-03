from scipy.sparse import kron, diags, block_diag
from scipy.sparse import eye as speye
import scipy.sparse.linalg as spla
import numpy as np
from time import time

#######          PARTIAL DIFFERENTIAL OPERATORS FOR ELLIPTIC PDES #########
# as described in this paper
# https://users.oden.utexas.edu/~pgm/Pubs/2016_HPS_3D_final.pdf

class PDO2d:
    def __init__(self, c11, c22, c12=None, \
                 c1= None, c2 = None, c = None):
        self.c11, self.c22 = c11, c22
        self.c12 = c12
        self.c1, self.c2    = c1, c2
        self.c = c

class PDO3d:
    def __init__(self, c11, c22, c33, c12=None, c13 = None, c23 = None, \
                 c1= None, c2 = None, c3 = None, c = None):
        self.c11, self.c22, self.c33 = c11, c22, c33
        self.c12, self.c13, self.c23 = c12, c13, c23
        self.c1, self.c2, self.c3    = c1, c2, c3
        self.c = c

# Input: n x d matrix of locations of n points in d dimensions
# Output: n vector of function values
def pdo_const(xxloc,c=1):
    return c * np.ones(xxloc.shape[0])

#######          INDEXES FOR RECTANGLES IN 2D and 3D      #########
# useful functions that return left, right, up, down, etc boundary indices
def get_inds_2d(XX,box_geom,h,n0,n1):
    I_L = np.argwhere(XX[0,:] < 0.5 * h + box_geom[0,0])
    I_L = I_L.copy().reshape(n1,)
    I_R = np.argwhere(XX[0,:] > -0.5 * h + box_geom[0,1])
    I_R = I_R.copy().reshape(n1,)
    I_D = np.argwhere(XX[1,:] < 0.5 * h + box_geom[1,0])
    I_D = I_D.copy().reshape(n0,)
    I_U = np.argwhere(XX[1,:] > -0.5 * h + box_geom[1,1])
    I_U = I_U.copy().reshape(n0,)

    I_DIR = np.hstack((I_D,I_U))
    I_DIR = np.unique(I_DIR)
    I_L = np.setdiff1d(I_L,I_DIR)
    I_R = np.setdiff1d(I_R,I_DIR)
    return I_L,I_R,I_DIR

def get_inds_3d(XX,box_geom,h,n0,n1,n2):
    I_L = np.argwhere(XX[0,:] < 0.5 * h + box_geom[0,0])
    I_L = I_L.copy().reshape(n1*n2,)
    I_R = np.argwhere(XX[0,:] > -0.5 * h + box_geom[0,1])
    I_R = I_R.copy().reshape(n1*n2,)
    I_D = np.argwhere(XX[1,:] < 0.5 * h + box_geom[1,0])
    I_D = I_D.copy().reshape(n0*n2,)
    I_U = np.argwhere(XX[1,:] > -0.5 * h + box_geom[1,1])
    I_U = I_U.copy().reshape(n0*n2,)

    I_B = np.argwhere(XX[2,:] < 0.5 * h + box_geom[2,0])
    I_B = I_B.copy().reshape(n0*n1,)
    I_F = np.argwhere(XX[2,:] > -0.5 * h + box_geom[2,1])
    I_F = I_F.copy().reshape(n0*n1,)

    I_DIR = np.hstack((I_D,I_U,I_B,I_F))
    I_DIR = np.unique(I_DIR)
    I_L   = np.setdiff1d(I_L,I_DIR)
    I_R   = np.setdiff1d(I_R,I_DIR)
    return I_L,I_R,I_DIR

#######          GRID FOR 2D and 3D      #########
# Given box geometry, generates the grid points in a rectangle.
def grid(box_geom,h):
    d = box_geom.shape[0]
    xx0 = np.arange(box_geom[0,0],box_geom[0,1]+0.5*h,h)
    xx1 = np.arange(box_geom[1,0],box_geom[1,1]+0.5*h,h)
    if (d == 3):
        xx2 = np.arange(box_geom[2,0],box_geom[2,1]+0.5*h,h)

    if (d == 2):
        n0 = xx0.shape[0]
        n1 = xx1.shape[0]

        XX0 = np.repeat(xx0,n1)
        XX1 = np.repeat(xx1,n0).reshape(-1,n0).T.flatten()
        XX = np.vstack((XX0,XX1))
        I_X_inds = get_inds_2d(XX,box_geom,h,n0,n1)
        I_X = np.hstack((I_X_inds))
        ns = np.array([n0,n1])

    elif (d == 3):
        n0 = xx0.shape[0]
        n1 = xx1.shape[0]
        n2 = xx2.shape[0]

        XX0 = np.repeat(xx0,n1*n2)
        XX1 = np.repeat(xx1,n0*n2).reshape(-1,n0).T.flatten()
        XX2 = np.repeat(xx2,n0*n1).reshape(-1,n0*n1).T.flatten()
        XX = np.vstack((XX0,XX1,XX2))
        I_X_inds = get_inds_3d(XX,box_geom,h,n0,n1,n2)
        I_X = np.hstack(I_X_inds)
        ns = np.array([n0,n1,n2])
    I_X = np.unique(I_X)
    return XX,I_X_inds,I_X,ns

def assemble_sparse(pdo_op,h,ns,XX):
    d = XX.shape[-1]
    assert d == 3

    if (d == 2):

        n0,n1 = ns
        d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
        d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')

        d0   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n0,n0),format='csc')
        d1   = (1/(2*h)) * diags([-1, 0, +1], [-1, 0, 1], shape=(n1,n1),format='csc')

        D00 = kron(d0sq,speye(n1))
        D11 = kron(speye(n0),d1sq)

        c00_diag = np.array(pdo_op.c11(XX)).reshape(n0*n1,)
        C00 = diags(c00_diag, 0, shape=(n0*n1,n0*n1))
        c11_diag = np.array(pdo_op.c22(XX)).reshape(n0*n1,)
        C11 = diags(c11_diag, 0, shape=(n0*n1,n0*n1))

        A = - C00 @ D00 - C11 @ D11

        if (pdo_op.c12 is not None):
            c_diag = np.array(pdo_op.c12(XX)).reshape(n0*n1,)
            S      = diags(c_diag,0,shape=(n0*n1,n0*n1))

            D01 = kron(d0,d1)
            A  -= 2 * S @ D01

        if (pdo_op.c1 is not None):
            c_diag = np.array(pdo_op.c1(XX)).reshape(n0*n1,)
            S      = diags(c_diag,0,shape=(n0*n1,n0*n1))

            D0 = kron(d0,speye(n1))
            A  += S @ D0

        if (pdo_op.c2 is not None):
            c_diag = np.array(pdo_op.c1(XX)).reshape(n0*n1,)
            S      = diags(c_diag,0,shape=(n0*n1,n0*n1))

            D0 = kron(speye(n0),d1)
            A  += S @ D1

        if (pdo_op.c is not None):
            c_diag = np.array(pdo_op.c(XX)).reshape(n0*n1,)
            S = diags(c_diag, 0, shape=(n0*n1,n0*n1))
            A += S

    elif (d == 3):

        n0,n1,n2 = ns
        d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
        d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')
        d2sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n2, n2),format='csc')

        D00 = kron(d0sq,kron(speye(n1),speye(n2)))
        D11 = kron(speye(n0),kron(d1sq,speye(n2)))
        D22 = kron(speye(n0),kron(speye(n1),d2sq))

        N = n0*n1*n2
        c00_diag = np.array(pdo_op.c11(XX)).reshape(N,)
        C00 = diags(c00_diag, 0, shape=(N,N))
        c11_diag = np.array(pdo_op.c22(XX)).reshape(N,)
        C11 = diags(c11_diag, 0, shape=(N,N))
        c22_diag = np.array(pdo_op.c33(XX)).reshape(N,)
        C22 = diags(c22_diag, 0, shape=(N,N))

        A = - C00 @ D00 - C11 @ D11 - C22 @ D22

        if ((pdo_op.c1 is not None) or \
            (pdo_op.c2 is not None) or \
            (pdo_op.c3 is not None) or \
            (pdo_op.c12 is not None) or \
            (pdo_op.c13 is not None) or \
            (pdo_op.c23 is not None)):
            raise ValueError

        if (pdo_op.c is not None):
            c_diag = np.array(pdo_op.c(XX)).reshape(N,)
            S = diags(c_diag, 0, shape=(N,N))
            A += S
    return A

#######          GRID FOR 2D and 3D      #########
# Given box geometry, generates the grid points in a rectangle.
class FDDiscretization:
    def __init__(self,box_geom,h,pdo_op,kh=0):
        XX, inds_tuple, self.I_X, self.ns = grid(box_geom,h)
        self.XX = XX.T
        self.h = h
        self.box_geom = box_geom
        self.d = self.ns.shape[0]
        self.pdo_op = pdo_op
        self.kh     = kh

        self.I_L,self.I_R,self.I_DIR = inds_tuple

        I_tot = np.arange(self.XX.shape[0])
        self.I_C = np.setdiff1d(I_tot,self.I_X)

        self.A = assemble_sparse(self.pdo_op,self.h,self.ns,self.XX)

    def setup_solver_CC(self,solver_op=None):

    	if (solver_op is None):

    		LU_CC = spla.splu(fd.A[fd.I_C][:,fd.I_C].tocsc())
    		self.solver_CC = spla.LinearOperator(shape=(LU_CC.shape[0],LU_CC.shape[0]),\
    			matvec = lambda x: LU_CC.solve(x),\
    			rmatvec= lambda x: LU_CC.solve(x,trans='T'),\
    			matmat = lambda x: LU_CC.solve(x),\
    			rmatmat= lambda x: LU_CC.solve(x,trans='T'))
    	else:
    		self.solver_CC = solver_op


    def solve_dir(self,dir_data):

        uu_sol           = np.zeros(self.XX.shape[0])
        uu_sol[self.I_X] = dir_data
        tmp              = self.A[self.I_C][:,self.I_X] @  dir_data
        uu_sol[self.I_C] = - self.solver_CC.matvec ( tmp  )
        return uu_sol

    def check_discretization(self):
        XX_tmp = self.XX.copy();
        #### check that we have discretized the PDE correctly
        dd0 = XX_tmp[:,0] - 2; dd1 = XX_tmp[:,1] - 2; dd2 = XX_tmp[:,2] - 2;

        ddsq      = np.multiply(dd0,dd0) + np.multiply(dd1,dd1) + np.multiply(dd2,dd2)
        r_points  = np.sqrt(ddsq)

        if (self.kh == 0):
            uu_exact = np.divide(np.ones(ddsq.shape), 4 * np.pi * r_points)
        else:
            uu_exact = np.divide( np.sin(self.kh * r_points), 4 * np.pi * r_points)

        uu_calc     = self.solve_dir(uu_exact[self.I_X])
        relerr_disc = np.linalg.norm(uu_exact - uu_calc) / np.linalg.norm(uu_exact)

        if (relerr_disc > 1):
            print("\t The pollution effect leads to inaccurate discretization. Rerun with more ppw.")
            raise ValueError
        return relerr_disc