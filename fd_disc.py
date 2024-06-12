from scipy.sparse import kron, diags, block_diag
from scipy.sparse import eye as speye
import scipy.sparse.linalg as spla
import numpy as np
from time import time

#######          INDEXES FOR RECTANGLES IN 3D      #########
# useful functions that return left, right, up, down, etc boundary indices

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

#######          GRID FOR 3D      #########
# Given box geometry, generates the grid points in a rectangle.
def grid(box_geom,h):
    d = box_geom.shape[0]
    assert d == 3

    xx0 = np.arange(box_geom[0,0],box_geom[0,1]+0.5*h,h)
    xx1 = np.arange(box_geom[1,0],box_geom[1,1]+0.5*h,h)
    xx2 = np.arange(box_geom[2,0],box_geom[2,1]+0.5*h,h)

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

def assemble_sparse(h,ns,kh_param):

    n0,n1,n2 = ns
    d0sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n0, n0),format='csc')
    d1sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n1, n1),format='csc')
    d2sq = (1/(h*h)) * diags([1, -2, 1], [-1, 0, 1], shape=(n2, n2),format='csc')

    D00 = kron(d0sq,kron(speye(n1),speye(n2)))
    D11 = kron(speye(n0),kron(d1sq,speye(n2)))
    D22 = kron(speye(n0),kron(speye(n1),d2sq))

    N = n0*n1*n2
    A = - D00 - D11 - D22

    c_diag = np.ones(N,) * -kh_param ** 2
    S = diags(c_diag, 0, shape=(N,N))
    A += S
    return A


#######          GRID FOR 3D      #########
# Given box geometry, generates the grid points in a rectangle.
class FD_disc:
    def __init__(self,box_geom,h,kh_param=0):
        XX, inds_tuple, self.I_X, self.ns = grid(box_geom,h)
        self.XX = XX.T
        self.h = h
        self.box_geom = box_geom
        self.kh_param = kh_param
        self.d = self.ns.shape[0]

        self.I_L,self.I_R,self.I_DIR = inds_tuple

        I_tot = np.arange(self.XX.shape[0])
        self.I_C = np.setdiff1d(I_tot,self.I_X)

        self.A = assemble_sparse(self.h,self.ns,self.kh_param)

        A_CC = self.A[self.I_C][:,self.I_C].tocsc()
        tic = time()
        self.LU_CC = spla.splu(A_CC)
        toc = time() - tic
        print("\t Time for sparse LU using scipy is %5.2f" % toc)
        self.schur_complement_XX   = self.XX[self.I_L,1:]


    def solve_dir(self,dir_data):
        uu_sol = np.zeros(self.XX.shape[0])
        uu_sol[self.I_X] = dir_data

        tmp = self.A[self.I_C][:,self.I_X] @  dir_data
        uu_sol[self.I_C] = - self.LU_CC.solve( tmp  )

        return uu_sol

    # Apply Schur complement A[I_R, I_C] @ (A_CC^{-1} @ A[I_C, I_R] to a vector
    # also can apply the transpose
    def apply_schur_complement(self,str_schur,vec,transpose=False):

        if (not (len(str_schur) == 2)):
            raise ValueError("invalid str_schur -- must be LL,LR,RL,or RR")

        if (str_schur[0] == 'L'):
            I1 = self.I_L
        elif (str_schur[0] == 'R'):
            I1 = self.I_R
        else:
            raise ValueError("invalid str_schur -- must be LL,LR,RL,or RR")

        if (str_schur[1] == 'L'):
            I2 = self.I_L
        elif (str_schur[1] == 'R'):
            I2 = self.I_R
        else:
            raise ValueError("invalid str_schur -- must be LL,LR,RL,or RR")

        if (not transpose):
            tmp     = self.A[self.I_C][:,I2] @ vec
            result  = - self.A[I1][:,self.I_C] @ self.LU_CC.solve(tmp)

        else:
            tmp     = self.A[I1][:,self.I_C].T @ vec
            result  = - self.A[self.I_C][:,I2].T @ self.LU_CC.solve(tmp,trans='T')
        return result + self.A[I1][:,I2] @ vec

    def check_disc(self):
        XX = self.XX; kh = self.kh_param

        #### check that we have discretized the PDE correctly
        dd0 = XX[:,0] - 2; dd1 = XX[:,1] - 2; dd2 = XX[:,2] - 2;

        ddsq = np.multiply(dd0,dd0) + np.multiply(dd1,dd1) + np.multiply(dd2,dd2)
        r_points    = np.sqrt(ddsq)

        if (kh == 0):
            uu_exact = np.divide(np.ones(ddsq.shape), 4 * np.pi * r_points)
        else:
            uu_exact = np.divide( np.sin(kh * r_points), 4 * np.pi * r_points)

        uu_calc = self.solve_dir(uu_exact[self.I_X])
        relerr = np.linalg.norm(uu_exact - uu_calc) / np.linalg.norm(uu_exact)
        print("\t Discretization is accurate to %5.2e compared to known solution of PDE"%relerr)
        if (relerr > 1e-1):
            print("\t The pollution effect may be leading to inaccurate discretization. Rerun with more ppw.")
