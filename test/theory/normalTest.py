import numpy as np
import hps.pdo as pdo
import matplotlib.pyplot as plt
import multislab.oms as oms
import matAssembly.matAssembler as mA
import solver.solver as solverWrap
import hps.pdo as pdo
from solver.stencil.stencilSolver import stencilSolver as stencil
import solver.stencil.geom as stencilGeom
from matplotlib import cm
from multislab.oms import slab
from scipy.sparse.linalg import LinearOperator

class stMap:
    def __init__(self,A:LinearOperator,XXI,XXJ):
        self.XXI = XXI
        self.XXJ = XXJ
        self.A = A

def compute_stmaps(Il,Ic,Ir,XXi,XXb,solver):
        A_solver = solver.solver_ii    
        def smatmat(v,I,J,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[...,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = (A_solver@(solver.Aib[...,J]@v_tmp))[I,...]
            else:
                result      = np.zeros(shape=(len(solver.Ii),v.shape[1]))
                result[I,:] = v_tmp
                result      = solver.Aib[...,J].T @ (A_solver.T@(result))
            if (v.ndim == 1):
                result = result.flatten()
            return result

        Linop_r = LinearOperator(shape=(len(Ic),len(Ir)),\
            matvec = lambda v:smatmat(v,Ic,Ir), rmatvec = lambda v:smatmat(v,Ic,Ir,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Ir), rmatmat = lambda v:smatmat(v,Ic,Ir,transpose=True))
        Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
            matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))
        
        st_r = stMap(Linop_r,XXb[Ir,...],XXi[Ic,...])
        st_l = stMap(Linop_l,XXb[Il,...],XXi[Ic,...])
        return st_l,st_r


def c11(p):
    return np.ones_like(p[:,0])+.1*np.cos(5.*np.pi*p[:,0])
def c22(p):
    return np.ones_like(p[:,1])+.1*np.sin(10.*np.pi*p[:,1])*(p[:,0]**2)
def c12(p):
    return .1*np.ones_like(p[:,1])+.1*p[:,1]*np.sin(9.*np.pi*p[:,0])

Lapl = pdo.PDO2d(c11=c11,c22=c22,c12=c12)
bnds = [[0,0],[1,1]]
Om = stencilGeom.BoxGeometry(np.array(bnds))
def bc(p):
    return np.sin(np.pi*p[:,0])*np.sinh(np.pi*p[:,1])
def gb_vec(P):
    # P is (N, 2)
    return (
        (np.abs(P[:, 0] - bnds[0][0]) < 1e-14) |
        (np.abs(P[:, 0] - bnds[1][0]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[0][1]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[1][1]) < 1e-14)  
    )

Om = stencilGeom.BoxGeometry(np.array([[0,0],[1,1]]))

kvec = [2,3,4,5,6,7]
errInf = np.zeros(shape=(len(kvec),))
Hvec = np.zeros(shape=(len(kvec),))

for indk in range(len(kvec)):
    k = kvec[indk]
    H = 1./(2**k)
    print("H = ",H)
    Hvec[indk] = H
    ordy = 250
    ordx = int(np.round(ordy*H))
    if not ordx%2:
        ordx += 1
    ord = [ordx,ordy]

    N = (int)(1./H)

    slabs = []
    for n in range(N):
        bnds_n = [[n*H,0.],[(n+1)*H,1.]]
        slabs+=[bnds_n]

    connectivity = []
    for i in range(N-1):
        connectivity+=[[i,i+1]]

    if_connectivity = []
    for i in range(N-1):
        if i==0:
            if_connectivity+=[[-1,(i+1)]]
        elif i==N-2:
            if_connectivity+=[[(i-1),-1]]
        else:
            if_connectivity+=[[(i-1),(i+1)]]

    assembler = mA.denseMatAssembler()
    opts = solverWrap.solverOptions('stencil',ord)
    OMS = oms.oms(slabs,Lapl,gb_vec,opts,connectivity,if_connectivity)
    S_op,rhs = OMS.construct_Stot_and_rhstot(bc,assembler)
    E = np.identity(S_op.shape[0])
    S = S_op@E
    e = np.linalg.eigvals(S)
    ae = np.abs(e)
    s = np.linalg.svdvals(S)
    errInf[indk] = np.linalg.norm(np.sort(ae)-np.sort(s),ord=np.inf)

plt.figure(1)
plt.loglog(Hvec,errInf)
plt.show()
