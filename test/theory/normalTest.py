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
from solver.spectral import spectralSolver as spectral
import geometry.geom_2D.square as square


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
    return np.ones_like(p[:,0])+.5*np.cos(2.*np.pi*p[:,0])
def c22(p):
    return np.ones_like(p[:,1])+.5*np.sin(2.*np.pi*p[:,1])*(p[:,0]**2)
def c12(p):
    return .1*np.ones_like(p[:,1])+.1*p[:,1]*np.sin(3.*np.pi*p[:,0])

Lapl = pdo.PDO2d(c11=c11,c22=c22)#,c12=c12)
def bc(p):
    return np.sin(np.pi*p[:,0])*np.sinh(np.pi*p[:,1])

kvec = [2,3,4,5,6,7]
errInf = np.zeros(shape=(len(kvec),))
errBlock = np.zeros(shape=(len(kvec),))
relerrBlock = np.zeros(shape=(len(kvec),))
cond_eig = np.zeros(shape=(len(kvec),))
cond_svd = np.zeros(shape=(len(kvec),))
Hvec = np.zeros(shape=(len(kvec),))

method = 'stencil'

for indk in range(len(kvec)):
    k = kvec[indk]
    N = (2**k)
    dSlabs,connectivity,H = square.dSlabs(N)
    Hvec[indk] = H
    if method =='spectral':
        ordy = 128
    if method =='stencil':
        ordy = 128
    ordx = int(np.round(2*ordy*H))
    if method == 'spectral':
        if ordx%2:
            ordx += 1
    if method == 'stencil':
        if not ordx%2:
            ordx += 1
    ord = [ordx,ordy]

    assembler = mA.denseMatAssembler()
    opts = solverWrap.solverOptions(method,ord)
    OMS = oms.oms(dSlabs,Lapl,lambda p:square.gb(p,False),opts,connectivity)
    S_op,rhs = OMS.construct_Stot_and_rhstot(bc,assembler)
    S = S_op@np.identity(S_op.shape[0])
    
    if method=='spectral':
        cc= spectral.clenshaw_curtis_compute(ordy+1)[1]
        w = np.sqrt(cc[1:ordy])
        W = np.diag(w)
        S = np.kron(np.identity(N-1),W)@S@np.kron(np.identity(N-1),np.linalg.inv(W))
    nc = OMS.nc
    
    
    IND = (N-1)//2
    S12 = S[IND*nc:(IND+1)*nc,:][:,(IND+1)*nc:(IND+2)*nc]
    S21 = S[(IND+1)*nc:(IND+2)*nc,:][:,IND*nc:(IND+1)*nc]
    errBlock[indk] = np.linalg.norm(S12-S21.T,ord=2)
    relerrBlock[indk] = np.linalg.norm(S12-S21.T,ord=2)/np.linalg.norm(S12,ord=2)
    e = np.linalg.eigvals(S)
    ae = np.abs(e)
    s = np.linalg.svdvals(S)
    ae = np.sort(ae)
    s = np.sort(s)
    errInf[indk] = np.linalg.norm(ae-s,ord=np.inf)
    cond_eig[indk] = ae[-1]/ae[0]
    cond_svd[indk] = s[-1]/s[0]
    print("============SUMMARY ERRS===============")
    print(cond_svd[indk])
    print(cond_eig[indk])
    print(errBlock[indk])
    print("=======================================")

fileName = 'err_svd_eig_'+method+'.csv'
errMat = np.zeros(shape=(len(kvec),5))
errMat[:,0] = Hvec
errMat[:,1] = errInf
errMat[:,2] = cond_svd
errMat[:,3] = cond_eig
errMat[:,4] = errBlock
with open(fileName,'w') as f:
    f.write('H,err,cond_svd,cond_eig,errBlock\n')
    np.savetxt(f,errMat,fmt='%.8e',delimiter=',')


cH = 1./(Hvec*Hvec)
cH*=2*cond_svd[0]/cH[0]

plt.figure(1)
plt.loglog(Hvec,errInf)
plt.figure(2)
plt.loglog(Hvec,cond_svd)
plt.loglog(Hvec,cond_eig)
plt.loglog(Hvec,cH)
plt.legend(['svd','eig','cH'])
plt.figure(3)
plt.loglog(Hvec,np.abs(cond_svd-cond_eig)/cond_svd)
plt.figure(4)
plt.loglog(Hvec,cond_svd/cond_eig)

plt.figure(5)
plt.loglog(Hvec,errBlock)
plt.loglog(Hvec,Hvec)
plt.legend(['err','H2'])
plt.show()
