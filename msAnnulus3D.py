import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import time
import hps.hps_multidomain as HPS
import hps.geom as hpsGeom
from scipy.sparse        import block_diag
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.sparse.linalg   import LinearOperator
from scipy.sparse.linalg import gmres
from solver.solver import stMap
import matAssembly.matAssembler as mA
from matplotlib.patches import Polygon
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


try:
	from petsc4py import PETSc
	petsc_imported = True
except:
	petsc_imported = False

class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.resList=[]
    def __call__(self, rk=None):
        self.niter += 1
        self.resList+=[rk]
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

kh = 20.247

def bfield(xx,kh):
    
    b = np.ones(shape = (xx.shape[0],))
    
    kh_fun = -kh**2 * b
    return kh_fun


def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c33(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return bfield(p)
Lapl=pdo.PDO3d(c11,c22,c33,None,None,None,None,None,None,c)

def bc(p):
    return 1.

bnds = [[0.,0.,0],[3.,1.,.5]]
box_geom   = np.array([[0,0,0],[3.0,1.0,.5]])
Om=stdGeom.Box(bnds)

const_theta = 1/(np.pi/3)
r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

z1 = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.cos(zz[:,0]/const_theta) )
z2 = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.sin(zz[:,0]/const_theta) )
z3 = lambda zz: zz[:,2]

y1 = lambda zz: const_theta* np.atan2(zz[:,1],zz[:,0])
y2 = lambda zz: r(zz) - 1
y3 = lambda zz: zz[:,2]

y1_d1    = lambda zz: -const_theta     * np.divide(zz[:,1], r(zz)**2)
y1_d2    = lambda zz: +const_theta     * np.divide(zz[:,0], r(zz)**2)
y1_d1d1  = lambda zz: +2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r(zz)**4)
y1_d2d2  = lambda zz: -2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r(zz)**4)
y1_d1d1 = None; y1_d2d2 = None


y2_d1    = lambda zz: np.divide(zz[:,0], r(zz))
y2_d2    = lambda zz: np.divide(zz[:,1], r(zz))
y2_d1d1  = lambda zz: np.divide(zz[:,1]**2, r(zz)**3)
y2_d2d2  = lambda zz: np.divide(zz[:,0]**2, r(zz)**3)

y3_d3    = lambda zz: np.ones(shape=(zz.shape[0],))

param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                    y1_d1=y1_d1, y1_d2=y1_d2,\
                    y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                    y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,y3_d3=y3_d3)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield, kh)


def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-14 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14 or np.abs(p[2]-bnds[0][2])<1e-14 or np.abs(p[2]-bnds[1][2])<1e-14

H = 1./4.
N = (int)(bnds[1][0]/H)
p = 20
a = H/4.
Sl_list = []
Sr_list = []

Sl_rk_list = []
Sr_rk_list = []

rhs_list = []
trk     = 0
tol = 1e-5
data = 0
for slabInd in range(N-1):
    xl = slabInd*H
    xr = (slabInd+2)*H
    xc = (slabInd+1)*H
    geom = hpsGeom.BoxGeometry(np.array([[xl,0.,0.],[xr,1.,.5]]))
    disc = HPS.HPSMultidomain(pdo_mod, geom, a, p)
    XX = disc._XX
    XXb = XX[disc.Jx,:]
    XXi = XX[disc.Ji,:]
    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-10]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-10]
    Ic = [i for i in range(len(disc.Ji)) if np.abs(XXi[i,0]-xc)<1e-10]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]
    print("len(Il) = ",len(Il))
    print("len(Ir) = ",len(Ir))
    print("len(Ic) = ",len(Ic))
    print("len(Igb) = ",len(Igb))
    fgb = np.array([bc(XXb[i,:]) for i in Igb])
    start_rk = time.time()    
    def smatmat(v,I,J,transpose=False):
        if (v.ndim == 1):
            v_tmp = v[:,np.newaxis]
        else:
            v_tmp = v

        if (not transpose):
            result = (disc.solver_Aii@(disc.Aix[:,J]@v_tmp))[I]
        else:
            result      = np.zeros(shape=(len(disc.Ji),v.shape[1]))
            result[I,:] = v_tmp
            result      = disc.Aix[:,J].T @ (disc.solver_Aii.T@(result))
        if (v.ndim == 1):
            result = result.flatten()
        return result

    Linop_r = LinearOperator(shape=(len(Ic),len(Ir)),\
        matvec = lambda v:smatmat(v,Ic,Ir), rmatvec = lambda v:smatmat(v,Ic,Ir,transpose=True),\
        matmat = lambda v:smatmat(v,Ic,Ir), rmatmat = lambda v:smatmat(v,Ic,Ir,transpose=True))
    Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
        matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
        matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))
    st_r = stMap(Linop_r,XXb[Ir,:],XXi[Ic,:])
    st_l = stMap(Linop_l,XXb[Il,:],XXi[Ic,:])
    assembler = mA.rkHMatAssembler((p//2)*(p//2))
    rkMat_r = assembler.assemble(st_r)
    data+=assembler.tree.total_bytes()
    del assembler
    assembler = mA.rkHMatAssembler((p//2)*(p//2))
    rkMat_l = assembler.assemble(st_l)
    data+=assembler.tree.total_bytes()
    del assembler
    print("data = ",data)
    Sl_rk_list += [rkMat_l]
    Sr_rk_list += [rkMat_r]
    stop_rk = time.time()
    trk+=stop_rk-start_rk
    rhs = splinalg.spsolve(disc.Aii,disc.Aix[:,Igb]@fgb)
    rhs = rhs[Ic]
    rhs_list+=[rhs]
    loopstr = 'loop '+str(slabInd)+' of '+str(N-1)+' done'
    print(loopstr)
    del disc,st_r,st_l,geom,XX,XXi,XXb

print("blocks done")
Sl_rk_list = tuple(Sl_rk_list)
Sr_rk_list = tuple(Sr_rk_list)
print("tuples done")
stop_construct = time.time()
nc = len(Ic)
print("nc = ",nc)
Ntot = (N-1)*nc
print("Ntot = ",Ntot)
print("Nslab = ",(N-1))
print("data (MB) = ",data/1e6)
rhstot = np.zeros(shape = (Ntot,))


for rhsInd in range(len(rhs_list)):
    rhstot[rhsInd*nc:(rhsInd+1)*nc]=-rhs_list[rhsInd]
print("rhs formed")
start_solve = time.time()
def smatmat(v,transpose=False):
    if (v.ndim == 1):
        v_tmp = v[:,np.newaxis]
    else:
        v_tmp = v
    result  = v_tmp.copy()
    if (not transpose):
        for i in range(N-2):
            result[i*nc:(i+1)*nc]+=Sr_rk_list[i]@v_tmp[(i+1)*nc:(i+2)*nc]
        for i in range(N-2):
            result[(i+1)*nc:(i+2)*nc]+=Sl_rk_list[i+1]@v_tmp[i*nc:(i+1)*nc]
    else:
        for i in range(N-2):
            result[i*nc:(i+1)*nc]+=Sl_rk_list[i+1].T@v_tmp[(i+1)*nc:(i+2)*nc]
        for i in range(N-2):
            result[(i+1)*nc:(i+2)*nc]+=Sr_rk_list[i].T@v_tmp[i*nc:(i+1)*nc]
    if (v.ndim == 1):
        result = result.flatten()
    return result

Linop = LinearOperator(shape=(Ntot,Ntot),\
matvec = smatmat, rmatvec = lambda v: smatmat(v,transpose=True),\
matmat = smatmat, rmatmat = lambda v: smatmat(v,transpose=True))
print("Linop formed")

gInfo = gmres_info()
stol = 1e-8*H*H
print("Gmres start")
if petsc_imported == True:
    uhat,info   = gmres(Linop,rhstot,rtol=stol,callback=gInfo,maxiter=500,restart=500)
else:
    uhat,info   = gmres(Linop,rhstot,tol=stol,callback=gInfo,maxiter=500,restart=500)
print("Gmres stop, info = ",print(info))
stop_solve = time.time()
res = Linop@uhat-rhstot

print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhs))
print("GMRES iters              = ", gInfo.niter)
print("constuction time rk.     = ",trk)
print("par. constuction time rk.= ",trk/(N-1))
print("solve time               = ",(stop_solve-start_solve))
print("par. solve time          = ",(stop_solve-start_solve)/(N-1))
print("data (MB)                = ",data/1e6)
print("data orig (MB)           = ",(8*Ntot+8*(nc*nc)*2.*(N-1))/1e6)
print("==================================")
np.save('uhat_3dAnnulus.npy', uhat)
del Linop,Sr_rk_list,Sl_rk_list
