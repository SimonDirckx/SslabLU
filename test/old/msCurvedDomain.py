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

kh = 0.24*2.*np.pi

def bfield(xx,kh):
    
    b = np.ones(shape = (xx.shape[0],))
    
    kh_fun = -kh**2 * b
    return kh_fun


def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return bfield(p,kh)
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)



bnds = [[0.,0.],[3.,1.]]
box_geom   = np.array(bnds)
Om=stdGeom.Box(bnds)

const_theta = 1/(np.pi/3)
r           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

z1 = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.cos(zz[:,0]/const_theta) )
z2 = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.sin(zz[:,0]/const_theta) )

y1 = lambda zz: const_theta* np.atan2(zz[:,1],zz[:,0])
y2 = lambda zz: r(zz) - 1

y1_d1    = lambda zz: -const_theta     * np.divide(zz[:,1], r(zz)**2)
y1_d2    = lambda zz: +const_theta     * np.divide(zz[:,0], r(zz)**2)
y1_d1d1  = lambda zz: +2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r(zz)**4)
y1_d2d2  = lambda zz: -2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r(zz)**4)
y1_d1d1 = None; y1_d2d2 = None


y2_d1    = lambda zz: np.divide(zz[:,0], r(zz))
y2_d2    = lambda zz: np.divide(zz[:,1], r(zz))
y2_d1d1  = lambda zz: np.divide(zz[:,1]**2, r(zz)**3)
y2_d2d2  = lambda zz: np.divide(zz[:,0]**2, r(zz)**3)

param_geom = ParametrizedGeometry2D(box_geom,z1,z2,y1,y2,\
                    y1_d1=y1_d1, y1_d2=y1_d2,\
                    y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                    y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield, kh)
def bc(p):
    return np.sin(kh*z1(p))

def u_exact(p):
    return np.sin(kh*z1(p))

def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-10 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14

H = bnds[1][0]/16.
N = (int)(bnds[1][0]/H)
p = 18
a = 1./32.
Sl_list = []
Sr_list = []

Sl_rk_list = []
Sr_rk_list = []

rhs_list = []
disc_list = []
trk     = 0
tol = 1e-5
assembler = mA.denseMatAssembler()#(tol*H*H*.5,32)
data = 0
for slabInd in range(N-1):
    xl = slabInd*H
    xr = (slabInd+2)*H
    xc = (slabInd+1)*H

    geom = hpsGeom.BoxGeometry(np.array([[xl,0.],[xr,1.]]))
    disc = HPS.HPSMultidomain(pdo_mod, geom, a, p)
    XX = disc._XX
    XXb = XX[disc.Jx,:]
    XXi = XX[disc.Ji,:]

    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
    Ic = [i for i in range(len(disc.Ji)) if np.abs(XXi[i,0]-xc)<1e-14]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]

    fgb = bc(XXb[Igb,:])
    start_rk = time.time()
    solver = disc.solver_Aii    
    def smatmat(v,I,J,transpose=False):
        if (v.ndim == 1):
            v_tmp = v[:,np.newaxis]
        else:
            v_tmp = v

        if (not transpose):
            result = (solver@(disc.Aix[:,J]@v_tmp))[I]
        else:
            result      = np.zeros(shape=(len(disc.Ji),v.shape[1]))
            result[I,:] = v_tmp
            result      = disc.Aix[:,J].T @ (solver.T@(result))
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
    rkMat_r = assembler.assemble(st_r)
    #data+=assembler.tree.total_bytes()
    rkMat_l = assembler.assemble(st_l)
    #data+=assembler.tree.total_bytes()
    Sl_rk_list += [rkMat_l]
    Sr_rk_list += [rkMat_r]
    stop_rk = time.time()
    trk+=stop_rk-start_rk
    rhs = splinalg.spsolve(disc.Aii,disc.Aix[:,Igb]@fgb)
    rhs = rhs[Ic]
    rhs_list+=[rhs]


print("blocks done")
stop_construct = time.time()
nc = len(Ic)
print("nc = ",nc)
Ntot = (N-1)*nc
print("Ntot = ",Ntot)
print("Nslab = ",(N-1))


rhstot = np.zeros(shape = (Ntot,))
for rhsInd in range(len(rhs_list)):
    rhstot[rhsInd*nc:(rhsInd+1)*nc]=-rhs_list[rhsInd]

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

gInfo = gmres_info()
stol = 1e-5*H*H
uhat,info   = gmres(Linop,rhstot,tol=stol,callback=gInfo,maxiter=25000,restart=25000)
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


uitot = np.zeros(shape=(0,1))
btot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,2))
dofs = 0

for i in range(N):
    xl = i*H
    xr = (i+1)*H
    geom = hpsGeom.BoxGeometry(np.array([[xl,0.],[xr,1.]]))
    disc = HPS.HPSMultidomain(pdo_mod, geom, a, p)
    XX = disc._XX
    XXb = XX[disc.Jx,:]
    XXi = XX[disc.Ji,:]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]
    bvec = np.zeros(shape=(len(disc.Jx),1))
    bvec[Igb,0] = bc(XXb[Igb,:])
    if i>0:
        bvec[Il,0] = uhat[(i-1)*nc:i*nc]
    if i<N-1:
        bvec[Ir,0] = uhat[i*nc:(i+1)*nc]
    dofs+=bvec.shape[0]
    print('linf interface err left = ',np.linalg.norm(bvec[Il,0]-u_exact(XXb[Il,:]),ord=np.inf))
    print('linf interface err right= ',np.linalg.norm(bvec[Ir,0]-u_exact(XXb[Ir,:]),ord=np.inf))
    print('linf interface err gb   = ',np.linalg.norm(bvec[Igb,0]-u_exact(XXb[Igb,:]),ord=np.inf))
    ui = disc.solve_dir_full(bvec)
    uitot=np.append(uitot,ui,axis=0)
    XXfull = disc._XXfull
    ZZ = np.zeros(shape = XXfull.shape)
    ZZ[:,0] = z1(XXfull)
    ZZ[:,1] = z2(XXfull)
    XXtot=np.append(XXtot,ZZ,axis=0)

dofs+=XXtot.shape[0]
print('u shape = ',uitot.shape)
print('XX shape = ',XXtot.shape)
ui_exact = np.sin(kh*XXtot[:,0])
print('u err inf = ',np.linalg.norm(ui_exact-uitot[:,0],ord=np.inf))

resolution = 1000
min_x = np.min(XXtot[:,0])#bnds[0][0]
max_x = np.max(XXtot[:,0])#bnds[1][0]
min_y = np.min(XXtot[:,1])#bnds[0][1]
max_y = np.max(XXtot[:,1])#bnds[1][1]
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]

grid_solution           = griddata(XXtot, uitot[:,0]-ui_exact, (grid_x, grid_y), method='cubic').T

plot_pad=0.1
max_sol = np.max(uitot[:,0])
min_sol = np.min(uitot[:,0])
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
im = ax.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                       min_y-plot_pad,max_y+plot_pad),
                   origin='lower',cmap='jet')
box_geom   = np.array([[0,0],[3.0,1.0]])
hres = 100
ix = np.linspace(box_geom[0,0],box_geom[1,0],hres)
iy = np.linspace(box_geom[0,1],box_geom[1,1],hres)


ext_points = np.zeros(shape=(hres*4,2))

ext_points[:hres,0] = box_geom[0,0]
ext_points[:hres,1] = iy

ext_points[hres:2*hres,1] = box_geom[1,1]
ext_points[hres:2*hres,0] = ix



ext_points[2*hres:3*hres,0] = box_geom[1,0]
ext_points[2*hres:3*hres,1] = np.flip(iy,[0])

ext_points[3*hres:,0] = np.flip(ix,[0])
ext_points[3*hres:,1] = box_geom[0,1]

zz = np.zeros(shape = ext_points.shape)
zz[:,0] = z1(ext_points)
zz[:,1] = z2(ext_points)
poly = Polygon(zz, facecolor='none',edgecolor='none')
ax.add_patch(poly)
im.set_clip_path(poly)

cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=min_sol, vmax=max_sol)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig('annulus.png', transparent=True,format='png',bbox_inches='tight')
plt.show()