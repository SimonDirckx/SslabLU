import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
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
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay 
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import scipy.special as special
from matplotlib.patches import Polygon
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import solver.HPSInterp3D as interp
import multislab.oms as oms
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
def mesh(slab):
    xpts = np.linspace(slab[0][0],slab[1][0],20)
    ypts = np.linspace(slab[0][1],slab[1][1],20)
    nx = len(xpts)
    ny = len(ypts)
    XY = np.zeros(shape = (nx*ny,2))
    for j in range(ny):
        for i in range(nx):
            XY[j+i*ny] = [xpts[i],ypts[j]]
    return XY


# description: slab1<slab2 assumed!
def join(slab1,slab2,period=None):
    xl1 = slab1[0][0]
    xr1 = slab1[1][0]
    yl1 = slab1[0][1]
    yr1 = slab1[1][1]
    zl1 = slab1[0][2]
    zr1 = slab1[1][2]

    xl2 = slab2[0][0]
    xr2 = slab2[1][0]
    yl2 = slab2[0][1]
    yr2 = slab2[1][1]
    zl2 = slab2[0][2]
    zr2 = slab2[1][2]
    if(np.abs(xr1-xl2)>1e-10):
        if period:
            xl1 -= period
            xr1 -= period
            return join([[xl1,yl1,zl1],[xr1,yr1,zr1]],slab2)
        else:
            ValueError("slab shift did not work (is your period correct?)")
    else:
        totalSlab = [[xl1, yl1,zl1],[xr2,yr2,zr2]]
    return totalSlab
# the final diameter of the domain is 4, so kh = (nwaves/4)*2pi
nwaves = 1.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return -kh*kh*np.ones(shape=(p.shape[0],))
def bfield(p,kh):
    return -kh*kh*np.ones(shape=(p.shape[0],))


const_theta = 1/(2.*np.pi)
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

y3_d3    = lambda zz: np.ones(shape=zz[:,2].shape)



def bc(p):
    r = np.sqrt((z1(p)+3)**2+(z2(p))**2)
    return special.yn(0, kh*r)/4.
def u_exact(p):
    r = np.sqrt((z1(p)+3)**2+(z2(p))**2)
    return special.yn(0, kh*r)/4

# periodic: x=0 and x=1 NOT part of gb
def gb(p):
    return np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14 or np.abs(p[2]-bnds[0][2])<1e-14 or np.abs(p[2]-bnds[1][2])<1e-14


bnds = [[0.,0.,0.],[1.,1.,1.]]
box_geom   = np.array(bnds)
Om=stdGeom.Box(bnds)
param_geom = ParametrizedGeometry3D(box_geom,z1,z2,z3,y1,y2,y3,\
                    y1_d1=y1_d1, y1_d2=y1_d2,\
                    y1_d1d1=y1_d1d1, y1_d2d2=y1_d2d2,\
                    y2_d1=y2_d1, y2_d2=y2_d2, y2_d1d1=y2_d1d1, y2_d2d2=y2_d2d2,\
                    y3_d3=y3_d3)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)


H = 1./4.
N = (int)(1./H)
slabs = []
for n in range(N):
    bnds_n = [[n*H,0.,0.],[(n+1)*H,1.,1.]]
    slabs+=[bnds_n]

connectivity = [[N-1,0]]
for i in range(N-1):
    connectivity+=[[i,i+1]]

print(connectivity)


period = 1.
tol = 1e-5

data = 0
p = 6
a = H/2.
assembler = mA.denseMatAssembler()#rkHMatAssembler(((p+2)//2)*((p+2)//2))


Sl_list = []
Sr_list = []

Sl_rk_list = []
Sr_rk_list = []

rhs_list = []
Ntotcheck = 0

glob_source_dofs=[]
glob_target_dofs=[]

for slabInd in range(len(connectivity)):
    slab = join(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period)
    xl = slab[0][0]
    xr = slab[1][0]
    yl = slab[0][1]
    yr = slab[1][1]
    zl = slab[0][2]
    zr = slab[1][2]
    xc = (xl+xr)/2.
    bnds = [[xl,0.,0.],[xr,1.,1.]]
    box_geom   = np.array(bnds)
    
    print("building HPS...")
    geom = hpsGeom.BoxGeometry(np.array([[xl,yl,zl],[xr,yr,zr]]))
    disc = HPS.HPSMultidomain(pdo_mod, geom, a, p)
    print("...done")
    XX = disc._XX
    XXb = XX[disc.Jx,:]
    XXi = XX[disc.Ji,:]

    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
    Ic = [i for i in range(len(disc.Ji)) if np.abs(XXi[i,0]-xc)<1e-14]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]


    nc = len(Ic)
    
    IFLeft  = connectivity[slabInd][0]
    IFRight = connectivity[slabInd][1]
    startLeft = IFLeft*nc
    startRight = ((IFRight+1)%len(connectivity))*nc
    startCentral = (IFLeft+1)%len(connectivity)*nc
    glob_source_dofs+=[[range(startLeft,startLeft+nc),range(startRight,startRight+nc)]]
    glob_target_dofs+=[range(startCentral,startCentral+nc)]
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
    rhs = splinalg.spsolve(disc.Aii,disc.Aix[:,Igb]@fgb)
    rhs = rhs[Ic]
    rhs_list+=[rhs]
    Ntotcheck+=len(rhs)

print("dofs sources : ",glob_source_dofs)
print("dofs targets : ",glob_target_dofs)

nc = len(Ic)
print("nc = ",nc)
Ntot = N*nc
print("Ntot = ",Ntot)
print("Ntotcheck = ",Ntot)
print("Nslab = ",(N-1))
rhstot = np.zeros(shape = (Ntot,))


for rhsInd in range(len(rhs_list)):
    rhstot[rhsInd*nc:(rhsInd+1)*nc]=-rhs_list[rhsInd]

def smatmat(v,transpose=False):
    if (v.ndim == 1):
        v_tmp = v[:,np.newaxis]
    else:
        v_tmp = v
    result  = v_tmp.copy()
    if (not transpose):
        for i in range(len(glob_target_dofs)):
            result[glob_target_dofs[i]]+=Sl_rk_list[i]@v_tmp[glob_source_dofs[i][0]]
            result[glob_target_dofs[i]]+=Sr_rk_list[i]@v_tmp[glob_source_dofs[i][1]]
    else:
        for i in range(len(glob_target_dofs)):
            result[glob_source_dofs[i][0]]+=Sl_rk_list[i].T@v_tmp[glob_target_dofs[i]]
            result[glob_source_dofs[i][1]]+=Sr_rk_list[i].T@v_tmp[glob_target_dofs[i]]
    if (v.ndim == 1):
        result = result.flatten()
    return result

Linop = LinearOperator(shape=(Ntot,Ntot),\
matvec = smatmat, rmatvec = lambda v: smatmat(v,transpose=True),\
matmat = smatmat, rmatmat = lambda v: smatmat(v,transpose=True))

opts = solverWrap.solverOptions('hps',[p,p,p],a)
OMS = oms.oms(slabs,pdo_mod,gb,opts,connectivity)
Stot0_LO,rhstot0 = OMS.construct_Stot_and_rhstot(bc,assembler)
E = np.identity(Ntot)

Stot0 = Stot0_LO@E
Stot = Linop@E

Stot0T = Stot0_LO.T@E
StotT = Linop.T@E


print("Stot err. = ",np.linalg.norm(Stot-Stot0))
print("Stot T err. = ",np.linalg.norm(StotT-Stot0T))
print("rhs err. = ",np.linalg.norm(rhstot-rhstot0))

gInfo = gmres_info()
stol = 1e-10*H*H
uhat,info   = gmres(Linop,rhstot,tol=stol,callback=gInfo,maxiter=100,restart=100)
stop_solve = time.time()
res = Linop@uhat-rhstot
print('wavelength = ',wavelength)
print('ppw = ',wavelength*nc)

print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhs))
print("GMRES iters              = ", gInfo.niter)
#print("constuction time rk.     = ",trk)
#print("par. constuction time rk.= ",trk/(N-1))
#print("solve time               = ",(stop_solve-start_solve))
#print("par. solve time          = ",(stop_solve-start_solve)/(N-1))
#print("data (MB)                = ",data/1e6)
#print("data orig (MB)           = ",(8*Ntot+8*(nc*nc)*2.*(N-1))/1e6)
print("==================================")

uitot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,3))
dofs = 0

for i in range(N):
    print("i = ",i)
    print("H = ",H)
    xl = i*H
    xr = (i+1)*H
    print("xl = ",xl)
    print("xr = ",xr)
    geom = hpsGeom.BoxGeometry(np.array([[xl,0.,0.],[xr,1.,1.]]))
    disc = HPS.HPSMultidomain(pdo_mod, geom, a, p)
    XX = disc._XX
    XXb = XX[disc.Jx,:]
    XXi = XX[disc.Ji,:]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]
    bvec = np.zeros(shape=(len(disc.Jx),1))
    bvec[Igb,0] = bc(XXb[Igb,:])
    bvec[Il,0] = uhat[i*nc:(i+1)*nc]
    start = ((i+1)%N)*nc
    bvec[Ir,0] = uhat[start:start+nc]
    dofs+=bvec.shape[0]
    ui = disc.solve_dir_full(bvec)
    
    resx = 50
    resy = 30
    x_eval = np.linspace(disc._box_geom[0][0],disc._box_geom[1][0],resx)
    y_eval = np.linspace(disc._box_geom[0][1],disc._box_geom[1][1],resy)
    #eval at z=.6

    XY = np.zeros(shape=(resx*resy,3))
    XY[:,0] = np.kron(x_eval,np.ones(shape=y_eval.shape))
    XY[:,1] = np.kron(np.ones(shape=x_eval.shape),y_eval)
    XY[:,2] = .6*np.ones(shape = (resx*resy,))
    u_approx,XYlist = interp.interpHPS(disc,ui[:,0],XY)
    uitot=np.append(uitot,u_approx,axis=0)
    u_exact_vec = np.zeros(shape=(0,1))
    for i in range(len(XYlist)):
        ue = u_exact(XYlist[i])
        u_exact_vec= np.append(u_exact_vec,ue)
        XXtot=np.append(XXtot,XYlist[i],axis=0)
    errInf = np.linalg.norm(u_exact_vec-u_approx,ord=np.inf)
    print('errInf = ',errInf)
XXtot,I=np.unique(XXtot,axis=0,return_index=True)
ui_exact = u_exact(XXtot)
uitot=uitot[I]
print('u err inf = ',np.linalg.norm(ui_exact-uitot,ord=np.inf))
ZZ = np.zeros(shape = XXtot.shape)
ZZ[:,0] = z1(XXtot)
ZZ[:,1] = z2(XXtot)
ZZ[:,2] = z3(XXtot)
tri = Delaunay(XXtot[:,0:2])
plt.figure(0)
plt.tripcolor(ZZ[:,0],ZZ[:,1],uitot,triangles = tri.simplices.copy(),cmap='jet',shading='gouraud',antialiased=False,linewidth=0)
plt.colorbar()
plt.axis('equal')
plt.show()