import sys
sys.path.append('/home/simond/SslabLU')
import numpy as np
import solver.spectralmultidomain.hps.pdo as pdo
import scipy
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import time
import hps.hps_multidomain as HPS
import hps.geom as hpsGeom
import scipy.sparse.linalg as splinalg
from scipy.interpolate import griddata
from scipy.sparse.linalg   import LinearOperator
from scipy.sparse.linalg import gmres
from solver.solver import stMap
import matAssembly.matAssembler as mA
#import jax.numpy as np
from packaging.version import Version
import solver.HPSInterp2D as interp
import jax.numpy as jnp
from scipy.spatial import Delaunay
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


nwaves = 24.623521102434587
#nwaves = 24.673521102434584
#kh = (nwaves+0.03)*2*np.pi+1.8
#kh=157.02
#print("kh = ",kh)
#kapp = 11.1
#nwaves = 24.673521102434584
kh = (nwaves+0.03)*2*np.pi+1.8
jax_avail=True
if jax_avail:
    def bfield(xx):
        
        mag   = 0.930655
        width = 2500; 
        
        b = jnp.zeros_like(xx[...,0])
        
        dist = 0.04
        x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
        y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
        
        # box of points [x0,x1] x [y0,y1]
        for x in jnp.arange(x0,x1,dist):
            for y in jnp.arange(y0,y1,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)

        # box of points [x0,x1] x [y0,y2]
        for x in jnp.arange(x2,x3,dist):
            for y in jnp.arange(y0,y2-0.5*dist,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)
                
        # box of points [x0,x3] x [y2,y3]
        for x in jnp.arange(x0,x3,dist):
            for y in jnp.arange(y2,y3,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)    
        
        kh_fun = -kh**2 * (1 - b)
        return kh_fun


    def c11(p):
        return jnp.ones_like(p[...,0])
    def c22(p):
        return jnp.ones_like(p[...,0])
    def c(p):
        return bfield(p)
else:
    def bfield(xx):
        
        mag   = 0.930655
        width = 2500; 
        
        b = np.zeros_like(xx[:,0])
        
        dist = 0.04
        x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
        y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
        
        # box of points [x0,x1] x [y0,y1]
        for x in np.arange(x0,x1,dist):
            for y in np.arange(y0,y1,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)

        # box of points [x0,x1] x [y0,y2]
        for x in np.arange(x2,x3,dist):
            for y in np.arange(y0,y2-0.5*dist,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)
                
        # box of points [x0,x3] x [y2,y3]
        for x in np.arange(x0,x3,dist):
            for y in np.arange(y2,y3,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)    
        
        kh_fun = -kh**2 * (1 - b)
        return kh_fun


    def c11(p):
        return np.ones_like(p[:,0])
    def c22(p):
        return np.ones_like(p[:,0])
    def c(p):
        return bfield(p)
    
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)

def bc(p):
    return 1.

bnds = [[0.,0.],[1.,1.]]
Om=stdGeom.Box(bnds)
def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-10 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14

H = 1./8.
N = (int)(1./H)
p = 20
a = [H/2.,1/8]
Sl_list = []
Sr_list = []

Sl_rk_list = []
Sr_rk_list = []

rhs_list = []
trk     = 0
tol = 1e-5
assembler = mA.denseMatAssembler()#(tol*H*H*.5,32)
data = 0
for slabInd in range(N-1):
    xl = slabInd*H
    xr = (slabInd+2)*H
    xc = (slabInd+1)*H
    if jax_avail:
        geom = hpsGeom.BoxGeometry(jnp.array([[xl,0.],[xr,1.]]))
    else:
        geom = hpsGeom.BoxGeometry(np.array([[xl,0.],[xr,1.]]))
    disc = HPS.HPSMultidomain(Lapl, geom, a, p)
    XX = disc._XX
    XXb = XX[disc.Jx,:]
    XXi = XX[disc.Ji,:]

    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
    Ic = [i for i in range(len(disc.Ji)) if np.abs(XXi[i,0]-xc)<1e-14]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]

    fgb = np.array([bc(XXb[i,:]) for i in Igb])
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
    del disc
    print("loop done")


print("blocks done")
stop_construct = time.time()
nc = len(Ic)
print("nc = ",nc)
Ntot = (N-1)*nc
print("Ntot = ",nc)
Stot = np.identity(Ntot)
Stotr = np.identity(Ntot)
rhstot = np.zeros(shape = (Ntot,))


for rhsInd in range(len(rhs_list)):
    rhstot[rhsInd*nc:(rhsInd+1)*nc]=-rhs_list[rhsInd]

start_solve = time.time()
def smatmat(v,transpose=False):
    if (v.ndim == 1):
        v_tmp = v[:,np.newaxis]
    else:
        v_tmp = v
    result  = v_tmp.copy().astype('float64')
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
#E = np.identity(Linop.shape[0])
#Stot = Linop@E
#plt.spy(Stot)
#plt.show()
gInfo = gmres_info()
stol = 1e-6*H*H
if Version(scipy.__version__)>=Version("1.14"):
    uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=1000,restart=1000)
else:
    uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=1000,restart=1000)
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


uitot = np.zeros(shape=(0,))
btot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,2))
dofs = 0

for i in range(N-1):
    
    xl = i*H
    xr = (i+1)*H
    geom = hpsGeom.BoxGeometry(jnp.array([[xl,0],[xr,1]]))
    disc = HPS.HPSMultidomain(Lapl, geom, a, p)
    
    XXb = disc._XX[disc.Jx,:]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10 and not gb(XXb[i,:])]
    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10 and not gb(XXb[i,:])]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]
    bvec = np.ones(shape=(len(disc.Jx),1))
    bvec[Igb,0] = bc(XXb[Igb,:])
    if i>0:
        bvec[Il,0] = uhat[(i-1)*nc:i*nc]
    if i<N-1:   
        bvec[Ir,0] = uhat[i*nc:(i+1)*nc]
    
    print("solving local dirichlet...")
    ui = disc.solve_dir_full(bvec)
    print("done")
    print("XXfull shape = ",disc._XXfull.shape)
    XXtot = np.append(XXtot,np.array(disc._XXfull),axis=0)
    print("ui shape = ",ui.shape)
    print("XXfull shape = ",disc._XXfull.shape)
    uitot=np.append(uitot,ui)
print("XXtot shape = ",XXtot.shape)
print("uitot shape = ",uitot.shape)
XXtot,I = np.unique(XXtot,axis=0,return_index=True)
uitot=uitot[I]
print(XXtot.shape)
print(uitot.shape)

tri = Delaunay(XXtot)
plt.figure(3)
plt.tripcolor(XXtot[:,0],XXtot[:,1],uitot,triangles = tri.simplices.copy(),cmap='jet',shading='gouraud',antialiased=False,linewidth=0)
plt.colorbar()
plt.axis('equal')
bfieldstr = 'bfield_cheb_grid_tripcolor_'+str(p)+'_'+str(a)+'.png'
plt.savefig(bfieldstr, transparent=True,format='png',bbox_inches='tight')
plt.show()
    
    



