import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
#import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
#import geometry.skeleton as skelTon
import time
import hps.hps_multidomain as HPS
import hps.geom as hpsGeom
from scipy.sparse        import block_diag
#import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
#from scipy import interpolate
from scipy.interpolate import griddata
from scipy.sparse.linalg   import LinearOperator
from scipy.sparse.linalg import gmres
from solver.solver import stMap
import matAssembly.matAssembler as mA
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


nwaves = 24.623521102434587
#nwaves = 24.673521102434584
#kh = (nwaves+0.03)*2*np.pi+1.8
#kh=157.02
#print("kh = ",kh)
#kapp = 11.1
#nwaves = 24.673521102434584
kh = (nwaves+0.03)*2*np.pi+1.8

def bfield(xx):
    
    mag   = 0.930655
    width = 2500; 
    
    b = np.zeros(shape = (xx.shape[0],))
    
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
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return bfield(p)
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)

def bc(p):
    return np.ones_like(p[:,0])

bnds = [[0.,0.],[1.,1.]]
Om=stdGeom.Box(bnds)
def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-10 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14

H = 1./16.
N = (int)(1./H)
p = 40
a = H/2.
Sl_list = []
Sr_list = []

Sl_rk_list = []
Sr_rk_list = []

rhs_list = []
disc_list = []
trk     = 0
tol = 1e-5
assembler = mA.tolHMatAssembler(tol,p,32)
data = 0

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
opts = solverWrap.solverOptions('hps',[p,p,p],a)
#assembler = mA.denseMatAssembler()#((p+2)*(p+2),50)
OMS = oms.oms(slabs,Lapl,gb,opts,connectivity,if_connectivity)
print("computing Stot & rhstot...")
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler)
print("done")
gInfo = gmres_info()
stol = 1e-6*H*H
uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=1000,restart=1000)
stop_solve = time.time()
res = Stot@uhat-rhstot

print("=============SUMMARY==============")
print("H                        = ",'%10.3E'%H)
print("ord                      = ",p)
print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
print("GMRES iters              = ", gInfo.niter)
#print("constuction time rk.     = ",trk)
#print("par. constuction time rk.= ",trk/(N-1))
#print("solve time               = ",(stop_solve-start_solve))
#print("par. solve time          = ",(stop_solve-start_solve)/(N-1))
#print("data (MB)                = ",data/1e6)
#print("data orig (MB)           = ",(8*Ntot+8*(nc*nc)*2.*(N-1))/1e6)
print("==================================")


uitot = np.zeros(shape=(0,1))
btot = np.zeros(shape=(0,))
XXtot = np.zeros(shape=(0,2))
dofs = 0
global_dofs = OMS.glob_target_dofs
for i in range(len(global_dofs)+1):
    xl = i*H
    xr = (i+1)*H
    print("constructing HPS...")
    geom = hpsGeom.BoxGeometry(np.array([[xl,0.],[xr,1.]]))
    disc = HPS.HPSMultidomain(Lapl, geom, a, p)
    print("done")
    XX = disc._XX
    XXb = XX[disc.Jx,:]
    XXi = XX[disc.Ji,:]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    bvec = np.ones(shape=(len(disc.Jx),1))
    if i>0:
        bvec[Il,0] = uhat[global_dofs[i-1]]
    if i<N-1:
        bvec[Ir,0] = uhat[global_dofs[i]]
    dofs+=bvec.shape[0]
    print("solving dirichlet...")
    ui = disc.solve_dir_full(bvec)
    print("done")
    uitot=np.append(uitot,ui,axis=0)
    XXtot=np.append(XXtot,disc._XXfull,axis=0)
    

dofs+=XXtot.shape[0]
print('u shape = ',uitot.shape)
print('XX shape = ',XXtot.shape)
print('total dofs = ',dofs)

resolution = 1000
min_x = 0.
max_x = 1.
min_y = 0.
max_y = 1.
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]

grid_solution           = griddata(XXtot, uitot[:,0], (grid_x, grid_y), method='cubic').T

plot_pad=0.1
max_sol = np.max(grid_solution[:])
min_sol = np.min(grid_solution[:])
plt.figure(0)
plt.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                    min_y-plot_pad,max_y+plot_pad),\
                                        #vmin=min_sol, vmax=max_sol,\
                origin='lower',cmap = 'jet')
plt.colorbar()
plt.savefig('bfield.png', transparent=True,format='png',bbox_inches='tight')
plt.show()