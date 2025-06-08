import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
from solver.pde_solver import AbstractPDESolver
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import itertools
import scipy.linalg as splinalg
from scipy.sparse.linalg import gmres
import time
import matplotlib.tri as tri
from scipy import interpolate
from scipy.interpolate import griddata

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

# set-up global geometry
start = time.time()
bnds = [[0.,0.],[1.,1.]]
Om=stdGeom.Box(bnds)
nwaves = 24.673521102434584
kh = (nwaves+0.03)*2*np.pi+1.8
#kapp = 20.
#set up pde


def bfield(xx,kh):
    
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
    return bfield(p,kh)
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)



########################
#   Set up skeleton
########################

#   Explanation:
#   this implementation is meant to allow for the general case of:
#           - unordered interfaces (e.g. hierarchical domain splitting)
#           - nonuniform interface discretizations (e.g. disc. galerkin)
#   (non-uniform here means that the discretization varies from one slab to the next)
#
#   in these cases, the interface connectivity ('skel.C')
#   and the global Idxs ('skel.globIdxs') are deferred to seperate methods.
#   The uniform, standardly ordered case is provided below

# set-up constants
# N             : number of interfaces in skeleton
# [ordx,ordy]   : stencil order in x-and-y direction

N=4
H=Om.bnds[1][0]/(N+1)
skel = skelTon.standardBoxSkeleton(Om,N)
ord=[6,6]
overlapping = True
k=1
aa = 2**k
a=H/aa
# solver wrapper
opts = solverWrap.solverOptions('hps',ord,a)

# a skeleton is a collection of interfaces each carrying their own indices
# referred to as 'global idxs'
# uniformGlobalIdxs is a special method that assumes each will be discretized
# in the same way
# a slablist is a list of overlapping/non-overlapping slabs

skel.setGlobalIdxs(skelTon.computeUniformGlobalIdxs(skel,opts))
slabList = skelTon.buildSlabs(skel,Lapl,opts,overlapping)


########################
#   Test assembly
########################

# known solution
dir = np.random.standard_normal(size=(2,))
dir = dir/np.linalg.norm(dir)
def f(xy):
    if xy.ndim==1:
        return 1.
    else:
        return np.ones(shape = (xy.shape[0],))

#default dense assembler
assembler = mA.denseMatAssembler()
assemblerList = [assembler for slab in slabList]
MultiSlab = MS.multiSlab(slabList,assemblerList)
MultiSlab.constructMats()
rhs = MultiSlab.RHS(f)
Linop       = MultiSlab.getLinOp()
'''

u=np.zeros(shape=(MultiSlab.N,))
n0=0
step=(int)(np.round((ord[0]-2)*(.5/a)))

for i in range(N):
    slabi:MS.Slab=slabList[i]
    if overlapping:
        J=slabi.Ii
    else:
        J=slabi.Ib
    u[range(n0,n0+step)] = slabi.eval_global_func( f,[J[i] for i in slabi.targetIdxs[0]] )
    n0+=step


gInfo = gmres_info()
if petsc_imported == True:
    uhat,info   = gmres(Linop,rhs,rtol=1e-10,callback=gInfo,maxiter=25000,restart=200)
else:
    uhat,info   = gmres(Linop,rhs,tol=1e-10,callback=gInfo,maxiter=25000,restart=200)
res = MultiSlab.apply(uhat)-rhs
stop = time.time()
print("=============SUMMARY==============")
print("H            = ",'%10.3E'%H)
print("ord     = ",ord)
print("L2 rel. res  = ", np.linalg.norm(res)/np.linalg.norm(rhs))
print("L2 rel. err  = ", np.linalg.norm(u-uhat)/np.linalg.norm(u))
print("GMRES iters  = ", gInfo.niter)
print("elapsed time = ",stop-start)
print("==================================")
'''

E = np.identity(Linop.shape[0])
S = Linop@E
uhat = np.linalg.solve(S,rhs)
ind =0
XXtot = np.zeros(shape=(0,2))
uinttot = np.array([])

slab0 = slabList[0]
Ir = [i for i in range(slab0.XX.shape[0]) if slab0.XX[i,0]>1e-8]
ind = 0
for slab in slabList:
    if ind == 0:
        XX = slab.geom.l2g(slab.XX)
    else:
        XX = slab.geom.l2g(slab.XX[Ir,:])
    XXtot = np.append(XXtot,XX,axis=0)
    ind+=1
dir = np.array([1.,1.])
dir=dir/np.linalg.norm(dir)
def btest(p):
    return bfield(p,kh)


XXtotTest = np.unique(XXtot,axis=0)
print("XXtot shape = ",XXtot.shape)
print("XXtotTest shape = ",XXtotTest.shape)


XX = slab0.XX
XXi = slab0.XX[slab0.Ii,:]
XXb = slab0.XX[slab0.Ib,:]

print("Ii = ",slab0.Ii)

print("XX shape = ",XX.shape)
print("XX uniq. shape = ",np.unique(XX,axis=0).shape)

print("XXi shape = ",XXi.shape)
print("XXi uniq. shape = ",np.unique(XXi,axis=0).shape)

print("XXb shape = ",XXb.shape)
print("XXb uniq. shape = ",np.unique(XXb,axis=0).shape)

btot = btest(XXtotTest)
plt.figure(0)
plt.plot(btest(XXtotTest))
resolution = 4000
min_x = 0.
max_x = 1.
min_y = 0.
max_y = 1.
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]

grid_solution     = griddata(XXtotTest, btot, (grid_x, grid_y), method='cubic').T
plot_pad=0.1
max_sol = np.max(btot)
min_sol = np.min(btot)



plt.figure(1)
plt.scatter(XX[:,0],XX[:,1])
plt.scatter(XXi[:,0],XXi[:,1])
plt.scatter(XXb[:,0],XXb[:,1])
plt.legend(['XX','XXi','XXb'])

plt.figure(2)
plt.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                    min_y-plot_pad,max_y+plot_pad),\
                vmin=min_sol, vmax=max_sol,\
                origin='lower',cmap = 'jet')
plt.arrow(.5, .5,.25*dir[0], .25*dir[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.axis('equal')
plt.colorbar()
plt.show()