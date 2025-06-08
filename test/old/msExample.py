import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import time
from scipy.sparse.linalg import gmres
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

#####################################
#       set up PDE
#####################################
nwaves = 24.623521102434587
#nwaves = 24.673521102434584
kh = (nwaves+0.03)*2*np.pi+1.8
#kh=157.02
print("kh = ",kh)
kapp = 11.1

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
def c(xx):
    return bfield(xx)
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)

#####################################
#       set up geom & skel
#####################################
bnds = [[0.,0.],[1.,1.]]
Om=stdGeom.Box(bnds)
k=0
N=2**k
H=Om.bnds[1][0]/(N+1)
skel = skelTon.standardBoxSkeleton(Om,N)

#####################################
#       set up solver
#####################################
ny = 1024+2
hy = 1./ny
hx = hy
overlapping = True
if overlapping:
    nx = (int)(np.ceil(2*H/hx))
    nx = nx-((nx+1)%2) #make sure this is even
    hx = 2*H/nx
else:
    nx = H/hx
    nx = (int)(np.ceil(nx))
    hx = H/nx
ord=[nx,ny]

ord=[18,18]
overlapping = True
k=4
aa = 2**k
a=H/aa

opts = solverWrap.solverOptions('hps',ord,a)

#uniformly spaced slabs
skel.setGlobalIdxs(skelTon.computeUniformGlobalIdxs(skel,opts))
slabList = skelTon.buildSlabs(skel,Lapl,opts,overlapping)
#specify how blocks should be computed ('dense'/'HBS')
assembler = mA.denseMatAssembler()
assemblerList = [assembler for slab in slabList]
MultiSlab = MS.multiSlab(slabList,assemblerList)

start = time.time()
MultiSlab.constructMats()
def bdry_func(p):
     return 1.
def load_func(p):
     return 0

rhs = MultiSlab.RHS(bdry_func,load_func)

Linop       = MultiSlab.getLinOp()
E = np.identity(Linop.shape[0])
Stot = Linop@E
uhat = np.linalg.solve(Stot,rhs)
res = MultiSlab.apply(uhat)-rhs
stop = time.time()
print(len(slabList))
slab0 = slabList[0]
discHPS = slab0.solverWrap.solver
XXb = slab0.geom.l2g(slab0.XX[slab0.Ib,:])
bcvec = np.array([bdry_func(XXb[i,:]) for i in range(XXb.shape[0])])
#print(slab0.sourceIdxs[0])
#bcvec[slab0.sourceIdxs[0]]=uhat[0:len(slab0.sourceIdxs[0])]

ui = discHPS.solveInterior(bcvec)

leaves = discHPS.getLeaves()

XXtot = np.zeros(shape=(0,2))
for leaf in leaves:
    xx = leaf.xxloc[leaf.Jc]
    XXtot=np.append(XXtot,xx,axis=0)


resolution = 1000
min_x = -.5
max_x = .5
min_y = 0.
max_y = 1.
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]

grid_solution           = griddata(XXtot, ui, (grid_x, grid_y), method='cubic').T

plot_pad=0.1
max_sol = np.max(grid_solution[:])
min_sol = np.min(grid_solution[:])
plt.figure(0)
plt.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                    min_y-plot_pad,max_y+plot_pad),\
                                        #vmin=min_sol, vmax=max_sol,\
                origin='lower',cmap = 'jet')

plt.figure(1)
plt.scatter(XXb[:,0],XXb[:,1])
#plt.scatter(XXb[slab0.sourceIdxs[0],0],XXb[slab0.sourceIdxs[0],1])

plt.figure(2)
plt.scatter(XXtot[:,0],XXtot[:,1])
#plt.scatter(XXb[slab0.sourceIdxs[0],0],XXb[slab0.sourceIdxs[0],1])
plt.show()

plt.show()

'''
ind=0
XXtot = np.zeros(shape=(0,2))
uinttot = np.array([])
slab0 = slabList[0]
XX0glob,I = np.unique(slab0.geom.l2g(slab0.XX),axis=0,return_index=True)
XX0 = slab0.XX[slab0.Ii,:]
XX0 = np.append(XX0,slab0.XX[slab0.Ib,:],axis=0)
Ir = [i for i in range(XX0.shape[0]) if XX0[i,0]>1e-10]
XX0r = XX0[Ir,:]
for slab in slabList:
    if ind == 0:
        XXloc = XX0
    else:
        XXloc = XX0r
    b = slab.evalGlobalFuncAtLocPts(bfield,XXloc)
    XXtot = np.append(XXtot,slab.geom.l2g(XXloc),axis=0)
    uinttot = np.append(uinttot,b,axis=0)
    ind = ind+1
print("XXtot done")
print("XXtot shape = ",XXtot.shape)
print("XXtot uniq shape = ",np.unique(XXtot,axis=0).shape)
plt.show()
resolution = 2000
min_x = 0.
max_x = 1.
min_y = 0.
max_y = 1.
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]

grid_solution     = griddata(XXtot, uinttot, (grid_x, grid_y), method='cubic').T
plot_pad=0.1
max_sol = np.max(grid_solution[:])
min_sol = np.min(grid_solution[:])
plt.figure(1)
plt.scatter(XXtot[:,0],XXtot[:,1])
plt.figure(2)
plt.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                    min_y-plot_pad,max_y+plot_pad),\
                                        #vmin=min_sol, vmax=max_sol,\
                origin='lower',cmap = 'jet')
plt.colorbar()
plt.show()
'''