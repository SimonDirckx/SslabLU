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

#nwaves = 24.623521102434587
#nwaves = 24.673521102434584
#kh = (nwaves+0.03)*2*np.pi+1.8
#kh=157.02
#print("kh = ",kh)
#kapp = 11.1
nwaves = 24.673521102434584
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
Lapl=pdo.PDO2d(c11,c22)#,None,None,None,c)

def bc(p):
    return 0.
def load(p):
    return 2.*np.pi*np.pi*np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])


bnds = [[0.,0.],[1.,1.]]
Om=stdGeom.Box(bnds)
geom = hpsGeom.BoxGeometry(np.array([Om.bnds[0],Om.bnds[1]]))
p=24
a=1./16.
disc = HPS.HPSMultidomain(Lapl, geom, a, p)



XX = disc._XX
XXb = XX[disc.Jx,:]

#################################
#     get interior solution     #
#################################

bcvec = np.array([bc(XXb[i,:]) for i in range(XXb.shape[0])])
ui = disc.solveInterior(bcvec,load)



################################
#    plot interior solution    #
################################
leaves = disc.getLeaves()

XXtot = np.zeros(shape=(0,2))
u_known=np.zeros(shape=(0,))
for leaf in leaves:
    xx = leaf.xxloc[leaf.Jc]
    u_known=np.append(u_known,.5*np.sin(np.pi*xx[:,0])*np.sin(np.pi*xx[:,1]),axis=0)
    XXtot=np.append(XXtot,xx,axis=0)


resolution = 1000
min_x = 0.
max_x = 1.
min_y = 0.
max_y = 1.
grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]

grid_solution           = griddata(XXtot, ui, (grid_x, grid_y), method='cubic').T
grid_solution_known     = griddata(XXtot, u_known, (grid_x, grid_y), method='cubic').T

plot_pad=0.1
max_sol = np.max(grid_solution[:])
min_sol = np.min(grid_solution[:])
plt.figure(0)
plt.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                    min_y-plot_pad,max_y+plot_pad),\
                                        #vmin=min_sol, vmax=max_sol,\
                origin='lower',cmap = 'jet')
plt.colorbar()
plt.figure(1)
plt.imshow(np.abs(grid_solution-grid_solution_known), extent=(min_x-plot_pad,max_x+plot_pad,\
                                    min_y-plot_pad,max_y+plot_pad),\
                                        #vmin=min_sol, vmax=max_sol,\
                origin='lower',cmap = 'jet')
plt.colorbar()
plt.show()