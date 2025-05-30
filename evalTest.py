import numpy as np
import matplotlib.pyplot as plt
import hps.hps_multidomain as HPS
import hps.geom as hpsGeom
import pdo.pdo as pdo
import solver.HPSInterp2D as HPSInterp2D
import time



# set-up of functions

def f(xpts,ypts):
    X,Y=np.meshgrid(xpts,ypts)
    return np.sin(20.*np.pi*X**2)+np.sin(25.*np.pi*Y)

def fvec(xypts):
    return np.sin(20.*np.pi*xypts[:,0]**2)+np.sin(25.*np.pi*xypts[:,1])



def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22)

geom = hpsGeom.BoxGeometry(np.array([[0.,0.],[1.,1.]]))

prange = range(4,21)
arange = [.5**i for i in range(2,6)]


errInf=np.zeros(shape=(len(arange),len(prange)))
timings=np.zeros(shape=(len(arange),len(prange)))

for p in prange:
    ctra = 0
    for a in arange:
        disc = HPS.HPSMultidomain(Lapl, geom, a, p)
        tic = time.time()
        x_eval = np.linspace(disc._box_geom[0][0],disc._box_geom[1][0],100)
        y_eval = np.linspace(disc._box_geom[0][1],disc._box_geom[1][1],100)
        XY = np.zeros(shape=(100*100,2))
        XY[:,0] = np.kron(x_eval,np.ones(shape=x_eval.shape))
        XY[:,1] = np.kron(np.ones(shape=y_eval.shape),y_eval)
        vals = fvec(disc._XXfull)
        F_approx,XYlist = HPSInterp2D.interpHPS(disc,vals,XY)
        F_exact = np.zeros(shape=(0,1))
        ndofs = (p+2)*(p+2)
        for i in range(len(XYlist)):
            F_exact= np.append(F_exact,fvec(XYlist[i]))
        errInf[ctra,p-prange.start] = np.linalg.norm(F_exact-F_approx,ord=np.inf)
        toc = time.time()
        timings[ctra,p-prange.start] = toc-tic
        ctra+=1
        print(p,"//",a)
        



PP,AA = np.meshgrid(arange,prange)

print(errInf)
plt.figure(0)
cf = plt.contourf(PP,AA,np.log10(errInf.T))
contour_levels = cf.levels
plt.xlabel('a')
plt.ylabel('p')
plt.colorbar()

plt.figure(1)
plt.contourf(PP,AA,-.001*np.divide(np.log2(errInf.T),timings.T))
plt.colorbar()
plt.contour(PP,AA, np.log10(errInf.T), levels=contour_levels, colors='black',linestyles='dashed')
plt.xlabel('a')
plt.ylabel('p')
plt.title('digits per ms')
plt.show()

