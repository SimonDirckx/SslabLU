import numpy as np
import solver.HPSInterp as interp
import jax.numpy as jnp
import solver.spectralmultidomain.hps.geom as hpsGeom
from solver.spectralmultidomain.hps import hps_multidomain as hps
import solver.solver as solverWrap
import hps.pdo as pdo
import matplotlib.pyplot as plt
import hps.cheb_utils as cheb

def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,1])
def c33(p):
    return jnp.ones_like(p[...,2])

PDE = pdo.PDO3d(c11=c11,c22=c22,c33=c33)
geom = jnp.array([[0,0,0],[1,1,1]])
a=[.125,.125,.125]
p=12


opts = solverWrap.solverOptions('hps',[p,p,p],a)
solver = solverWrap.solverWrapper(opts)
solver.construct(geom,PDE)
XX = solver.XXfull


kh = 15
f=np.sin(XX[:,0])+np.cos(kh*np.pi*XX[:,0]*XX[:,1])

pts = np.random.uniform(size=(10,3))
ghat = solver.interp(pts,f)

g = np.sin(pts[:,0])+np.cos(kh*np.pi*pts[:,0]*pts[:,1])
print(np.linalg.norm(g-ghat,ord=np.inf))
