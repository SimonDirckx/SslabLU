import numpy as np
import solver.HPSInterp as interp
import jax.numpy as jnp
import solver.spectralmultidomain.hps.geom as hpsGeom
from solver.spectralmultidomain.hps import hps_multidomain as hps
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

geom = hpsGeom.BoxGeometry(jnp.array([[0,0,0],[1,1,1]]))
a=[.25,.25,.25]
p=2
solver = hps.HPSMultidomain(PDE, geom,a, p)

npan_dim = solver.npan_dim
boxes = interp.construct_boxes_3d(npan_dim,geom)
for box in boxes:
    print(box)


XX = solver._XXfull
xpts = .25+cheb.cheb_3d([.25,.25,.25],p+2)[0].T
J = interp.idxs_3d(XX,boxes[0])
XX0 = XX[J,:]
XX0 = jnp.unique(XX0,axis=0)
print("cheb err = ",np.linalg.norm(XX0-xpts))
