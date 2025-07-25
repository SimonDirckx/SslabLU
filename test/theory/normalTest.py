import numpy as np
from solver.spectralmultidomain.hps import hps_multidomain as hps
import jax.numpy as jnp
import hps.pdo as pdo
import matplotlib.pyplot as plt
import solver.spectralmultidomain.hps.geom as hpsGeom
import multislab.oms as oms
import matAssembly.matAssembler as mA
import solver.solver as solverWrap

def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,0])
Lapl = pdo.PDO2d(c11,c22)

H = 1./3.
a=[H/4,1/8]
p=20



jax_avail = True
if jax_avail:
    bnds = [[0.,0.],[1.,1.]]
    box_geom   = jnp.array(bnds)
else:
    bnds = [[0.,0.],[1.,1.]]
    box_geom   = np.array(bnds)


def gb_vec(P):
    # P is (N, 3)
    return (
        (np.abs(P[:, 0] - bnds[0][0]) < 1e-14) |
        (np.abs(P[:, 0] - bnds[1][0]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[0][1]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[1][1]) < 1e-14)
    )

def bc(p):
    return np.ones_like(p[:,0])

N = (int)(1./H)
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



######################################
#       set up Psi1
######################################

Psi1 = hpsGeom.BoxGeometry(jnp.array([[0,0],[2*H,1.]]))

discr1 = hps.HPSMultidomain(Lapl, Psi1, a, p)
XX1 = discr1._XX


######################################
#       set up Psi2
######################################

Psi2 = hpsGeom.BoxGeometry(jnp.array([[H,0],[1.,1.]]))
discr2 = hps.HPSMultidomain(Lapl, Psi2, a, p)
XX2 = discr2._XX



######################################
#               OMS
######################################


assembler = mA.denseMatAssembler()
opts = solverWrap.solverOptions('hps',[p,p],a)
OMS = oms.oms(slabs,Lapl,gb_vec,opts,connectivity,if_connectivity)
print("computing Stot & rhstot...")
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)
print("done")


E = np.identity(Stot.shape[0])
K = Stot@E
print('nrml err (no weighting) = ',np.linalg.norm(K@K.T-K.T@K)/np.linalg.norm(K@K.T))

[e,V] = np.linalg.eig(K)
ae = np.abs(e)

[_,s,_] = np.linalg.svd(K)



print(np.linalg.norm(V-np.conj(V).T,ord=2)/np.linalg.norm(V))
print(np.linalg.norm(np.linalg.inv(V)-np.conj(V).T,ord=2)/np.linalg.norm(V))
print(np.linalg.norm(np.conj(V).T@V-np.identity(V.shape[0]),ord=2))
print('smallest eig err = ',min(ae)-min(s))
plt.figure(1)
plt.scatter(np.real(ae),np.imag(ae))
plt.scatter(np.real(s),np.imag(s))
plt.legend(['ae','s'])
plt.show()

