import numpy as np
from solver.spectralmultidomain.hps import hps_multidomain as hps
import jax.numpy as jnp
import hps.pdo as pdo
import matplotlib.pyplot as plt
import solver.spectralmultidomain.hps.geom as hpsGeom
import multislab.oms as oms
import matAssembly.matAssembler as mA
import solver.solver as solverWrap
import clenshawCurtis as cc

kh = 10.
def c11(p):
    return jnp.ones_like(p[...,0])+jnp.sin(5.*jnp.pi*p[...,0])**2
def c22(p):
    return jnp.ones_like(p[...,0])+jnp.cos(20.*jnp.pi*p[...,0])**2
def c(p):
    return -kh*jnp.ones_like(p[...,0])
Helmholtz = pdo.PDO2d(c11=c11,c22=c22,c=c)

H = 1./8.
a=[H/8,1/8]
p=40



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
#               OMS
######################################


assembler = mA.denseMatAssembler()
opts = solverWrap.solverOptions('hps',[p,p],a)
OMS = oms.oms(slabs,Helmholtz,gb_vec,opts,connectivity,if_connectivity)
print("computing Stot & rhstot...")
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler)
print("done")


E = np.identity(Stot.shape[0])
K = Stot@E




x,w = cc.clenshaw_curtis_compute(p+2)
x = (1.+x)/2.
w*=2*a[1]


#w = np.pi*np.sqrt(x-x**2)/(2*(p+2))
wi = w[1:-1]
#wi[0]+=w[0]
#wi[-1]+=w[-1]
ni = len(wi)

nboxes = (int)(.5/a[1])

W = np.zeros(shape=(nboxes*ni,nboxes*ni))

for i in range(nboxes):
    W[i*ni:(i+1)*ni,:][:,i*ni:(i+1)*ni] = np.diag(wi)

Wtot = np.zeros(shape=((N-1)*W.shape[0],(N-1)*W.shape[0]))
for i in range(N-1):
    Wtot[i*W.shape[0]:(i+1)*W.shape[0],:][:,i*W.shape[0]:(i+1)*W.shape[0]] = W


K = np.sqrt(Wtot)@K@np.linalg.inv(np.sqrt(Wtot))

S12 = K[0:OMS.nc,:][:,OMS.nc:2*OMS.nc]
S21 = K[OMS.nc:2*OMS.nc,:][:,0:OMS.nc]
M=S12@S12.T-S21.T@S21
print("block normality : ",np.linalg.norm(M,ord=2))

v = np.random.standard_normal(size=(S12.shape[0],))
v=v/np.linalg.norm(v)

print("ip normality : ",np.linalg.norm(S12@v,ord=2)-np.linalg.norm(S21.T@v,ord=2))


v=np.random.standard_normal(size=(K.shape[0],))+1j*np.random.standard_normal(size=(K.shape[0],))
v=v/np.linalg.norm(v)
ip = np.conj(v).T@K@v
print(ip)

ip = np.conj(v).T@(K.T@K)@v
print(ip)

e = np.linalg.eigvals(K.astype(np.complex128))
ae = np.abs(e)

[_,s,_] = np.linalg.svd(K)
print("normality error = ",np.linalg.norm(np.sort(ae)-np.sort(s),ord=np.inf))
plt.figure(0)
plt.scatter(np.real(e),np.imag(e))
plt.figure(1)
plt.scatter(np.real(ae),np.imag(ae))
plt.scatter(np.real(s),np.imag(s))
plt.show()