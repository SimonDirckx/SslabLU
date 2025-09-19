import geometry.geom_2D.square as square
import solver.stencil.stencilSolver as stencil
from solver.spectralmultidomain.hps import pdo
import numpy as np
import scipy.sparse.linalg as splinalg

import matplotlib.pyplot as plt

def c11(p):
    return np.ones_like(p[:,0])
def c22(p):
    return np.ones_like(p[:,0])

Lapl=pdo.PDO2d(c11,c22)
k=6
ord = (2**k)+1
disc = stencil.stencilSolver(Lapl,np.array([[0,0],[1,1]]),[ord,ord])


#test correctness of stencil

def bc(p):
    return np.sin(p[:,0])*np.sinh(p[:,1])

XXi = disc.XXi
XXb = disc.XXb

rhs = -disc.Aix@bc(XXb)
sol = splinalg.spsolve(disc.Aii,rhs)
u = bc(XXi)
print("err = ",np.linalg.norm(u-sol)/np.linalg.norm(u))


#select the interfaces
H=1/16
h=1./ord
I = np.where(np.abs(XXi[:,0]/H-np.round(XXi[:,0]/H))<h/2)[0]
Ic = np.where(np.abs(XXi[:,0]/H-np.round(XXi[:,0]/H))>=h/2)[0]


Aii = disc.Aii
T = Aii[I,:][:,I]-Aii[I,:][:,Ic]@(splinalg.spsolve(Aii[Ic,:][:,Ic],Aii[Ic,:][:,I])).todense()

nc=ord-2

S=T.copy()


for i in range(len(I)//nc):
    S[i*nc:(i+1)*nc,:] = np.linalg.solve(T[i*nc:(i+1)*nc,:][:,i*nc:(i+1)*nc],T[i*nc:(i+1)*nc,:])
plt.figure(1)
plt.spy(S,1e-8)



[e,V] = np.linalg.eig(T)
[eS,VS] = np.linalg.eig(S)

print("mne = ",np.min(e))
print("mxe = ",np.max(e))
print("condest = ",np.max(e)/np.min(e))
print("=========================")
print("mneS = ",np.min(eS))
print("mxeS = ",np.max(eS))
print("condestS = ",np.max(eS)/np.min(eS))


plt.figure(2)
plt.scatter(np.real(e),np.imag(e))

plt.figure(3)
plt.scatter(np.real(eS),np.imag(eS))

plt.show()