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
k=5
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
Ir = []
Ib = []



S=T.copy()


for i in range(len(I)//nc):
    S[i*nc:(i+1)*nc,:] = np.linalg.solve(T[i*nc:(i+1)*nc,:][:,i*nc:(i+1)*nc],T[i*nc:(i+1)*nc,:])
    if i%2==0:
        Ir+=[j for j in range(i*nc,(i+1)*nc)]
    else:
        Ib+=[j for j in range(i*nc,(i+1)*nc)]

nr = len(Ir)
nb = len(Ib)
Trr = T[Ir,:][:,Ir]
Tbb = T[Ib,:][:,Ib]
Trb = T[Ir,:][:,Ib]
Tbr = T[Ib,:][:,Ir]


Trbbr = np.zeros(shape = T.shape)
Trbbr[:nr,:][:,nr:nr+nb] = Trb
Trbbr[nr:nr+nb,:][:,:nr] = Tbr
plt.figure(1)
plt.spy(Trbbr)
plt.show()


[e,V] = np.linalg.eig(T)
eMat = np.zeros(shape=(len(e),2))
eMat[:,0] = np.real(e)
eMat[:,1] = np.imag(e)

filename = "spectrumT_"+str(h)+".csv"
header = "real,imag"
with open(filename,'w') as f:
    f.write(header+'\n')
    np.savetxt(f,eMat,fmt='%.16e',delimiter=',')