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
k=8
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
H=1/8
h=1./ord
I = np.where(np.abs(XXi[:,0]/H-np.round(XXi[:,0]/H))<h/2)[0]
Ic = np.where(np.abs(XXi[:,0]/H-np.round(XXi[:,0]/H))>=h/2)[0]


Aii = disc.Aii
T = Aii[I,:][:,I]-Aii[I,:][:,Ic]@(splinalg.spsolve(Aii[Ic,:][:,Ic],Aii[Ic,:][:,I]))
T = T.todense()
nc = (ord-2)
Ir=[]
Ib=[]
for i in range((int)(1/H)-1):
    if i%2==0:
        Ir+=[j for j in range(nc*i,nc*(i+1))]
    else:
        Ib+=[j for j in range(nc*i,nc*(i+1))]
Trr = T[Ir,:][:,Ir]
Trb = T[Ir,:][:,Ib]
Tbr = T[Ib,:][:,Ir]
Tbb = T[Ib,:][:,Ib]

Tperm = np.zeros(shape=T.shape)
print("Tperm shape = ",Tperm.shape)
print("len(Ir) = ",len(Ir))
Tperm[0:len(Ir),:][:,0:len(Ir)] = Trr
Tperm[len(Ir):len(Ir)+len(Ib),:][:,len(Ir):len(Ir)+len(Ib)] = Tbb
Tperm[0:len(Ir),:][:,len(Ir):len(Ir)+len(Ib)] = Trb
Tperm[len(Ir):len(Ir)+len(Ib),:][:,0:len(Ir)] = Tbr

#now compute the inequality

[e,V] = np.linalg.eig(Tperm)
imin = np.argmin(np.abs(e))
vmin = V[:,imin]

vr = vmin[0:len(Ir)]
vb = vmin[len(Ir):len(Ir)+len(Ib)]
ip1 = vr.T@Trr@vr+vb.T@Tbb@vb
ip2 = vmin.T@Tperm@vmin
print("ip1/ip2 = ",ip1/ip2)
print("H2 = ",1/(H*H))
print("===========================")
[er,Vr] = np.linalg.eig(Trr)
[eb,Vb] = np.linalg.eig(Tbb)
imaxr = np.argmax(np.abs(er))
imaxb = np.argmax(np.abs(eb))
vmaxr = Vr[:,imaxr]
vmaxb = Vb[:,imaxb]

v=np.append(vmaxr,vmaxb,axis=0)
ip1 = vmaxr.T@Trr@vmaxr+vmaxb.T@Tbb@vmaxb
ip2 = v.T@Tperm@v
print("ip1/ip2 = ",ip1/ip2)
print("H2 = ",1/(H*H))