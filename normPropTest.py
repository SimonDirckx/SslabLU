import numpy as np
import multiSlab2D as mS
import matplotlib.pyplot as plt
import stencilDisc as stencil
from simpleoctree import simpletree
import HBSTree as HBS
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg



'''
script for Hmat approx. of 2D stencil S-operator
'''

#############################
#   PART ONE: LR from DENSE
#############################

nSpow = 3
H   =   1./(2**nSpow)
L   =   1.

k   =   6 #depending on sys specs you may have to switch to a sparse version of multiSlab

ordy   =    2**k+2
ordx   =    int(np.ceil(.5*ordy*H)) #x dir is resolved more finely
h = 1/ordy
print("H    = ",H)
print("h    = ",h)
print("ordx = ",ordx)
print("ordy = ",ordy)


ndSlabs = 0
i=1
while i*H<1:
    ndSlabs+=1
    i+=1
def f_factory(i):
    def f(x,y):
        return x+i*H,y  # i is now a *local* variable of f_factory and can't ever change
    return f
l2glist=[]
for ind in range(ndSlabs):
    l2glist+=[f_factory(ind)]

part = mS.partition(L,H)
disc = mS.discretization(ordx,ordy,'stencil')
doubleSlabList=[]
for ind in range(ndSlabs):
    dS = mS.dSlab(disc,part,l2glist[ind],H,L)
    doubleSlabList+=[dS]
doubleSlabList[0].setAsEdge(1)
doubleSlabList[ndSlabs-1].setAsEdge(2)
totalDofs = 0
for dS in doubleSlabList:
    totalDofs+=dS.ndofs
    dS.computeL()

MS=mS.multiSlab(doubleSlabList)


Stot = MS.form_total_S()
SS = Stot.T@Stot
print("cond = ",np.linalg.cond(Stot))

SM = np.identity(Stot.shape[0])-Stot

[e,V]=np.linalg.eig(SS)
imin = np.argmin(np.abs(e))
imax = np.argmax(np.abs(e))
plt.figure(1)
plt.scatter(np.real(e),np.imag(e),marker='x')

plt.figure(2)
plt.plot(V[:,imin])
plt.plot(V[:,imax])
plt.legend(['vmin','vmax'])

vtest = np.zeros(shape=(V.shape[0],))
sgn = 1
for i in range(ndSlabs):
    vtest[i*(ordy-2):(i+1)*(ordy-2)] = sgn*V[i*(ordy-2):(i+1)*(ordy-2),imin]
    sgn*=-1


lamv = SS@vtest
c = lamv.T@vtest
print("c=",c)
mn = np.linalg.norm(vtest-lamv/c,ord=np.inf)
print("inf err = ",mn)
l=np.linspace(c-1,c+1,1000)
for f in l:
    if mn>np.linalg.norm(vtest-lamv/f,ord=np.inf):
        c=f
        mn=np.linalg.norm(vtest-lamv/f,ord=np.inf)
print("inf err = ",mn)
plt.figure(3)
plt.plot(vtest)
plt.plot(lamv/c)
plt.legend(['vtest','lamv'])


plt.figure(4)
plt.plot(vtest-lamv/c)
plt.legend(['vtest','lamv'])

#########################################
#   Compare to Eigenfunctions of L
#########################################
ordx = (ndSlabs+1)*(ordx//2)+1
print("ordx = ",ordx)
disc = mS.discretization(ordx,ordy,'stencil')
disc.discretize(.5,1.)
XY=disc.XY
N=XY.shape[0]

f=np.zeros(shape=(N,))
dnf=np.zeros(shape=(N,))
for i in range(N):
    x=XY[i,0]
    y=XY[i,1]
    f[i] = np.sin(np.pi*x)*np.sinh(np.pi*y)
    if np.abs(x)<1e-14:
        dnf[i]=-np.pi*np.cos(np.pi*x)*np.sinh(np.pi*y)
    if np.abs(x-1)<1e-14:
        dnf[i]=np.pi*np.cos(np.pi*x)*np.sinh(np.pi*y)
    if np.abs(y)<1e-14:
        dnf[i]=-np.pi*np.sin(np.pi*x)*np.cosh(np.pi*y)
    if np.abs(y-1)<1e-14:
        dnf[i]=np.pi*np.sin(np.pi*x)*np.cosh(np.pi*y)


Dx  = disc.get_Dx()
Dy  = disc.get_Dy()
Dxx = disc.get_Dxx()
Dyy = disc.get_Dyy()
Ex  = disc.get_Ex()
Ey  = disc.get_Ey()

hx=disc.xpts[1]-disc.xpts[0]
hy=disc.ypts[1]-disc.ypts[0]
ex=np.ones(shape=(len(disc.xpts)-1,))/(2.*hx)
ey=np.ones(shape=(len(disc.ypts)-1,))/(2.*hy)
Dx0=np.diag(ex,1)-np.diag(ex,-1)
Dy0=np.diag(ey,1)-np.diag(ey,-1)

Dxy=np.kron(Dx0,Dy0)


Islabs=[]
Ii=[]
Ib=[]
for i in range(XY.shape[0]):
    if(np.abs(XY[i,1])<1e-10 or np.abs(XY[i,1]-1)<1e-10 or np.abs(XY[i,0])<1e-10 or np.abs(XY[i,0]-1)<1e-10):
        Ib+=[i]
    else:
        Ii+=[i]
for i in range(len(Ii)):
    if(np.abs(np.round(XY[Ii[i],0]/H)-XY[Ii[i],0]/H)<1e-10):
            Islabs+=[i]
print("#Islabs = ",len(Islabs))
L = np.kron(Dxx,Ey)+np.kron(Ex,Dyy)
Lii=L[Ii,:][:,Ii]
Lib=L[Ii,:][:,Ib]
Lbi=L[Ib,:][:,Ii]
Lbb=L[Ib,:][:,Ib]

fi      = f[Ii]
fb      = f[Ib]
m=len(Ii)
print('sol. err. = ',np.linalg.norm(Lii@fi+Lib@fb)/(np.sqrt(m)*np.linalg.norm(fi)))
dnfb    = dnf[Ib]
Sch     = Lbb-Lbi@np.linalg.solve(Lii,Lib)
utot    = L@f
ub      = utot[Ib]
utest   = Sch@fb
utest=utest/np.linalg.norm(utest)
ub=ub/np.linalg.norm(ub)
dnfb=dnfb/np.linalg.norm(dnfb)
plt.figure(6)
plt.plot(ub.T)
plt.plot(utest.T)
plt.plot(dnfb)
#plt.plot(dnfb)#
plt.legend(["u","utest","d/dn"])
plt.show()

