import numpy as np
import multiSlab2D as mS
import stencilDisc as stencil
import matplotlib.pyplot as plt
import spectralDisc as spectral
from simpleoctree import simpletree
import HBSTree as HBS



'''
script for Hmat approx. of 2D stencil S-operator
'''

#############################
#   PART ONE: LR from DENSE
#############################

nSpow = 4
H   =   1./(2**nSpow)
print("H = ",H)
L   =   1.

k   =   6 #depending on sys specs you may have to switch to a sparse version of multiSlab
N   =   2**k+2
h = 1/N

xpts = np.linspace(0,1,N-1)
ypts = np.linspace(0,1,N)
xpts = 2.*H*(xpts)
ypts = L*ypts

XY=np.zeros(shape=(len(xpts)*len(ypts),2))
for j in range(len(ypts)):
    y = ypts[j]
    for i in range(len(xpts)):    
        XY[j+i*len(ypts),:] = [xpts[i],y]

def classify(p):
    x=p[0]
    y=p[1]
    if np.abs(y)<1e-15 or np.abs(y-L)<1e-15:
        return 0
    elif np.abs(x)<1e-15:
        return 1
    elif np.abs(x-2*H)<1e-15:
        return 2
    elif np.abs(x-H)<1e-15:
        return 4
    else:
        return 3
def compute_partition(XY):
    Ibl =   []
    Ibr =   []
    Ii  =   []
    Ic  =   []
    Ib  =   []

    for i in range(XY.shape[0]):
        n=classify(XY[i,:])
        if n<3:
            Ib+=[i]
            if n==1:
                Ibl+=[i]
            if n==2:
                Ibr+=[i]
        else:
            Ii+=[i]
            if n==4:
                Ic+=[i]
    
    return Ibl,Ibr,Ii,Ic,Ib
Ibl,Ibr,Ii,Ic,Ib = compute_partition(XY)
#solve problem
#1:RHS
f=np.zeros(XY.shape[0])
for i in Ibl:
    f[i]=np.sin(np.pi*XY[i,1])
for i in Ibr:
    f[i]=np.sin(2.*np.pi*XY[i,1])


#2:diff ops
Dx = stencil.Diffmat(xpts)
Dy = stencil.Diffmat(ypts)
Ex = np.identity(len(xpts))
Ey = np.identity(len(ypts))
Dxx = Dx.T@Dx
Dyy = Dy.T@Dy
a=np.diag(np.linspace(2.,5.,Dx.shape[0])*np.linspace(2.,5.,Dx.shape[0]))
b=.5#np.diag(np.linspace(4.999,5.,Dx.shape[0])*np.linspace(4.999,5.,Dx.shape[0]))
#bmax = max(np.diag(b))
c=10.
hx=xpts[1]-xpts[0]
hy=ypts[1]-ypts[0]
ex=np.ones(shape=(len(xpts)-1,))/(2.*hx)
ey=np.ones(shape=(len(ypts)-1,))/(2.*hy)
Dx0=np.diag(ex,1)-np.diag(ex,-1)
Dy0=np.diag(ey,1)-np.diag(ey,-1)

Dxy=np.kron(Dx0,Dy0)

Lop = np.kron(Dxx,Ey)+np.kron(Ex,Dyy)+100.*np.kron(Dx,Ey)
Lii = Lop[Ii,:][:,Ii]
Lib = Lop[Ii,:][:,Ib]

#3: discr. system
fb=f[Ib]
u=np.zeros(shape=(XY.shape[0],))
u[Ii]=np.linalg.solve(Lii,-Lib@fb)
u[Ib]=fb

#3.5:res check
utest=np.zeros(shape=(XY.shape[0],))
for i in Ii:
    utest[i]=(np.sin(np.pi*XY[i,1]))*((2*H-XY[i,0])/(2*H))
utest[Ib]=fb
res = np.linalg.norm(utest[Ic]-u[Ic],ord=np.inf)/np.sqrt(N)
print('res = ',res)
print('min(lin-u) = ',np.min(utest[Ic]-u[Ic]))
plt.figure(5)
plt.plot(utest[Ic]-u[Ic])
#4: plot
plt.figure(1)
[X,Y]=np.meshgrid(xpts,ypts)
umat = u.reshape(len(xpts),len(ypts))
ax = plt.axes(projection ='3d')
ax.plot_surface(X,Y,umat.T,rstride=1,cstride=1)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')

plt.figure(2)
umattest = utest.reshape(len(xpts),len(ypts))
ax = plt.axes(projection ='3d')
ax.plot_surface(X,Y,umattest.T,rstride=1,cstride=1)
ax.set(xlabel='X', ylabel='Y', zlabel='Z')

plt.figure(3)
plt.plot(xpts,umat[:,10]-umattest[:,10])
plt.figure(4)
plt.plot(ypts,umat[0,:])
print(umat[0,0])
print(umat[0,len(ypts)-1])


integral = np.zeros(shape=(len(xpts),1))
for i in range(len(ypts)-1):
    integral[i]=np.linalg.norm(umat[i,:],ord=2)
plt.figure(6)
plt.plot(xpts,integral)
plt.show()