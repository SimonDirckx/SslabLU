import numpy as np
import solver.stencil.stencilSolver as stencil
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
# 1D tests
kapp = 20.51
xmax = 1.
nx = (int)(np.ceil(40*kapp*xmax/np.pi))
nx = 2*(nx//2) + 1
xpts = np.linspace(0.,xmax,nx)

def f(x):
    return np.sin(kapp*x)
def df(x):
    return kapp*np.cos(kapp*x)
def ddf(x):
    return -kapp*kapp*np.sin(kapp*x)

fx = np.array([f(x) for x in xpts])
dfx = np.array([df(x) for x in xpts])
ddfx = np.array([ddf(x) for x in xpts])

Dx = stencil.stencilD(xpts)
Iix = range(1,nx-1)
Dxx = stencil.stencilD2(xpts)
Ex = np.identity(nx)


print("D err. = ",np.linalg.norm(dfx[Iix]-(Dx@fx)[Iix])/np.linalg.norm(dfx))
print("D2 err. = ",np.linalg.norm(ddfx[Iix]-(Dxx@fx)[Iix])/np.linalg.norm(ddfx))
print("L err. = ",np.linalg.norm((Dxx@fx)[Iix]+kapp*kapp*fx[Iix])/np.linalg.norm(kapp*kapp*fx[Iix]))

L = Dxx + kapp*kapp*np.identity(len(fx))
Ibx = [0,nx-1]
u = -np.linalg.solve(L[Iix][:,Iix],L[Iix][:,Ibx]@fx[Ibx])
print("err u : ",np.linalg.norm(u-fx[Iix])/np.linalg.norm(fx[Iix]))


# 2D test
ny = (int)(np.ceil(40*kapp/np.pi))
ypts = np.linspace(0.,1.,ny)

XX=np.zeros(shape=(nx*ny,2))
g = np.zeros(shape=(nx*ny,))
ddg = np.zeros(shape=(nx*ny,))
Ii = []
Ib = []
Ic0 = []
Il = []
Ir = []
Itb = []
for ij in range(nx*ny):
    i = ij//ny
    j = ij%ny 
    XX[ij,:] = [xpts[i],ypts[j]]
    g[ij] = np.sin(kapp*xpts[i])
    ddg[ij] = -kapp*kapp*np.sin(kapp*xpts[i])
    if i == 0 or j==0 or i==nx-1 or j==ny-1:
        Ib+=[ij]
        if j==0 or j==ny-1:
            Itb+=[ij]
        elif i==0:
            Il+=[ij]
        elif i==nx-1:
            Ir+=[ij]
    else:
        Ii+=[ij]
        if i==nx//2-1:
            Ic0+=[ij]
Ic=[]
for ind in range(len(Ii)):
    i = Ii[ind]//ny
    if i==nx//2-1:
        Ic+=[ind]

Dyy = stencil.stencilD2(ypts)
Ey = np.identity(ny)
L= sparse.kron(Dxx,Ey)+sparse.kron(Ex,Dyy)+kapp*kapp*sparse.identity(nx*ny)
Lii =   L[Ii][:,Ii]
Lib =   L[Ii][:,Ib]
Litb=   L[Ii][:,Itb]
Lil =   L[Ii][:,Il]
Lir =   L[Ii][:,Ir]
Sr  = -splinalg.spsolve(Lii,Lir)[Ic,:].toarray()
Sl  = -splinalg.spsolve(Lii,Lil)[Ic,:].toarray()
print("L err. = ",np.linalg.norm((L@g)[Ii])/np.linalg.norm(kapp*kapp*g[Ii]))

ui = g[Ii]
ub = g[Ib]
ur = g[Ir]
ul = g[Il]
uc = g[Ic0]
utb = g[Itb]
###########################
#   check 1: Lii@ui=-Lib@ub
###########################

print(np.linalg.norm((Lii@ui)[Ic]+(Lib@ub)[Ic])/np.linalg.norm(np.ones(shape=ui[Ic].shape)))

##########################################
#   check 2: Lib@ub = Litbutb+Lilul+Lirur
##########################################
u0 = (Lib@ub)
u00 = Litb@utb+Lil@ul+Lir@ur
print(np.linalg.norm(u0[Ic]-u00[Ic])/np.sqrt(len(Ic)))

##########################################
#   check 3: Sr map
##########################################
b0 = np.random.standard_normal(size=(len(Ir),))
u0 = Sr@b0
u00 = -(splinalg.spsolve(Lii,Lir@b0))[Ic]
print(np.linalg.norm(u0-u00)/np.sqrt(len(Ic)))

##########################################
#   check 4: balance condition
##########################################

rhs = -splinalg.spsolve(Lii,(Litb@utb))[Ic]
u0 = uc-Sl@ul-Sr@ur
print(np.linalg.norm(u0-rhs,ord=np.inf))

plt.figure(1)
plt.plot(u0)
plt.plot(rhs)
plt.legend(['u0','rhs'])

plt.figure(2)
plt.plot(uc)
plt.plot(ul)
plt.plot(ur)
plt.legend(['uc','ul','ur'])
plt.figure(3)
plt.plot(uc)
plt.plot(Sl@ul)
plt.plot(Sr@ur)
plt.legend(['uc','Sul','Sur'])
plt.show()