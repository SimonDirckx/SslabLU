import numpy as np
import solver.stencil.stencilSolver as stencil
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.special as special
import greens
import spectralDisc as spectral

kapp = 20.12
lam = 2*np.pi/kapp
n_pts = 10./lam


ny = 20#10*int(np.ceil(n_pts))
nx=ny
if nx%2==0:
    nx=nx+1
xpts = (1+spectral.cheb_pts(nx))/2.
ypts = (1+spectral.cheb_pts(ny))/2.
nx=len(xpts)
ny=len(ypts)
Dx = spectral.Diffmat(xpts)
Dy = spectral.Diffmat(ypts)
Ex = np.identity(nx)
Ey = np.identity(ny)
Dxx = Dx@Dx
Dyy = Dy@Dy

L = np.kron(Dxx,Ey)+np.kron(Ex,Dyy)+kapp*kapp*np.identity(nx*ny)
XX = np.zeros(shape=(nx*ny,2))

Ii=[]
Ib=[]
Ic=[]
Ic0=[]
Il=[]
Ir=[]
Igb=[]
for ij in range(nx*ny):
    i=ij//ny
    j=ij%ny
    XX[ij,:]=[xpts[i],ypts[j]]
    if i==0 or i==nx-1 or j==0 or j==ny-1:
        Ib+=[ij]
        if i==0 and not j==0 and not j==ny-1:
            Il+=[ij]
        elif i==nx-1 and not j==0 and not j==ny-1:
            Ir+=[ij]
        else:
            Igb+=[ij]
    else:
        Ii+=[ij]
        if i==(nx-1)//2:
            Ic0+=[ij]
ind = 0
for ind in range(len(Ii)):
    ij = Ii[ind]
    i=ij//ny
    j=ij%ny
    if i==(nx-1)//2:
        Ic+=[ind]
u = greens.get_known_greens(XX,kapp)
ub=u[Ib]
Lii = L[Ii][:,Ii]
Lib = L[Ii][:,Ib]
print("\|(Lu)_i\| = ",np.linalg.norm((L@u)[Ii])/np.linalg.norm(u))
print("\|Liiu_i\| = ",np.linalg.norm(Lii@u[Ii]))
rhs = -Lib@ub
uhat=np.zeros(shape=u.shape)
uhat[Ib]=ub
uhat[Ii] = np.linalg.solve(Lii,rhs)
print("err at (hx,hy)=( ",1./nx," , ",1./ny, " ) is ",np.linalg.norm(u-uhat)/np.linalg.norm(u))
triang = tri.Triangulation(XX[:,0], XX[:,1])


plt.figure(1)
plt.tricontourf(triang, u,100)
plt.colorbar()
plt.axis('equal')
plt.figure(2)
plt.tricontourf(triang, uhat,100)
plt.colorbar()
plt.axis('equal')
plt.show()

