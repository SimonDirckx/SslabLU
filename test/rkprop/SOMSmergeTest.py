import numpy as np
import SOMS
import matplotlib.pyplot as plt
import scipy.special as special






def bc(p,sources,kh):
    f = 0
    for i in range(sources.shape[0]):
        r = np.sqrt((p[:,0]-sources[i,0])**2+(p[:,1]-sources[i,1])**2)
        f+= special.yn(0,kh*r)/4
    return f

kh = 30.
Lx = 2.
Ly = 2.

sources = np.zeros(shape = (0,2))
k = 0
d = .2
while k<50:
    pt = np.array([Lx/2,Ly/2])+np.random.standard_normal(size=(1,2))
    if not (pt[0,0]>-d and pt[0,0]<Lx+d and pt[0,1]>-d and pt[0,1]<Ly+d):
        sources=np.append(sources,pt,axis=0)
        k+=1




px = (int) (np.ceil(10*kh/(2*np.pi)))
py = px
print("p = ",px)
nbx = 2
nby = 2



Stot,XYtot,Ii,Ib = SOMS.SOMS_solver(px,py,nbx,nby,kh,Lx,Ly)


plt.figure(1)
plt.scatter(sources[:,0],sources[:,1])
plt.scatter(XYtot[:,0],XYtot[:,1])
plt.show()

Sii = Stot[Ii,:][:,Ii]
Sib = Stot[Ii,:][:,Ib]

ui = bc(XYtot[Ii,:],sources,kh)
ub = bc(XYtot[Ib,:],sources,kh)
print("balance err = ",np.linalg.norm(Sii@ui+Sib@ub)/np.linalg.norm(ui))
print("sol err = ",np.linalg.norm(ui+np.linalg.solve(Sii,Sib@ub))/np.linalg.norm(ui))


XYi = XYtot[Ii,:]
Ic = np.where(XYi[:,0]==1.)[0]

Icc = [i for i in range(XYi.shape[0]) if not i in Ic]


Sc = Sii[Ic,:]
Scc1 = Sii[Ic,:][:,Icc]
Scc2 = Sii[Icc,:][:,Ic]
H = (Sii[Ic,:][:,Icc]@Sib[Icc,:]-Sib[Ic,:])
SS = np.identity(len(Ic))-Sii[Ic,:][:,Icc]@Sii[Icc,:][:,Ic]

Sts = np.linalg.solve(SS,H)

print("err SS = ",np.linalg.norm(Sts@ub-ui[Ic]))
print("SS shape = ",Sts.shape)
print("rank SS = ",np.linalg.matrix_rank(Sts,rtol = 1e-6))
[U,s,Vh]= np.linalg.svd(Sts)
plt.figure(1)
plt.semilogy(s)
plt.show()





# Verify the factorization property

#Sii@ui = -Sib@ub