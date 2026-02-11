import numpy as np
import torch
import matplotlib.pyplot as plt

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsaltGeom
import SOMS3D

"""
script that illustrates the 2-box problem in 3D

     ___________ __________
    |           |           |
    |   tau     |   sig     |
  ul|         uc|           |
    |___________|___________|

Solution map S: ul-> uc has known eigenvalues and eigenfunction, both for Laplace and Helmholtz
Constructed in two ways: overlapping and non-overlapping

"""

k = 3
Lx = 2
Ly = 1
Lz = 1
def  c11(p):
    return torch.ones_like(p[:,0])
def  c22(p):
    return torch.ones_like(p[:,1])
def  c33(p):
    return torch.ones_like(p[:,2])
def  bc(p):
    return torch.sin(np.pi*k*p[:,1])*torch.sin(np.pi*k*p[:,2])*torch.sinh(np.sqrt(2)*k*np.pi*(Lx-p[:,0]))/np.sinh(np.sqrt(2)*k*np.pi*Lx)
def  bc_np(p):
    return np.sin(np.pi*k*p[:,1])*np.sin(np.pi*k*p[:,2])*np.sinh(np.sqrt(2)*k*np.pi*(Lx-p[:,0]))/np.sinh(np.sqrt(2)*k*np.pi*Lx)
Lapl = pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33)

cx = Lx/2
bnds = np.array([[0,0,0],[Lx,Ly,Lz]])
Om = hpsaltGeom.BoxGeometry(bnds)
nbz = 2
nby = 2
nbx = 2
ax = .5*(bnds[1,0]/nbx)
ay = .5*(bnds[1,1]/nby)
az = .5*(bnds[1,2]/nbz)
px = 24
py = 12
pz = 12


print("px,py,pz = ",px," , ",py," , ",pz)


solver = hpsalt.Domain_Driver(Om, Lapl, 0, np.array([ax,ay,az]), [px+1,py+1,pz+1], 3)
solver.build("reduced_cpu", "MUMPS",verbose=False)

XX = solver.XX
XXfull = solver.XXfull

Jb = solver._Jx
Ji = solver.Ji

print("Ji size = ",len(Ji))
print("Jb size = ",len(Jb))

XXi = XX[Ji,:]
XXb = XX[Jb,:]

Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0))[0]

Aii = np.array(solver.Aii.todense())
Aib = np.array(solver.Aix.todense())


#test if I set it up correctly
ui = bc(XXi).cpu().detach().numpy()
rhs = bc(XXb).cpu().detach().numpy()
ui_hat = -np.linalg.solve(Aii,Aib@rhs)

print("err hps = ",np.linalg.norm(ui-ui_hat)/np.linalg.norm(ui))


rhsT = bc(XXb).cpu().detach().numpy()
uT = bc(XXi[Jc,:]).cpu().detach().numpy()

ST = -np.linalg.solve(Aii,Aib[:,Jl])[Jc,:]
print("ST shape = ",ST.shape)
uhat_T = ST@rhsT[Jl]




print("err1 = ",np.linalg.norm(uhat_T-uT,ord=2)/np.linalg.norm(uT,ord=2))

Sii,Sib,XYtot,Ii,Ib = SOMS3D.SOMS_solver(px,py,pz,nbx,nby,nbz,Lx,Ly,Lz,0,0)


XXi = XYtot[Ii,:]
XXb = XYtot[Ib,:]


Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0))[0]




AiiS = Sii
AibS = Sib

rhsS = bc_np(XXb)
uS = bc_np(XXi[Jc,:])
SS = -np.linalg.solve(AiiS,AibS[:,Jl])[Jc,:]
uhat_S = SS@rhsS[Jl]


print("err2 = ",np.linalg.norm(uhat_S-uS,ord=2)/np.linalg.norm(uS,ord=2))


# The continuum eigenvalues are known, they are k^2, k \in Nat

N = len(Jl)

[eS,VS] = np.linalg.eig(SS)
[eT,VT] = np.linalg.eig(ST)
print("SS shape = ",SS.shape)
print("Jl len = ",len(Jl))

nex = 5
ney = 5
e_known = np.zeros(shape = (nex*ney,))
inds = np.zeros(shape = (nex*ney,2))
for i in range(nex):
    for j in range(ney):
        kx = np.sqrt((i+1)*(i+1)+(j+1)*(j+1))
        e_known[i+j*nex] = np.sinh(np.pi*kx)/np.sinh(2*np.pi*kx)
        inds[i+j*nex] = [i,j]
p = np.argsort(e_known)
p = p[::-1]
e_known = e_known[p]
inds = inds[p,:]
#print(1+inds[:25,:])
eS = np.sort(np.abs(eS))[::-1]
eT = np.sort(np.abs(eT))[::-1]


#fileName = 'eigvalsST3D'+str(nbx)+'.csv'
#eMat = np.zeros(shape=(25,4))
#eMat[:,0] = np.arange(0,25)
#eMat[:,1] = e_known[:25]
#eMat[:,2] = eS[:25]
#eMat[:,3] = eT[:25]
#with open(fileName,'w') as f:
#    f.write('ind,e,eS,eT\n')
#    np.savetxt(f,eMat,fmt='%.16e',delimiter=',')


plt.figure(1)
plt.semilogy(eS[:25])
plt.semilogy(eT[:25])
plt.semilogy(e_known[:25])
plt.legend(['eS','eT','e'])
plt.show()





