import numpy as np
import torch
import matplotlib.pyplot as plt

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsaltGeom
import SOMS3D
import solver.spectral.spectralSolver as spectral

import matAssembly.matAssembler as mA
import solver.solver as solver

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

ky = 2
kz = 2
kx = np.sqrt(ky**2+kz**2)
Lx = 1/8
Ly = 1
Lz = 1
def  c11(p):
    return torch.ones_like(p[:,0])
def  c22(p):
    return torch.ones_like(p[:,1])
def  c33(p):
    return torch.ones_like(p[:,2])
def  bc(p):
    return torch.sin(np.pi*ky*p[:,1])*torch.sin(np.pi*kz*p[:,2])*torch.sinh(kx*np.pi*(Lx-p[:,0]))/np.sinh(kx*np.pi*Lx)
def  bc_np(p):
    return np.sin(np.pi*ky*p[:,1])*np.sin(np.pi*kz*p[:,2])*np.sinh(kx*np.pi*(Lx-p[:,0]))/np.sinh(kx*np.pi*Lx)
Lapl = pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33)

cx = Lx/2
bnds = np.array([[0,0,0],[Lx,Ly,Lz]])
Om = hpsaltGeom.BoxGeometry(bnds)
nbz = 8
nby = 8
nbx = 2
ax = .5*(bnds[1,0]/nbx)
ay = .5*(bnds[1,1]/nby)
az = .5*(bnds[1,2]/nbz)
px = 8
py = 8
pz = 8

print("px,py,pz = ",px," , ",py," , ",pz)


solver_hps = hpsalt.Domain_Driver(Om, Lapl, 0, np.array([ax,ay,az]), [px+1,py+1,pz+1], 3)
solver_hps.build("reduced_cpu", "MUMPS",verbose=False)

XX = solver_hps.XX
XXfull = solver_hps.XXfull

Jb = solver_hps._Jx
Ji = solver_hps.Ji

print("Ji size = ",len(Ji))
print("Jb size = ",len(Jb))

XXi = XX[Ji,:]
XXb = XX[Jb,:]

Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0))[0]

Aii = np.array(solver_hps.Aii.todense())
Aib = np.array(solver_hps.Aix.todense())


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
#rk = (px-1)*min(nby*(py-1),nbz*(pz-1))
rk = 150
print("rank = ",rk)

assemblerS = mA.rkHMatAssembler((py-1)*(pz-1),rk,tree=None,ndim=3)
assemblerT = mA.rkHMatAssembler((py-1)*(pz-1),rk,tree=None,ndim=3)


SSmap = solver.stMap(SS,XXb[Jl,:],XXi[Jc,:])
#STmap = solver.stMap(ST,XXb[Jl,:],XXi[Jc,:])

SSlinop = assemblerS.assemble(SSmap)
#STlinop = assemblerT.assemble(STmap)

E = np.identity(SS.shape[1])
v=np.random.standard_normal(size=(SS.shape[1],))
Sv1 = SSlinop@v
Sv2 = SS@v

print("Hmat err SS = ",np.linalg.norm(Sv1-Sv2)/np.linalg.norm(Sv2))
#print("Hmat err ST = ",np.linalg.norm(ST-STHdense)/np.linalg.norm(ST))