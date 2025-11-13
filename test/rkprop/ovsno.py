import numpy as np
import solver.spectral.spectralSolver as spectral

import torch
import geometry.geom_2D.square as square
import matplotlib.pyplot as plt

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hps
from solver.hpsmultidomain.hpsmultidomain import pdo
import solver.hpsmultidomain.hpsmultidomain.geom as hpsGeom

geomHPS = hpsGeom.BoxGeometry(np.array([[0,0],[2,1]]))
xl = 0
xr = geomHPS.bounds[1,0]

def bc(p):
    return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

def c11(p):
    return torch.ones_like(p[:,0])
def c22(p):
    return torch.ones_like(p[:,0])
Lapl = pdo.PDO_2d(c11=c11,c22=c22)


n_box_vec = np.array([2,4,6,8],dtype=np.int32)
p=10
n_box_y = 2
svdSmat = np.zeros(shape=(n_box_y*(p-2),len(n_box_vec)))
svdTmat = np.zeros(shape=(n_box_y*(p-2),len(n_box_vec)))
errSvec = np.zeros(shape=(len(n_box_vec),))
errTvec = np.zeros(shape=(len(n_box_vec),))
for indb in range(len(n_box_vec)):
    n_boxes = n_box_vec[indb]
    a=np.array([xr/(2*n_boxes),1/(2*n_box_y)])
    
    solver = hps.Domain_Driver(geomHPS, Lapl, 0, a, p, d=(int)(2))
    solver.build("reduced_cpu", "MUMPS")
    XX = solver.XX
    Ii = solver.Ji
    Ib = solver.Jx

    XXi = XX[Ii,:]
    XXb = XX[Ib,:]

    mid = (xr+xl)/2.
    IL = np.where(np.abs(XXb[:,0])<1e-14)[0]
    IC = np.where(np.abs(XXi[:,0]-mid)<1e-14)[0]
    IR = np.where(np.abs(XXb[:,0]-xr)<1e-14)[0]




    SlT = -(solver.solver_Aii@(solver.Aix[:,IL]).todense())[IC,:]
    SrT = -(solver.solver_Aii@(solver.Aix[:,IR]).todense())[IC,:]


    ui = bc(XXi).numpy()
    ub = bc(XXb).numpy()

    

    uL = ub[IL]
    uC = ui[IC]
    uR = ub[IR]


    errT = np.linalg.norm(SlT@uL+SrT@uR-uC)/np.linalg.norm(uC)
    errTvec[indb] = errT

    Ntot = solver._XXfull.shape[0]
    #print("Ntot = ",Ntot)
    py = 2*p-3
    px = Ntot//py
    px = px-px%2-n_boxes+2

    Dx, xpts = spectral.cheb(px)
    Dy, ypts = spectral.cheb(py)

    xpts = xr*(1+xpts[::-1])/2
    ypts = (1+ypts[::-1])/2
    Dx = -2*Dx/xr
    Dy = -2*Dy

    Dxx = Dx@Dx
    Dyy = Dy@Dy

    nx=len(xpts)
    ny=len(ypts)
    #print("Ntot new = ",nx*ny)
    XY = np.zeros(shape=(nx*ny,2))

    XY[:,0] = np.kron(xpts,np.ones_like(ypts))
    XY[:,1] = np.kron(np.ones_like(xpts),ypts)

    


    Ii = np.where((XY[:,0]>0)&(XY[:,0]<xr)&(XY[:,1]>0)&(XY[:,1]<1))[0]
    Ib = [i for i in range(XY.shape[0]) if not i in Ii]
    XYi = XY[Ii,:]
    XYb = XY[Ib,:]
    IL = np.where(XYb[:,0]==0)[0]
    IR = np.where(np.abs(XYb[:,0]-xr)<1e-14)[0]
    IC = np.where(np.abs(XYi[:,0]-mid)<1e-14)[0]

    L = np.kron(Dxx,np.identity(ny))+np.kron(np.identity(nx),Dyy)

    Lii = L[Ii,:][:,Ii]
    Lib = L[Ii,:][:,Ib]

    SlS = -np.linalg.solve(Lii,Lib[:,IL])[IC,:]
    SrS = -np.linalg.solve(Lii,Lib[:,IL])[IC,:]


    ui = bc(XYi)
    ub = bc(XYb)

    uL = ub[IL]
    uC = ui[IC]
    uR = ub[IR]

    errS = np.linalg.norm(SlS@uL+SrS@uR-uC)/np.linalg.norm(uC)
    errSvec[indb] = errS
    

    [_,svdS,_]=np.linalg.svd(SlS)
    [_,svdT,_]=np.linalg.svd(SlT)
    svdSmat[:,indb] = svdS
    svdTmat[:,indb] = svdT
plt.figure(1)
plt.semilogy(svdTmat,label=['T, nbx = 2','T, nbx = 4','T, nbx = 6','T, nbx = 8'])
plt.gca().set_prop_cycle(None)
plt.semilogy(svdSmat,linestyle='dashed',label=['S, nbx = 2','S, nbx = 4','S, nbx = 6','S, nbx = 8'])
plt.legend()
plt.show()


plt.figure(2)
plt.semilogy(errTvec)
plt.gca().set_prop_cycle(None)
plt.semilogy(errSvec,linestyle='dashed')
plt.legend(["T","S"])
plt.show()

kvec = np.zeros(shape = (svdSmat.shape[0],1))

kvec[:,0] = range(1,len(kvec)+1)

print(kvec)

svdSmat=np.append(kvec,svdSmat,axis=1)
svdTmat=np.append(kvec,svdTmat,axis=1)

print("svdTmat = ",svdTmat)


fileName = 'ovsnoS.csv'
with open(fileName,'w') as f:
    f.write('k,two,four,six,eight\n')
    np.savetxt(f,svdSmat,fmt='%.16e',delimiter=',')

fileName = 'ovsnoT.csv'
with open(fileName,'w') as f:
    f.write('k,two,four,six,eight\n')
    np.savetxt(f,svdTmat,fmt='%.16e',delimiter=',')


