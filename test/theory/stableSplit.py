import geometry.geom_2D.square as square
import solver.stencil.stencilSolver as stencil
from solver.spectralmultidomain.hps import pdo
import numpy as np
import scipy.sparse.linalg as splinalg

import matplotlib.pyplot as plt

ndim = 3

def c11(p):
    return np.ones_like(p[:,0])
def c22(p):
    return np.ones_like(p[:,0])
def c33(p):
    return np.ones_like(p[:,0])

Lapl=pdo.PDO3d(c11,c22,c33)
k=4
ord = (2**k)+1
ord = [ord,ord,ord]
disc = stencil.stencilSolver(Lapl,np.array([[0,0,0],[1,1,1]]),ord)


#test correctness of stencil
print("============= TEST CORRECTNESS =============")



def bc(p):
    return np.sin(p[:,0])*np.sinh(p[:,1])

XXi = disc.XXi
XXb = disc.XXb

rhs = -disc.Aix@bc(XXb)
sol = splinalg.spsolve(disc.Aii,rhs)
u = bc(XXi)
print("err = ",np.linalg.norm(u-sol)/np.linalg.norm(u))



# harmonic extension property
print("============= TEST HARM. EXT. =============")
H=1./8.

ordx = 100
bnds = np.array([[0,0,0],[H,1,1]])

discH = stencil.stencilSolver(Lapl,bnds,[ordx,ord[1],ord[2]])
XXi = discH.XXi
XXb = discH.XXb
Aii = discH.Aii
Aib = discH.Aix
Abi = discH.Axi
Abb = discH.Axx



Il = np.where((np.abs(XXb[:,0])<1e-10) & ( np.abs(XXb[:,1]) > 1e-10 ) & ( np.abs(XXb[:,1] -1 ) > 1e-10 ) )[0]
gl = np.sin(np.pi*XXb[Il,1])
sol_l = splinalg.spsolve(Aii,-Aib[:,Il]@gl)
Tll = Abb[Il,:][:,Il]-Abi[Il,:]@splinalg.spsolve(Aii,Aib[:,Il])

ip1 = gl@Tll@gl
ip2 = gl@Abi[Il,:]@sol_l+gl@Abb[Il,:][:,Il]@gl
print("ip err = ",np.abs(ip1-ip2)/np.abs(ip1))


# thin strip property
print("============= TEST THIN STRIP =============")
kvec = [3,4,5]
c = np.zeros(shape = (len(kvec),))
ctild = np.zeros(shape = (len(kvec),))
for indk in range(len(kvec)):
    H=2**(-kvec[indk])
    ord = 16
    ordx = (int)(4*ord*H)
    print("ordx = ",ordx)
    bnds = np.array([[0,0,0],[H,1,1]])

    discH = stencil.stencilSolver(Lapl,bnds,[ordx,ord,ord])
    XXi = discH.XXi
    XXb = discH.XXb
    Aii = discH.Aii
    Aib = discH.Aix
    Abi = discH.Axi
    Abb = discH.Axx


    Il = np.where((np.abs(XXb[:,0])<1e-10) & ( np.abs(XXb[:,1]) > 1e-10 ) & ( np.abs(XXb[:,1] -1 ) > 1e-10 ) )[0]
    Ir = np.where((np.abs(XXb[:,0]-H)<1e-10) & ( np.abs(XXb[:,1]) > 1e-10 ) & ( np.abs(XXb[:,1] -1 ) > 1e-10 ) )[0]



    Tll = Abb[Il,:][:,Il]-Abi[Il,:]@splinalg.spsolve(Aii,Aib[:,Il])
    Trr = Abb[Ir,:][:,Ir]-Abi[Ir,:]@splinalg.spsolve(Aii,Aib[:,Ir])
    Tlr = Abb[Il,:][:,Ir]-Abi[Il,:]@splinalg.spsolve(Aii,Aib[:,Ir])
    Trl = Abb[Ir,:][:,Il]-Abi[Ir,:]@splinalg.spsolve(Aii,Aib[:,Il])

    nl = len(Il)
    nr = len(Ir)

    Ttot = np.zeros(shape = (nl+nr,nl+nr))
    T0 = np.zeros(shape = (nl+nr,nl+nr))

    Ttot[0:nl,:][:,0:nl] = Tll.todense()
    Ttot[nl:nl+nr,:][:,nl:nl+nr] = Trr.todense()
    Ttot[0:nl,:][:,nl:nl+nr] = Tlr.todense()
    Ttot[nl:nl+nr,:][:,0:nl] = Trl.todense()

    T0[0:nl,:][:,nl:nl+nr] = Tlr.todense()
    T0[nl:nl+nr,:][:,0:nl] = Trl.todense()



    [e,V] = np.linalg.eig(Ttot)

    imin = np.argmin(abs(e))
    vmin = V[:,imin]
    vl = vmin[0:nl]
    vr = vmin[nl:nl+nr]


    ipl = vl.T@(Tll@vl)
    ipr = vr.T@(Trr@vr)
    iplr = vl.T@(Tlr@vr)
    iprl = vr.T@(Trl@vl)

    print("ipr+ipl = ",ipr+ipl)
    print("iprl = ",iprl)
    print("iplr = ",iplr)
    print("iptot = ",ipr+ipl+iprl+iplr)
    print("(ipr+ipl)/iptot = ",(ipr+ipl)/(ipr+ipl+iprl+iplr))
    print("========================================")
    c[indk] = (ipr+ipl)/(ipr+ipl+iprl+iplr)

    [e0,V0] = np.linalg.eig(T0)

    imin = np.argmin(e0)
    vmin = V0[:,imin]
    vl = vmin[0:nl]
    vr = vmin[nl:nl+nr]


    ipl = vl.T@(Tll@vl)
    ipr = vr.T@(Trr@vr)
    iplr = vl.T@(Tlr@vr)
    iprl = vr.T@(Trl@vl)

    print("ipr+ipl = ",ipr+ipl)
    print("iprl = ",iprl)
    print("iplr = ",iplr)
    print("iptot = ",ipr+ipl+iprl+iplr)
    print("(ipr+ipl)/iptot = ",(ipr+ipl)/(ipr+ipl+iprl+iplr))
    ctild[indk] = (ipr+ipl)/(ipr+ipl+iprl+iplr)





Hvec = 2.**(-np.array(kvec))

plt.figure(1)
plt.loglog(Hvec,c)
plt.loglog(Hvec,ctild)
plt.loglog( Hvec,(1./(Hvec*Hvec))*c[0]*(Hvec[0]*Hvec[0]) ,linestyle="dashed")
plt.loglog( Hvec,(1./(Hvec**(1.5)))*c[0]*(Hvec[0]**(1.5)) ,linestyle="dashed")
plt.legend(['coeff','ctild','O(1/H2)','O(1/H1.5)'])
plt.show()



