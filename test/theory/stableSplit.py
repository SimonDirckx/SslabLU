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
k=6
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



# harmonic extension property

H=1./8.
ord = 100
ordx = ord
bnds = np.array([[0,0],[H,1]])

discH = stencil.stencilSolver(Lapl,bnds,[ordx,ord])
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

kvec = [1,2,3,4,5,6]
c = np.zeros(shape = (len(kvec),))
ctild = np.zeros(shape = (len(kvec),))
for indk in range(len(kvec)):
    H=2**(-kvec[indk])
    ord = 256
    ordx = 8*(int)(ord*H)
    print("ordx = ",ordx)
    bnds = np.array([[0,0],[H,1]])

    discH = stencil.stencilSolver(Lapl,bnds,[ordx,ord])
    XXi = discH.XXi
    XXb = discH.XXb
    Aii = discH.Aii
    Aib = discH.Aix
    Abi = discH.Axi
    Abb = discH.Axx


    Il = np.where((np.abs(XXb[:,0])<1e-10) & ( np.abs(XXb[:,1]) > 1e-10 ) & ( np.abs(XXb[:,1] -1 ) > 1e-10 ) )[0]
    Ir = np.where((np.abs(XXb[:,0]-H)<1e-10) & ( np.abs(XXb[:,1]) > 1e-10 ) & ( np.abs(XXb[:,1] -1 ) > 1e-10 ) )[0]
    gl = np.sin(2*np.pi*XXb[Il,1])
    gr = np.sqrt(4*np.pi*XXb[Il,1])-1
    gr = gr




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
plt.legend(['coeff','ctild','O(1/H2)'])
plt.show()
