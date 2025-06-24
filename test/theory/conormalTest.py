import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as splinalg
import clenshawCurtis
import chebdif

#########################################
#       stencil disc
#########################################

def stencilD(pts):
    h=pts[1]-pts[0]
    D=np.eye(len(pts))
    e=np.ones(shape=(len(pts)-1,))
    D=D-np.diag(e,-1)
    return D/h
def stencilD2(pts):
    D=stencilD(pts)
    D2 = D.T@D
    D2[0,0]=-D2[0,1]
    return D2

#########################################
#       cheb. disc.
#########################################

def cheb(p):
	"""
	Computes the Chebyshev differentiation matrix and Chebyshev points for a given degree p.
	
	Parameters:
	  p: The polynomial degree
	  
	Returns:
	  D: The Chebyshev differentiation matrix
	  x: The Chebyshev points (flipped)
	"""
	# Compute Chebyshev nodes on [-1,1]
	x = np.cos(np.pi * np.arange(p) / (p - 1))
	# Compute weights: endpoints are 2 and interior are 1, with alternating signs.
	c = np.concatenate((np.array([2]), np.ones(p - 2), np.array([2])))
	c = c * (-1)**np.arange(p)
	# Create difference matrix
	X = x.repeat(p).reshape((-1, p))
	dX = X - X.T
	# Off-diagonal entries
	D = np.outer(c, 1 / c) / (dX + np.eye(p))
	# Diagonal: force row sum to zero
	D = D - np.diag(np.sum(D, axis=1))
	return np.flip(x), np.flip(np.flip(D, axis=0), axis=1)

def u(p):
    return np.sin(np.pi*p[:,0])*np.sinh(np.pi*p[:,1])

def du(p):
    if np.abs(p[0])<1e-14:
        return -np.pi*np.cos(np.pi*p[0])*np.sinh(np.pi*p[1])
    if np.abs(p[0]-1.)<1e-14:
        return np.pi*np.cos(np.pi*p[0])*np.sinh(np.pi*p[1])
    if np.abs(p[1])<1e-14:
        return -np.pi*np.sin(np.pi*p[0])*np.cosh(np.pi*p[1])
    if np.abs(p[1]-1.)<1e-14:
        return np.pi*np.sin(np.pi*p[0])*np.cosh(np.pi*p[1])
    
def gradu(p):
    return np.pi*np.cos(np.pi*p[:,0])*np.sinh(np.pi*p[:,1]),np.pi*np.sin(np.pi*p[:,0])*np.cosh(np.pi*p[:,1])

def gb(p):
    return np.abs(p[0])<1e-14 or np.abs(p[0]-1.)<1e-14 or np.abs(p[1])<1e-14 or np.abs(p[1]-1.)<1e-14
def gbx(p):
    return np.abs(p[0])<1e-14 or np.abs(p[0]-1.)<1e-14
def gby(p):
    return np.abs(p[1])<1e-14 or np.abs(p[1]-1.)<1e-14
def corner(p):
    return gbx(p) and gby(p)

p=8
x,Dx=cheb(p)
x = (1+x)/4
Dx=4*Dx
x0,w=clenshawCurtis.clenshaw_curtis_compute(p)
Wx = np.diag(w/4)




y,Dy=cheb(p)
y = (1+y)/2
Dy=2*Dy
y0,w=clenshawCurtis.clenshaw_curtis_compute(p)
Wy = np.diag(w/2)



Dxx = Dx@Dx
Dyy = Dy@Dy
DxW = np.sqrt(Wx)@Dx@np.linalg.inv(np.sqrt(Wx))
print("normality of Dx",np.linalg.norm(DxW@DxW.T-DxW.T@DxW,ord=2)/np.linalg.norm(DxW,ord=2))
DxxW = np.sqrt(Wx)@Dxx@np.linalg.inv(np.sqrt(Wx))
print("normality of Dxx",np.linalg.norm(DxxW@DxxW.T-DxxW.T@DxxW,ord=2)/np.linalg.norm(DxxW,ord=2))


nx=len(x)
ny=len(y)






Wxy = np.kron(Wx,Wy)
L1 = -np.kron(Dxx,np.identity(ny))-np.kron(np.identity(nx),Dyy)
L2 = L1

x = np.append(x[0:-1],.5+x)
nx = len(x)


XY = np.zeros(shape = (nx*ny,2))
XY[:,0] = np.kron(x,np.ones(shape=(ny,)))
XY[:,1] = np.kron(np.ones(shape=(nx,)),y)


#########################################
#
#          compute indices
#
#########################################


# global
Ii = [i for i in range(XY.shape[0]) if not gb(XY[i,:])]
Ib = [i for i in range(XY.shape[0]) if gb(XY[i,:])]

Il = [i for i in range(len(Ib)) if np.abs(XY[Ib[i],0])<1e-14 and np.abs(XY[Ib[i],1]-1)>1e-14 and np.abs(XY[Ib[i],1])>1e-14]
Ir = [i for i in range(len(Ib)) if np.abs(XY[Ib[i],0]-1)<1e-14 and np.abs(XY[Ib[i],1]-1)>1e-14 and np.abs(XY[Ib[i],1])>1e-14]
Ic = [i for i in range(len(Ii)) if np.abs(XY[Ii[i],0]-.5)<1e-14]
IC = [i for i in range(XY.shape[0]) if np.abs(XY[i,0]-.5)<1e-14]
shift = IC[0]

# local
Ii_loc = [i for i in range(XY.shape[0]) if XY[i,0]<.5-1e-14 and not gb(XY[i,:])]
Ib_loc = [i for i in range(XY.shape[0]) if np.abs(XY[i,0]-.5)<1e-14 or gb(XY[i,:])]
IL_loc = [i for i in range(XY.shape[0]) if XY[i,0]<.5-1e-14]
IR_loc = [i for i in range(XY.shape[0]) if XY[i,0]>1e-14 and XY[i,0]<.5+1e-14]
Il_loc = [i for i in range(XY.shape[0]) if np.abs(XY[i,0])<1e-14]
Ir_loc = [i for i in range(XY.shape[0]) if np.abs(XY[i,0]-.5)<1e-14]
Itot_loc = [i for i in range(XY.shape[0]) if XY[i,0]<.5+1e-14]


###############################################
#
#          form global system (using nrml)
#
###############################################

N=nx*ny
L = np.zeros(shape=(N,N))

L[:,Itot_loc[0]:Itot_loc[-1]+1][IL_loc[0]:IL_loc[-1]+1,:]=L1[IL_loc[0]:IL_loc[-1]+1,:]
L[:,shift+Itot_loc[0]:shift+Itot_loc[-1]+1][shift+IR_loc[0]:shift+IR_loc[-1]+1,:] = L2[IR_loc,:]
N = np.kron(Dx,np.identity(ny))

L[:,Itot_loc[0]:Itot_loc[-1]+1][Ir_loc[0]:Ir_loc[-1]+1,:] = N[Ir_loc[0]:Ir_loc[-1]+1,:]
L[:,shift+Itot_loc[0]:shift+Itot_loc[-1]+1][shift+Il_loc[0]:shift+Il_loc[-1]+1,:] -= N[Il_loc[0]:Il_loc[-1]+1,:]


Lii = L[Ii,:][:,Ii]
Lib = L[Ii,:][:,Ib]
Lbi = L[Ib,:][:,Ii]
Lbb = L[Ib,:][:,Ib]

rhs = -Lib@np.array(u(XY[Ib,:]))
sol = np.linalg.solve(Lii,rhs)
uvec = np.array(u(XY[Ii,:]))
print("sol err = ",np.linalg.norm(uvec-sol)/np.linalg.norm(uvec))

########################################
#
#   Schur way of forming T
#
########################################


utot = np.array(u(XY))
Ic_loc = [i for i in  range(len(Ii)) if np.abs(XY[Ii[i],0]-.5)<1e-14]
Icc_loc = [i for i in  range(len(Ii)) if not np.abs(XY[Ii[i],0]-.5)<1e-14]

Liic =   Lii[Icc_loc][:,Icc_loc]
Lci =   Lii[Ic_loc][:,Icc_loc]
Lcc =   Lii[Ic_loc][:,Ic_loc]
Lic =   Lii[Icc_loc][:,Ic_loc]

uc  =   uvec[Ic_loc]
fc  =   rhs[Ic_loc]
fi  =   rhs[Icc_loc]

Tcc     = Lcc-Lci@np.linalg.solve(Liic,Lic)
rhsT    = fc-Lci@np.linalg.solve(Liic,fi)
uctest  = np.linalg.solve(Tcc,rhsT)
print("uc err = ",np.linalg.norm(uc-uctest)/np.linalg.norm(uc))

########################################
#
#       playing around with L
#
########################################


print("=================== L experiments ===================")

p=31

xl = 0
xr = .125
xc = (xl+xr)/2


def du(p):
    
    if np.abs(p[1])<1e-14:
        return -np.pi*np.sin(np.pi*p[0])*np.cosh(np.pi*p[1])
    if np.abs(p[1]-1.)<1e-14:
        return np.pi*np.sin(np.pi*p[0])*np.cosh(np.pi*p[1])
    
    if np.abs(p[0]-xl)<1e-14:
        return -np.pi*np.cos(np.pi*p[0])*np.sinh(np.pi*p[1])
    if np.abs(p[0]-xr)<1e-14:
        return np.pi*np.cos(np.pi*p[0])*np.sinh(np.pi*p[1])
    


def gbx(p):
    return np.abs(p[0]-xl)<1e-15 or np.abs(p[0]-xr)<1e-15
def gby(p):
    return np.abs(p[1])<1e-14 or np.abs(p[1]-1.)<1e-14
def corner(p):
    return gbx(p) and gby(p)
def gb(p):
    return gbx(p) or gby(p)

x,Dtot = chebdif.chebdif(p,2)
Dx=Dtot[0]
Dxx = Dtot[1]
x = (1+x)/2
x = (xr-xl)*x
d = (xr-xl)
Dxx=4*Dxx/(d*d)
Dx = (2/d)*Dx

y,Dtot = chebdif.chebdif(p,2)
Dy=Dtot[0]
Dyy = Dtot[1]
y = (1+y)/2
Dyy=4*Dyy
Dy=2*Dy



x0,w=clenshawCurtis.clenshaw_curtis_compute(p)
Wx = np.diag(w/2)*d
y0,w=clenshawCurtis.clenshaw_curtis_compute(p)
Wy = np.diag(w/2)

# check that Wx is correct

f1 = 1+x**2
f2 = np.sin(x)

def Icheck(x):
    return -np.cos(x)*x**2+2*x*np.sin(x)+np.cos(x)
Iw = f1.T@Wx@f2
Iexact = Icheck(xr)-Icheck(xl)
print("I err = ",Iw-Iexact)

nx = len(x)
ny = len(y)


XY = np.zeros(shape = (nx*ny,2))
XY[:,0] = np.kron(x,np.ones(shape=(ny,)))
XY[:,1] = np.kron(np.ones(shape=(nx,)),y)

Ii = [i for i in range(XY.shape[0]) if not gb(XY[i,:])]
Ib = [i for i in range(XY.shape[0]) if gb(XY[i,:]) and not corner(XY[i,:])]
Ibtot = [i for i in range(XY.shape[0]) if gb(XY[i,:])]
Ic = [i for i in range(len(Ii)) if np.abs(XY[Ii[i],0]-xc)<1e-14]
Icorner = [i for i in range(XY.shape[0]) if corner(XY[i,:])]

Il = [i for i in range(XY.shape[0]) if np.abs(XY[i,0]-xl)<1e-14]
Ir = [i for i in range(XY.shape[0]) if np.abs(XY[i,0]-xr)<1e-14]
Iu = [i for i in range(XY.shape[0]) if np.abs(XY[i,1]-1.)<1e-14]
Id = [i for i in range(XY.shape[0]) if np.abs(XY[i,1])<1e-14]


Ilb = [i for i in range(len(Ib)) if np.abs(XY[Ib[i],0]-xl)<1e-14]
Irb = [i for i in range(len(Ib)) if np.abs(XY[Ib[i],0]-xr)<1e-14]
Iub = [i for i in range(len(Ib)) if np.abs(XY[Ib[i],1]-1.)<1e-14]
Idb = [i for i in range(len(Ib)) if np.abs(XY[Ib[i],1])<1e-14]

Ilbtot = [i for i in range(len(Ibtot)) if np.abs(XY[Ibtot[i],0]-xl)<1e-14 ]
Irbtot = [i for i in range(len(Ibtot)) if np.abs(XY[Ibtot[i],0]-xr)<1e-14 ]
Iubtot = [i for i in range(len(Ibtot)) if np.abs(XY[Ibtot[i],1]-1.)<1e-14]
Idbtot = [i for i in range(len(Ibtot)) if np.abs(XY[Ibtot[i],1])<1e-14]

Wb = np.zeros(shape=(len(Ib),len(Ib)))
Wxsub = Wx[1:-1][:,1:-1]
Wysub = Wy[1:-1][:,1:-1]


L = -np.kron(Dxx,np.identity(ny))-np.kron(np.identity(nx),Dyy)
dx = np.kron(Dx,np.identity(ny))
dy = np.kron(np.identity(nx),Dy)
Wxy = np.kron(Wx,Wy)


Nx = np.kron(Dx,np.identity(ny))
Ny = np.kron(np.identity(nx),Dy)

cN = np.zeros(shape=Nx.shape)

cN[Il] = -dx[Il]
cN[Ir] =  dx[Ir]
cN[Iu] =  dy[Iu]
cN[Id] = -dy[Id]

L[Ib] = cN[Ib]
#L[Icorner] = 0
#L[:,Icorner] = 0

Lii = L[Ii,:][:,Ii]
Lib = L[Ii,:][:,Ib]
Lbi = L[Ib,:][:,Ii]
Lbb = L[Ib,:][:,Ib]

LW = np.sqrt(Wxy)@L@np.linalg.inv(np.sqrt(Wxy))

#L = np.append(np.append(Lii,Lib,axis=1),np.append(Lbi,Lbb,axis=1),axis=0)

uvec = u(XY)
ui = uvec[Ii]
ub = uvec[Ib]
ubtot = uvec[Ibtot]
dnu = np.array([du(XY[i,:]) for i in Ib])
dnutot = np.zeros(shape = (len(Ibtot),))
dnutot[Ilbtot] = ((-dx@uvec)[Ibtot])[Ilbtot]
dnutot[Irbtot] = ((dx@uvec)[Ibtot])[Irbtot]
dnutot[Iubtot] = ((dy@uvec)[Ibtot])[Iubtot]
dnutot[Idbtot] = ((-dy@uvec)[Ibtot])[Idbtot]
dnu0 = (L[Ib]@uvec)

uu = ubtot[Iubtot]
ud = ubtot[Iubtot]
uu_exact = np.sin(np.pi*x)*np.sinh(np.pi)

ud_exact = np.sin(np.pi*x)*np.sinh(np.pi)
dnuu = np.pi*np.sin(np.pi*x)*np.cosh(np.pi)
dnud = -np.pi*np.sin(np.pi*x)*np.cosh(0)
print("IPexact = ",uu.T@Wx@dnuu+ud.T@Wx@dnud)
plt.figure(1)
plt.plot(dnutot[Irbtot])
plt.show()


sol = -np.linalg.solve(Lii,Lib@ub)
#uvec[Icorner]=0
Lu = L@uvec
dnu00 = Lbb@ub-Lbi@np.linalg.solve(Lii,Lib@ub)


def L2bdry_ip(u,v):
    ul = u[Ilbtot]
    ur = u[Irbtot]
    uu = u[Iubtot]
    ud = u[Idbtot]

    vl = v[Ilbtot]
    vr = v[Irbtot]
    vu = v[Iubtot]
    vd = v[Idbtot]
    print('****')
    print(vl.T@(Wy@ul),' , ',vr.T@(Wy@ur),' , ',vu.T@(Wx@uu),' , ',vd.T@(Wx@ud))
    return vl.T@(Wy@ul)+vr.T@(Wy@ur)+vu.T@(Wx@uu)+vd.T@(Wx@ud)

def L2bdry_nrm(v):
    return np.sqrt(L2bdry_ip(v,v))
print("err sol   = ",np.linalg.norm((ui-sol))/np.linalg.norm(ui))
print("L2 err sol   = ",np.linalg.norm(np.sqrt(Wxy[Ii][:,Ii])@(ui-sol))/np.linalg.norm(np.sqrt(Wxy[Ii][:,Ii])@ui))
#print("L2 err cnrml = ",L2bdry_nrm(dnu-dnu0)/L2bdry_nrm(dnu))
print("err cnrml = ",np.linalg.norm(dnu-dnu00))
print("L2 nrm Lui   = ",np.linalg.norm(np.sqrt(Wxy[Ii][:,Ii])@Lu[Ii]))
print("Linf nrm Lui = ",np.linalg.norm(np.sqrt(Wxy[Ii][:,Ii])@Lu[Ii],ord=np.inf))
print(ubtot.shape)
print(dnutot.shape)
print(len(Ibtot))
ipCnrml = L2bdry_ip(dnutot,ubtot)
print("L2 ip cnrml  = ",ipCnrml)

dxu , dyu = gradu(XY)
#dxu[Icorner] = 0
#dyu[Icorner] = 0
ipGrad = dxu.T@Wxy@dxu + dyu.T@Wxy@dyu
print("L2 ip err  = ",np.abs(ipGrad-ipCnrml)/np.abs(ipGrad))

def Ix1(x):
    return .5*(x+np.sin(2*np.pi*x)/(2*np.pi))
def Ix2(y):
    return -.5*(y+np.sinh(-2*np.pi*y)/(2*np.pi))
def Iy1(x):
    return .5*(x-np.sin(2*np.pi*x)/(2*np.pi))
def Iy2(y):
    return .5*(y-np.sinh(-2*np.pi*y)/(2*np.pi))
I = (Ix1(xr)-Ix1(xl))*(Ix2(1)-Ix2(0.))+(Iy1(xr)-Iy1(xl))*(Iy2(1)-Iy2(0.))
I *= np.pi*np.pi
print(I)
print(ipGrad)
print(ipCnrml)
print(np.abs(I-ipGrad))
print(np.abs(I-ipCnrml))



'''
ip = ui.T@Wxy[Ii][:,Ii]@Lu[Ii]+ub.T@Wb@Lu[Ib]

print('ip0 = ',ip0)
print('ip = ',ip)
#print('ipdiff = ',np.abs(ip1-ip2)/np.abs(ip1))
#print('ipdiff = ',np.abs(ip1+ip2)/np.abs(ip1))

dnu = np.array([du(XY[i,:]) for i in Ib])
dnu0 = (L[Ib]@uvec)
print("nrm Lu = ",np.linalg.norm(L@uvec))
sol = -np.linalg.solve(Lii,Lib@ub)
print("err. sol = ",np.linalg.norm(ui-sol)/np.linalg.norm(ui))
print("err. dn = ",np.linalg.norm(dnu-dnu0)/np.linalg.norm(dnu))




[e,V]=np.linalg.eig(LW)
FOV = np.diag(V.conj().T@(LW@V))
plt.figure(1)
plt.scatter(np.real(FOV),np.imag(FOV))
#plt.scatter(np.real(e),np.imag(e))
plt.show()

print("================symmetries================")
print('Dxx      : ',np.linalg.norm(Dxx-Dxx.T,ord=2)/np.linalg.norm(Dxx,ord=2))
print('L      : ',np.linalg.norm(L-L.T,ord=2)/np.linalg.norm(L,ord=2))
print('LW      : ',np.linalg.norm(LW-LW.T,ord=2)/np.linalg.norm(LW,ord=2))
print('Li      : ',np.linalg.norm(L[Ii]-L[:,Ii].T,ord=2)/np.linalg.norm(L[Ii],ord=2))
print('Lbb      : ',np.linalg.norm(Lbb-Lbb.T,ord=2)/np.linalg.norm(Lbb,ord=2))
print("==========================================")
'''

########################################
#
#       normality of S
#
########################################

print("len(Ilb) = ",len(Ilb))
print("len(Ic) = ",len(Ic))

Scl = (np.linalg.solve(Lii,Lib[:,Ilb]))[Ic]
Scr = (np.linalg.solve(Lii,Lib[:,Irb]))[Ic]
n   = Scl.shape[0]
Stot = np.identity(2*n)
Stot[0:n,n:2*n] = Scr
Stot[n:2*n,0:n] = Scl

Ws = np.zeros(shape = (2*n,2*n))
Ws[0:n][:,0:n] = Wxsub
Ws[n:2*n][:,n:2*n] = Wxsub

SW = np.sqrt(Ws)@Stot@np.linalg.inv(np.sqrt(Ws))
[R,Z] = splinalg.schur(SW)
e = np.diag(R)
h = np.sqrt(np.linalg.norm(SW)**2-np.linalg.norm(e)**2)

print("symmetry of S",np.linalg.norm(Stot-Stot.T,ord=2)/np.linalg.norm(Stot,ord=2))
print("normality of S",np.linalg.norm(Stot@Stot.T-Stot.T@Stot,ord=2)/np.linalg.norm(Stot,ord=2))
print("symmetry of SW",np.linalg.norm(SW-SW.T,ord=2))
print("normality of SW",np.linalg.norm(SW@SW.T-SW.T@SW,ord=2))
print("norm of SW",np.linalg.norm(SW,ord=2))
print("norm of S",np.linalg.norm(Stot,ord=2))
print("norm of Scl",np.linalg.norm(Scl,ord=2))
print("norm2 of SW",np.linalg.norm(SW,ord=2)**2)
print("norm2 of S",np.linalg.norm(Stot,ord=2)**2)
print("h = ",h)

[e,V]=np.linalg.eig(SW)
FOV = np.zeros(shape=(100*V.shape[1],),dtype='complex')
for i in range(V.shape[1]):
    for k in range(100):
        v = np.random.standard_normal(size=(V.shape[1],))+1j*np.random.standard_normal(size=(V.shape[1],))#+V[:,i]
        v=v/np.linalg.norm(v)
        FOV[100*i+k] = v.conj().T@(SW@v)

plt.figure(1)
plt.scatter(np.real(FOV),np.imag(FOV))
#plt.axis('equal')
plt.show()