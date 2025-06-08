import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
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
    return np.sin(2.*np.pi*p[0])*np.sinh(np.pi*p[1])

def du(p):
    if np.abs(p[0])<1e-14:
        return -2.*np.pi*np.cos(2.*np.pi*p[0])*np.sinh(2.*np.pi*p[1])
    if np.abs(p[0]-1.)<1e-14:
        return 2.*np.pi*np.cos(2.*np.pi*p[0])*np.sinh(2.*np.pi*p[1])
    if np.abs(p[1])<1e-14:
        return -2*np.pi*np.sin(2.*np.pi*p[0])*np.cosh(2.*np.pi*p[1])
    if np.abs(p[1]-1.)<1e-14:
        return 2*np.pi*np.sin(2.*np.pi*p[0])*np.cosh(2.*np.pi*p[1])

def gb(p):
    return np.abs(p[0])<1e-14 or np.abs(p[0]-1.)<1e-14 or np.abs(p[1])<1e-14 or np.abs(p[1]-1.)<1e-14
def gbx(p):
    return np.abs(p[0])<1e-14 or np.abs(p[0]-1.)<1e-14
def gby(p):
    return np.abs(p[1])<1e-14 or np.abs(p[1]-1.)<1e-14


p=20

x,Dx=cheb(p)
print(x.shape)
Dxx = chebdif.chebdif(p,2)
Dxx=Dxx[1][1]
x=(x+1.)/2.
Dxx = 4*Dxx

y,Dy=cheb(p)
Dyy = chebdif.chebdif(p,2)
y=(y+1.)/2.
Dyy=Dyy[1][1]
Dyy = 4*Dyy

nx=len(x)
ny=len(y)
XY = np.zeros(shape = (nx*ny,2))
XY[:,0] = np.kron(x,np.ones(shape=(ny,)))
XY[:,1] = np.kron(np.ones(shape=(nx,)),y)




L = -np.kron(Dxx,np.identity(ny))-np.kron(np.identity(nx),Dyy)


Ii = [i for i in range(XY.shape[0]) if not gb(XY[i,:])]
Ib = [i for i in range(XY.shape[0]) if gb(XY[i,:])]
Ix1 = [i for i in range(XY.shape[0]) if np.abs(XY[i,0]-1)<1e-14]


Lii = L[Ii,:][:,Ii]
Lib = L[Ii,:][:,Ib]
Lbi = L[Ib,:][:,Ii]
Lbb = L[Ib,:][:,Ib]

ub = np.array([u(XY[i,:]) for i in Ib])
ui  = np.array([u(XY[i,:]) for i in Ii])
dux1 = np.array([du(XY[i,:]) for i in Ib])
sol = np.linalg.solve(Lii,-Lib@ub)

dtnu = -(Lbb@ub-Lbi@np.linalg.solve(Lii,Lib@ub))
N=L.shape[0]
print("sqrt(N) = ",np.sqrt(N))
dtnu=dtnu/np.linalg.norm(dtnu)
dux1=dux1/np.linalg.norm(dux1)

print("sol err = ",np.linalg.norm(ui-sol,ord=np.inf))
print("conormal err = ",np.linalg.norm(dux1-dtnu)/np.linalg.norm(dtnu))
plt.figure(1)
plt.plot(dux1)
plt.plot(dtnu)
plt.legend(["dux1","dtnu"])
plt.show()
