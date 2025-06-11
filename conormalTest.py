import numpy as np
import matplotlib.pyplot as plt
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
def stencil(p):
    x=np.linspace(0,1,p)
    h=x[1]-x[0]
    D=np.eye(len(x))
    e=np.ones(shape=(len(x)-1,))
    D=D-np.diag(e,-1)
    D=D/h
    D2 = D.T@D
    D2[0,0]=-D2[0,1]
    D2 = -D2
    return x,D,D2

p   =   100
x,Dx,D2x =   stencil(p)

y,Dy,D2y =   stencil(p)

nx=len(x)
ny=len(x)

def u(p):
    return np.sin(2.*np.pi*p[0])*np.sinh(np.pi*p[1])
def gb(p):
    return np.abs(p[0])<1e-14 or np.abs(p[0]-1)<1e-14 or np.abs(p[1])<1e-14 or np.abs(p[1]-1)<1e-14

XY = np.zeros(shape=(nx*ny,2))
XY[:,0] = np.kron(x,np.ones(y.shape))
XY[:,1] = np.kron(np.ones(x.shape),y)

print("XY shape = ",XY.shape)
Igb = [i for i in range(XY.shape[0]) if gb(XY[i,:])]
Ii = [i for i in range(XY.shape[0]) if not gb(XY[i,:])]
rhs = np.array([u(XY[i,:]) for i in Igb])


# diffOP with A = [1.,0;0.,2]
diffOP = -np.kron(D2x,np.identity(len(y)))-4*np.kron(np.identity(len(y)),D2y)

# verify that our solution is correct
sol = np.linalg.solve(diffOP[Ii,:][:,Ii],-diffOP[Ii,:][:,Igb]@rhs)
u_exact = np.array([u(XY[i,:]) for i in Ii])
print("sol err. = ",np.linalg.norm(sol-u_exact,ord=np.inf))

# the conormal derivative at y=1 should be 4*dy
# for given u this is 4*pi*sin(2*pi*x)*cosh(pi*y)

def cnrml(p):
    return -4.*np.pi*np.sin(2.*np.pi*p[0])*np.cosh(np.pi*p[1])


Aii = diffOP[Ii,:][:,Ii]
Aib = diffOP[Ii,:][:,Igb]
Abi = diffOP[Igb,:][:,Ii]
Abb = diffOP[Igb,:][:,Igb]


Iy1 = [i for i in range(len(Igb)) if np.abs(XY[Igb[i],1]-1.)<1e-14]
cnrml_y1 = [cnrml(XY[Igb[i],:]) for i in range(len(Igb)) if np.abs(XY[Igb[i],1]-1.)<1e-14]

u_exact_tot = np.array([u(XY[i,:]) for i in range(XY.shape[0])])
ui = u_exact_tot[Ii]
ub = u_exact_tot[Igb]
print("exact sol err = ",np.linalg.norm(Aii@sol+Aib@ub))

#sol = np.linalg.solve(diffOP[Ii,:][:,Ii],-diffOP[Ii,:][:,Igb]@rhs)
du_exact_bdry = -(Abb@ub - Abi@np.linalg.solve(Aii,Aib@ub))
uy1=du_exact_bdry[Iy1]

print("norm Abb         = ",np.linalg.norm(Abb,ord=2))
print("norm uy1         = ",np.linalg.norm(uy1))
print("norm cnrml_y1    = ",np.linalg.norm(cnrml_y1))

uy1=uy1/np.linalg.norm(uy1)
cnrml_y1=cnrml_y1/np.linalg.norm(cnrml_y1)

print("cnrml err. = ",np.linalg.norm(uy1-cnrml_y1,ord=np.inf))
plt.figure(1)
plt.plot(uy1)
plt.plot(cnrml_y1)
plt.legend(['u','cnrml'])
plt.show()



