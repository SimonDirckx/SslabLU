import numpy as np


##################################
#       px=py=p
##################################


def cheb_pts(p):
    xpts = np.zeros(shape=(p+1,))
    for i in range(p+1):
        xpts[i] = np.sin(np.pi*(p-2*i)/(2*p))
    return xpts
def cheb_pts0(p):
    xpts = np.zeros(shape=(p+1,))
    for i in range(p+1):
        xpts[i] = np.cos(((2*i+1)/(2*(p+1)))*np.pi)
    return xpts

def Diffmat(xpts0):
    
    p=len(xpts0)-1
    l = np.abs(xpts0[p]-xpts0[0])
    xpts = cheb_pts(p)
    D = np.zeros(shape=(p+1,p+1))
    for i in range(p+1):
        ci=1.
        if i==0 or i==p:
            ci=2.
        for j in range(p-i,p+1):
            cj=1.
            if j==0 or j==p:
                cj=2.
            if i!=j:
                sgn = (-1)**(i+j)
                D[i,j] = (ci/cj)*sgn*(1./((xpts[i]-xpts[j])))
                D[p-i,p-j]=-D[i,j]
    for i in range(p+1):
        d=D[i,:]
        d=np.sort(d)
        s=0.
        for j in range(p+1):
            s+=d[j]
        D[i,i] = -s
    n = np.floor(p+1).astype(int)
    for i in range(n):
        D[p-i,p-i]=-D[i,i]
    D*=(2./l)
    return D

def discrN(xpts,ypts):
    D = Diffmat(xpts)
    py=len(ypts)-1
    Ey = np.identity(py+1)
    N = np.kron(D,Ey)
    return N

def discrLaplace(xpts,ypts):
    px=len(xpts)-1
    py=len(ypts)-1
    Dx=Diffmat(xpts)
    Dy=Diffmat(ypts)
    Ex=np.identity(px+1)
    Ey=np.identity(py+1)
    Dxx=Dx@Dx
    Dyy=Dy@Dy
    Lx = np.kron(Dxx,Ey)
    Ly = np.kron(Ex,Dyy)
    L = Lx+Ly
    return L