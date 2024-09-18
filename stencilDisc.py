import numpy as np



def stencil_pts(n):
    xpts = np.linspace(0,1,n)
    return xpts

def Diffmat(xpts0):
    h=xpts0[1]-xpts0[0]
    nx=len(xpts0)
    e=np.ones(shape=(nx-1,))
    D = 2.*np.identity(nx)-1.*np.diag(e,-1)-1.*np.diag(e,-1)
    D/=h
    return D


def discrLaplace(xpts,ypts):
    Dx=Diffmat(xpts)
    Dy=Diffmat(ypts)
    nx=len(xpts)
    ny=len(ypts)
    Ex=np.identity(nx)
    Ey=np.identity(ny)
    Dxx=Dx@Dx
    Dyy=Dy@Dy
    Lx = np.kron(Dxx,Ey)
    Ly = np.kron(Ex,Dyy)
    L = Lx+Ly
    return L
