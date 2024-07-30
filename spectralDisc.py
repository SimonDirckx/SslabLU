import numpy as np

def cheb_pts(p):
    xpts = np.zeros(shape=(p+1,))
    for i in range(p+1):
        xpts[i] = np.sin(np.pi*(p-2*i)/(2*p))
    return xpts

def Diffmat(xpts):
    p=len(xpts)-1
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
    return D


def XYpoints(H,L,p):
    xpts = -cheb_pts(p)
    ypts = xpts
    xpts = .5*H*(xpts+1.)
    ypts =.5*L*(ypts+1.)
    XY=np.zeros(shape=((p+1)*(p+1),2))
    for j in range(0,p+1):
        for i in range(0,p+1):    
            XY[j+i*(p+1),:] = [xpts[i],ypts[j]]
    return XY,xpts,ypts

def discrN(pts):
    D = Diffmat(pts)
    n=len(pts)
    E = np.identity(n)
    N = np.kron(D,E)
    return N

def discrLaplace(xpts,ypts):
    p=len(xpts)-1
    Dx=Diffmat(xpts)
    Dy=Diffmat(ypts)
    E=np.identity(p+1)
    Dxx=Dx@Dx
    Dyy=Dy@Dy
    Lx = np.kron(Dxx,E)
    Ly = np.kron(E,Dyy)
    L = Lx+Ly
    return L
def partition_single_slab(H,L,XY):
    # XYL and XYR are vector of points in left and right slabs
    # Conforms with the differential matrices!
    # note that XYL+XYR =/= XY
    # slab is present in both!
    
    Iobl   =   []
    Ioblc  =   []
    Icbl   =   []
    Icblc  =   []

    Iobr   =   []
    Iobrc  =   []
    Icbr   =   []
    Icbrc  =   []

    Ii     =   []

    N=(XY.shape)[0]
    for ij in range(N):
        x = XY[ij,0]
        y = XY[ij,1]

        if np.abs(x)<1e-15:
            Icbl+=[ij]
            Iobrc+=[ij]
            Icbrc+=[ij]
            if (np.abs(y)>1e-15 and np.abs(y-L)>1e-15):
                Iobl+=[ij]
            else:
                Ioblc+=[ij]
        elif np.abs(x-H)<1e-15:
            Icbr+=[ij]
            Ioblc+=[ij]
            Icblc+=[ij]
            if (np.abs(y)>1e-15 and np.abs(y-L)>1e-15):
                Iobr+=[ij]
            else:
                Iobrc+=[ij]
        elif np.abs(y)<1e-15 or np.abs(y-L)<1e-15:
            Icblc+=[ij]
            Icbrc+=[ij]
            Ioblc+=[ij]
            Iobrc+=[ij]
        else:
            Ii+=[ij]
    return Iobl,Ioblc,Icbl,Icblc,Iobr,Iobrc,Icbr,Icbrc,Ii