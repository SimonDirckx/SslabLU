import numpy as np


#functions

def computeTransform(x):
    N = len(x)
    x0 = x[0]
    xN = x[-1]
    aT = lambda p: (p-x0)*(2./(xN-x0))-1.
    return N,aT

def chebInterpFromSamples(xpts,fvec,targetpts):
    N,aT = computeTransform(xpts)
    f = fvec[::-1] #fvec is in reverse order, technically
    DFT = np.fft.fft(np.hstack((f, f[N - 2: 0: -1]))).real / (2 * N - 2)
    coeffs      = DFT[:N] * 2
    coeffs[0]   /= 2
    coeffs[-1]  /= 2
    return np.polynomial.chebyshev.chebval(aT(targetpts), coeffs)

def chebInterpFromSamples2D(xpts,ypts,f,XY):
    F = f(xpts,ypts)
    [U,s,Vh] = np.linalg.svd(F.T)
    s = s[s>s[0]*(1e-10)]
    rk = len(s)
    U=U[:,0:rk]
    V=Vh[0:rk,:].T
    F_approx = np.zeros(shape =(XY.shape[0],))
    for k in range(rk):
        F_approx+=chebInterpFromSamples(xpts,U[:,k],XY[:,0])*s[k]*chebInterpFromSamples(ypts,V[:,k],XY[:,1])
    return F_approx

def chebInterpFromSamples2D(xpts,ypts,f,XY):
    F = f(xpts,ypts)
    [U,s,Vh] = np.linalg.svd(F.T)
    s = s[s>s[0]*(1e-10)]
    rk = len(s)
    U=U[:,0:rk]
    V=Vh[0:rk,:].T
    F_approx = np.zeros(shape =(XY.shape[0],))
    for k in range(rk):
        F_approx+=chebInterpFromSamples(xpts,U[:,k],XY[:,0])*s[k]*chebInterpFromSamples(ypts,V[:,k],XY[:,1])
    return F_approx
def chebInterpFromSamples2D_XX(xypts,p,f,XY):

    xpts0 = np.cos(np.arange(p+2) * np.pi / (p + 1))
    ypts0 = np.cos(np.arange(p+2) * np.pi / (p + 1))
    xpts0 = xpts0[::-1]
    ypts0 = ypts0[::-1]
    xmin = min(xypts[:,0])
    xmax = max(xypts[:,0])
    ymin = min(xypts[:,1])
    ymax = max(xypts[:,1])


    xpts = (xpts0+1)*(xmax-xmin)/2.+xmin
    ypts = (ypts0+1)*(ymax-ymin)/2.+ymin

    F = np.reshape(f,newshape=(p+2,p+2))
    [U,s,Vh] = np.linalg.svd(F)
    s = s[s>s[0]*(1e-10)]
    rk = len(s)
    U=U[:,0:rk]
    V=Vh[0:rk,:].T
    F_approx = np.zeros(shape =(XY.shape[0],))
    for k in range(rk):
        F_approx+=chebInterpFromSamples(xpts,U[:,k],XY[:,0])*s[k]*chebInterpFromSamples(ypts,V[:,k],XY[:,1])
    return F_approx

def sortInHPSBoxes(disc,XY):
    npan_dim = disc.npan_dim
    nx = npan_dim[0]
    ny = npan_dim[1]
    XYlist =[]
    xmin = disc._box_geom[0][0]
    xmax = disc._box_geom[1][0]
    ymin = disc._box_geom[0][1]
    ymax = disc._box_geom[1][1]
    dx = xmax-xmin
    dy = ymax-ymin
    for i in range(nx):
        for j in range(ny):
            XYlist+=[np.zeros(shape=(0,2))]
    for i in range(XY.shape[0]):
        x,y=XY[i,:]
        x = (x-xmin)/dx
        y = (y-ymin)/dy
        xmod = max((int)(np.ceil(x*nx))-1,0)
        ymod = max((int)(np.ceil(y*ny))-1,0)
        xy = np.reshape(XY[i,:],newshape=(1,2))
        XYlist[xmod+ymod*nx] = np.append(XYlist[xmod+ymod*nx],xy,axis=0)
    return XYlist

def interpHPS(disc,vals,XY):
    assert(vals.shape[0]==disc._XXfull.shape[0])
    p = disc._p
    XX = disc._XXfull
    XYlist = sortInHPSBoxes(disc,XY)
    ndofs = (p+2)*(p+2)
    F_approx = np.zeros(shape=(0,1))
    for i in range(len(XYlist)):
        ff = chebInterpFromSamples2D_XX(XX[ndofs*i:ndofs*(i+1),:],p,vals[ndofs*i:ndofs*(i+1)],XYlist[i])
        F_approx= np.append(F_approx,ff)
    return F_approx,XYlist