import numpy as np
import jax.numpy as jnp
import tensorly as tl
import tensorly.tenalg as tenalg
import solver.spectralmultidomain.hps.cheb_utils as cheb
import matplotlib.pyplot as plt

def interp(solver,p,f,typestr):

    if solver.ndim==2:
        
        return interp_2d(solver,p,f,typestr)
    elif solver.ndim==3:
        
        return interp_3d(solver,p,f,typestr)
    else:
        raise ValueError("ndim must be 2 or 3")
    

def interp_2d(solver,pts,f):
    g = np.zeros(shape=(pts.shape[0],))
    npan_dim = solver.npan_dim
    boxes = construct_boxes_2d(npan_dim,solver.geom)
    ord=[solver.p,solver.p]
    for box in boxes:
        I = idxs_2d(pts,box)
        J = idxs_2d(solver._XXfull,box)
        XX = solver._XXfull[J,:]
        g[I] = local_interp_2d(pts[I,:],f[J],XX,box,ord)
    return g

def interp_3d(solver,pts,f,typestr):
    g = np.zeros(shape=(pts.shape[0],))
    npan_dim = solver.npan_dim
    boxes = construct_boxes_3d(npan_dim,solver.geom)
    ord=[solver.p,solver.p,solver.p]
    if typestr == "hpsalt":
        ord = solver.p
    for box in boxes:
        I = idxs_3d(pts,box)
        J = idxs_3d(solver._XXfull,box)
        XX = solver._XXfull[J,:]
        g[I] = local_interp_3d(pts[I,:],f[J],XX,box,ord,typestr)
    return g



def construct_boxes_2d(npan_dim,geom):
    box = geom.box_geom
    xmin = box[0][0]
    xmax = box[1][0]

    ymin = box[0][1]
    ymax = box[1][1]

    nx = npan_dim[0]
    ny = npan_dim[1]

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny


    # make more efficient?
    boxes = []
    for i in range (nx):
        x0 = xmin+i*dx
        x1 = xmin+(i+1)*dx
        for j in range (ny):
            y0 = ymin+j*dy
            y1 = ymin+(j+1)*dy
            boxes+=[np.array([[x0,y0],[x1,y1]])]
    return boxes


def construct_boxes_3d(npan_dim,geom):
    box = geom.box_geom
    xmin = box[0][0]
    xmax = box[1][0]

    ymin = box[0][1]
    ymax = box[1][1]

    zmin = box[0][2]
    zmax = box[1][2]

    nx = npan_dim[0]
    ny = npan_dim[1]
    nz = npan_dim[2]

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    dz = (zmax-zmin)/nz


    # make more efficient?
    boxes = []
    for i in range (nx):
        x0 = xmin+i*dx
        x1 = xmin+(i+1)*dx
        for j in range (ny):
            y0 = ymin+j*dy
            y1 = ymin+(j+1)*dy
            for k in range (nz):
                z0 = zmin+k*dz
                z1 = zmin+(k+1)*dz
                boxes+=[np.array([[x0,y0,z0],[x1,y1,z1]])]
    return boxes

def idxs_2d(p,box):
    return np.where( (box[0][0]<=p[:,0]) & (box[1][0]>=p[:,0]) & (box[0][1]<=p[:,1]) & (box[1][1]>=p[:,1]))[0]
def idxs_3d(p,box):
    return np.where( (box[0][0]<p[:,0]+1e-10) & (box[1][0]>p[:,0]-1e-10) & (box[0][1]<p[:,1]+1e-10) & (box[1][1]>p[:,1]-1e-10) & (box[0][2]<p[:,2]+1e-10) & (box[1][2]>p[:,2]-1e-10) )[0]


def tucker_tol(Tens,tol):
    T0 = tl.unfold(Tens,0)
    T1 = tl.unfold(Tens,1)
    T2 = tl.unfold(Tens,2)

    U0,s0,_ = np.linalg.svd(T0)
    U1,s1,_ = np.linalg.svd(T1)
    U2,s2,_ = np.linalg.svd(T2)

    k0 = sum(s0>tol)
    k1 = sum(s1>tol)
    k2 = sum(s2>tol)

    core = tenalg.multi_mode_dot(Tens,[U0[:,:k0].T,U1[:,:k1].T,U2[:,:k2].T])
    return core,U0[:,:k0],U1[:,:k1],U2[:,:k2]


def computeTransform(x):
    N = len(x)
    x0 = x[0]
    xN = x[-1]
    aT = lambda p: (p-x0)*(2./(xN-x0))-1.
    return N,aT

def chebInterpFromSamples(xpts,ff,targetpts):
    if (ff.ndim == 1):
        f0 = ff[:,np.newaxis]
    else:
        f0 = ff
    N,aT = computeTransform(xpts)

    DFT = np.fft.fft(np.vstack((f0[::-1,:],f0[1:N-1,:])),axis=0).real / (2 * N - 2)
    coeffs      = DFT[:N,:] * 2
    coeffs[0,:]   /= 2
    coeffs[-1,:]  /= 2
    return np.polynomial.chebyshev.chebval(aT(targetpts), coeffs)

def local_interp_3d(pts,f,XX,box,ord0,typestr):
    if typestr=='hps':
        ord = [ord0[0]+2,ord0[1]+2,ord0[2]+2]
    elif typestr=='hpsalt':
        ord = ord0
    else:
        raise ValueError("solver type not recognized")
    _,I0  = np.unique(XX.round(decimals=10),axis=0,return_index=True)
    f0    = f[I0]
    F = np.reshape(f0,shape=(ord[0],ord[1],ord[2]))
    
    core,U0,U1,U2 = tucker_tol(F,1e-12)
    F_approx = np.zeros(shape =(pts.shape[0],))
    
    xpts = ((cheb.cheb(ord[0])[0]+1)/2.)*(box[1][0]-box[0][0])+box[0][0]
    ypts = ((cheb.cheb(ord[1])[0]+1)/2.)*(box[1][1]-box[0][1])+box[0][1]
    zpts = ((cheb.cheb(ord[2])[0]+1)/2.)*(box[1][2]-box[0][2])+box[0][2]


    cx = chebInterpFromSamples(xpts,U0,pts[:,0]).T
    cy = chebInterpFromSamples(ypts,U1,pts[:,1]).T
    cz = chebInterpFromSamples(zpts,U2,pts[:,2]).T
    
    for k0 in range(core.shape[0]):
        for k1 in range(core.shape[1]):
            for k2 in range(core.shape[2]):
                F_approx+=core[k0,k1,k2]*cx[:,k0]*cy[:,k1]*cz[:,k2]
    return F_approx
def local_interp_2d(pts,f,XX,box,ord0):
    ord = [ord0[0],ord0[1]]
    _,I0  = np.unique(XX,axis=0,return_index=True)
    f0      = f[I0]
    F = np.reshape(f0,shape=(ord[0],ord[1]))
    
    [U,s,Vh]=np.linalg.svd(F)
    F_approx = np.zeros(shape =(pts.shape[0],))
    
    xpts = ((cheb.cheb(ord[0])[0]+1)/2.)*(box[1][0]-box[0][0])+box[0][0]
    ypts = ((cheb.cheb(ord[1])[0]+1)/2.)*(box[1][1]-box[0][1])+box[0][1]
    cx = chebInterpFromSamples(xpts,U,pts[:,0]).T
    cy = chebInterpFromSamples(ypts,Vh.T,pts[:,1]).T
    
    for k0 in range(s.shape[0]):
        F_approx+=s[k0]*cx[:,k0]*cy[:,k0]
    return F_approx