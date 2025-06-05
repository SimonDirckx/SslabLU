import numpy as np
import tensorly as tl
import tensorly.tenalg as tenalg
import jax.numpy as jnp
import hps.geom as hpsGeom
import hps.hps_multidomain as HPS
#functions

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
    DFT = np.fft.fft(np.vstack((f0[::-1,],f0[1:N-1,:])),axis=0).real / (2 * N - 2)
    coeffs      = DFT[:N,:] * 2
    coeffs[0,:]   /= 2
    coeffs[-1,:]  /= 2
    return np.polynomial.chebyshev.chebval(aT(targetpts), coeffs)

def tucker_tol(Tens,tol):
    T0 = tl.unfold(Tens,0)
    T1 = tl.unfold(Tens,1)
    T2 = tl.unfold(Tens,2)

    U0,s0,V0h = np.linalg.svd(T0)
    U1,s1,V1h = np.linalg.svd(T1)
    U2,s2,V2h = np.linalg.svd(T2)

    k0 = sum(s0>tol)
    k1 = sum(s1>tol)
    k2 = sum(s2>tol)

    core = tenalg.multi_mode_dot(Tens,[U0[:,:k0].T,U1[:,:k1].T,U2[:,:k2].T])
    return core,U0[:,:k0],U1[:,:k1],U2[:,:k2]


def chebInterpFromSamples3D(xpts,ypts,zpts,f,XY):
    F = f(xpts,ypts,zpts)
    core,U0,U1,U2 = tucker_tol(F,1e-12)
    F_approx = np.zeros(shape =(XY.shape[0],))
    cx = chebInterpFromSamples(xpts,U0,XY[:,0])
    cy = chebInterpFromSamples(ypts,U1,XY[:,1])
    cz = chebInterpFromSamples(zpts,U2,XY[:,2])
    for k0 in range(core.shape[0]):
        for k1 in range(core.shape[1]):
            for k2 in range(core.shape[2]):
                F_approx+=core[k0,k1,k2]*cx[k0,:]*cy[k1,:]*cz[k2,:]
    return F_approx

def chebInterpFromSamples3D_XX(xyzpts,p,f,XY):

    xpts0 = np.cos(np.arange(p+2) * np.pi / (p + 1))
    ypts0 = np.cos(np.arange(p+2) * np.pi / (p + 1))
    zpts0 = np.cos(np.arange(p+2) * np.pi / (p + 1))
    xpts0 = xpts0[::-1]
    ypts0 = ypts0[::-1]
    zpts0 = zpts0[::-1]
    xmin = min(xyzpts[:,0])
    xmax = max(xyzpts[:,0])
    ymin = min(xyzpts[:,1])
    ymax = max(xyzpts[:,1])
    zmin = min(xyzpts[:,2])
    zmax = max(xyzpts[:,2])


    xpts = (xpts0+1)*(xmax-xmin)/2.+xmin
    ypts = (ypts0+1)*(ymax-ymin)/2.+ymin
    zpts = (zpts0+1)*(zmax-zmin)/2.+zmin

    F = np.reshape(f,newshape=(p+2,p+2,p+2))
    core,U0,U1,U2 = tucker_tol(F,1e-10)
    F_approx = np.zeros(shape =(XY.shape[0],))
    cx = chebInterpFromSamples(xpts,U0,XY[:,0])
    cy = chebInterpFromSamples(ypts,U1,XY[:,1])
    cz = chebInterpFromSamples(zpts,U2,XY[:,2])
    for k0 in range(core.shape[0]):
        for k1 in range(core.shape[1]):
            for k2 in range(core.shape[2]):
                F_approx+=core[k0,k1,k2]*cx[k0,:]*cy[k1,:]*cz[k2,:]
    return F_approx

def sortInHPSBoxes(geom,npan_dim,XY):
    nx = npan_dim[0]
    ny = npan_dim[1]
    nz = npan_dim[2]
    XYlist =[]
    xmin = geom.bounds[0][0]
    xmax = geom.bounds[1][0]
    ymin = geom.bounds[0][1]
    ymax = geom.bounds[1][1]
    zmin = geom.bounds[0][2]
    zmax = geom.bounds[1][2]
    dx = xmax-xmin
    dy = ymax-ymin
    dz = zmax-zmin
    for i in range(nx):
        for j in range(ny):
            for j in range(nz):
                XYlist+=[np.zeros(shape=(0,3))]
    for i in range(XY.shape[0]):
        x,y,z=XY[i,:]
        x = (x-xmin)/dx
        y = (y-ymin)/dy
        z = (z-zmin)/dz
        xmod = max((int)(np.ceil(x*nx))-1,0)
        ymod = max((int)(np.ceil(y*ny))-1,0)
        zmod = max((int)(np.ceil(z*nz))-1,0)
        xyz = np.reshape(XY[i,:],newshape=(1,3))
        XYlist[xmod+ymod*nx+zmod*nx*ny] = np.append(XYlist[xmod+ymod*nx+zmod*nx*ny],xyz,axis=0)
    return XYlist

def interpHPS(XX,geom,npan_dim,a,p,vals,XY):
    XYlist = sortInHPSBoxes(geom,npan_dim,XY)
    ndofs = (p+2)*(p+2)*(p+2)
    F_approx = np.zeros(shape=(0,1))
    for i in range(len(XYlist)):
        ff = chebInterpFromSamples3D_XX(XX[ndofs*i:ndofs*(i+1),:],p,vals[ndofs*i:ndofs*(i+1)],XYlist[i])
        F_approx= np.append(F_approx,ff)
    return F_approx,XYlist


def check_err(slab,ul,ur,a,p,pdo,gb,bc,u_exact):
    
    xl = slab[0][0]
    xr = slab[1][0]

    yl = slab[0][1]
    yr = slab[1][1]

    zl = slab[0][2]
    zr = slab[1][2]
    
    geom = hpsGeom.BoxGeometry(jnp.array([[xl,yl,zl],[xr,yr,zr]]))
    disc = HPS.HPSMultidomain(pdo, geom, a, p)
    
    XXb = disc._XX[disc.Jx,:]
    Ir = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xr)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Il = [i for i in range(len(disc.Jx)) if np.abs(XXb[i,0]-xl)<1e-10 and XXb[i,1]>1e-10 and XXb[i,1]<1-1e-10]
    Igb = [i for i in range(len(disc.Jx)) if gb(XXb[i,:])]
    
    bvec = np.zeros(shape=(len(disc.Jx),1))
    bvec[Igb,0] = bc(XXb[Igb,:])
    
    bvec[Il,0] = ul   
    bvec[Ir,0] = ur
    
    print("solving local dirichlet...")
    ui = disc.solve_dir_full(bvec)
    print("done")
    resx = 50
    resy = 30
    x_eval = np.linspace(disc._box_geom[0][0],disc._box_geom[1][0],resx)
    y_eval = np.linspace(disc._box_geom[0][1],disc._box_geom[1][1],resy)

    XY = np.zeros(shape=(resx*resy,3))
    XY[:,0] = np.kron(x_eval,np.ones(shape=y_eval.shape))
    XY[:,1] = np.kron(np.ones(shape=x_eval.shape),y_eval)
    XY[:,2] = .6*np.ones(shape = (resx*resy,))
    print("interpolating...")
    XXfull = np.array(disc._XXfull)
    npan_dim=disc.npan_dim
    u_approx,XYlist = interpHPS(XXfull,geom,npan_dim,a,p,ui[:,0],XY)
    print("done")
    u_exact_vec = np.zeros(shape=(0,1))
    for i in range(len(XYlist)):
        ue = u_exact(XYlist[i])
        u_exact_vec= np.append(u_exact_vec,ue)
    errInf = np.linalg.norm(u_exact_vec-u_approx,ord=np.inf)
    print('errInf = ',errInf)