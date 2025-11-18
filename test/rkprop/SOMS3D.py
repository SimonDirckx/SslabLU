import numpy as np
import solver.spectral.spectralSolver as spectral
import numpy.polynomial.chebyshev as chebpoly
import matplotlib.pyplot as plt
import scipy.linalg as sclinalg
import time

kh = 5.

def bc(p,kh):
    r=np.sqrt((p[:,0]+.1)**2+(p[:,1]+.1)**2+(p[:,2]+.1)**2)
    return np.sin(kh*r)/(4.*np.pi*r)

def L_op(dir,px,py,pz,scl_x,scl_y,scl_z,kh):
    """
    Construct differential operator for 2x1 joined block set-up for the Helmholtz equation at kappa = kh

    ----------------------------------------------------------------------------------------
    | 'joined' refers to (attributes of) block with the dimensions of two 1x1 blocks glued |
    | together BUT                                                                         |
    | with the DOFS on the faces along the glued dimension NOT two tiles of 2-d cheb pts,  |
    | but instead one 2-d cheb grid                                                        |
    ----------------------------------------------------------------------------------------

    INPUT:  - dir (in [0,1,2])      :    whether joined block is in x-dir (0), y-dir (1), z-dir (2)
            - (px,py,pz)            :    polynomial orders for each 1x1 block in the 2x1 set-up
            - (scl_x,scl_y,scl_z)   :    length scales of the 1x1 blocks
            - kh                    :    wave number
    
    OUTPUT: - L_joined_dir                  : diff op for the joined block
            - Ijl,Ijf,Ijd,Iju,Ijb,Ijr       : joined DOFS corresponding to left, front, down, up, back and right face (as subsets of boundary dofs)
            - Ic_dir                        : joined interior DOFs corresponding to the interior central interface
            - Ibox_joined_dir               : union of Ijl,Ijf,Ijd,Iju,Ijb,Ijr, in that order
            - Ii_dir,Ib_dir                 : interior and boundary dofs for the joined block  
            - XY_joined_dir                 : spatial locations of the total DOFs
    
    """



    Dx,xpts = spectral.cheb(px)
    Dy,ypts = spectral.cheb(py)
    Dz,zpts = spectral.cheb(pz)

    Dx = - 2*Dx/scl_x
    Dy = - 2*Dy/scl_y
    Dz = - 2*Dz/scl_z

    xpts = ((xpts[::-1]+1)/2)*scl_x
    ypts = ((ypts[::-1]+1)/2)*scl_y
    zpts = ((zpts[::-1]+1)/2)*scl_z


    nx = len(xpts)
    ny = len(ypts)
    nz = len(zpts)


    px_joined = (3*px)//2
    py_joined = (3*py)//2
    pz_joined = (3*pz)//2

    px_joined = px_joined-px_joined%2
    py_joined = py_joined-py_joined%2
    pz_joined = pz_joined-pz_joined%2


    Dx2,xpts2 = spectral.cheb(px_joined)
    Dy2,ypts2 = spectral.cheb(py_joined)
    Dz2,zpts2 = spectral.cheb(pz_joined)
    xpts2 = (1+xpts2[::-1])*scl_x
    ypts2 = (1+ypts2[::-1])*scl_y
    zpts2 = (1+zpts2[::-1])*scl_z


    Dx2 = - Dx2/scl_x
    Dy2 = - Dy2/scl_y
    Dz2 = - Dz2/scl_z

    if dir == 0:
        xjoined = xpts2
        yjoined = ypts
        zjoined = zpts
        Dx = Dx2
        xlim = 2*scl_x
        xc = scl_x
        ylim = scl_y
        zlim = scl_z
    elif dir == 1:
        xjoined = xpts
        yjoined = ypts2
        zjoined = zpts
        Dy = Dy2
        xlim = scl_x
        ylim = 2*scl_y
        xc = scl_y
        zlim = scl_z
    elif dir == 2:
        xjoined = xpts
        yjoined = ypts
        zjoined = zpts2
        Dz = Dz2
        xlim = scl_x
        ylim = scl_y
        zlim = 2*scl_z
        xc = scl_z
    njx = len(xjoined)
    njy = len(yjoined)
    njz = len(zjoined)

    XY_joined_dir = np.zeros(shape=(njx*njy*njz,3))
    XY_joined_dir[:,0] = np.kron(np.kron(xjoined,np.ones_like(yjoined)),np.ones_like(zjoined))
    XY_joined_dir[:,1] = np.kron(np.kron(np.ones_like(xjoined),yjoined),np.ones_like(zjoined))
    XY_joined_dir[:,2] = np.kron(np.kron(np.ones_like(xjoined),np.ones_like(yjoined)),zjoined)

    Ii_dir = np.where((XY_joined_dir[:,0]>0) & (XY_joined_dir[:,0]<xlim) & (XY_joined_dir[:,1]>0) & (XY_joined_dir[:,1]<ylim) & (XY_joined_dir[:,2]>0) & (XY_joined_dir[:,2]<zlim) )[0]
    Ib_dir = np.where((np.abs(XY_joined_dir[:,0])<1e-10) | (np.abs(XY_joined_dir[:,0]-xlim)<1e-10) | (np.abs(XY_joined_dir[:,1])<1e-10) | (np.abs(XY_joined_dir[:,1]-ylim)<1e-10) | (np.abs(XY_joined_dir[:,2])<1e-10) | (np.abs(XY_joined_dir[:,2]-zlim)<1e-10) )[0]

    XY_joined_dir_i = XY_joined_dir[Ii_dir,:]
    XY_joined_dir_b = XY_joined_dir[Ib_dir,:]

    Il_dir = np.where( (np.abs(XY_joined_dir_b[:,0])<1e-10) & (XY_joined_dir_b[:,1]>0) & (XY_joined_dir_b[:,1]<ylim) & (XY_joined_dir_b[:,2]>0) & (XY_joined_dir_b[:,2]<zlim) )[0]
    Ir_dir = np.where( (np.abs(XY_joined_dir_b[:,0]-xlim)<1e-10) & (XY_joined_dir_b[:,1]>0) & (XY_joined_dir_b[:,1]<ylim) & (XY_joined_dir_b[:,2]>0) & (XY_joined_dir_b[:,2]<zlim) )[0]

    If_dir = np.where( (np.abs(XY_joined_dir_b[:,1])<1e-10) & (XY_joined_dir_b[:,0]>0) & (XY_joined_dir_b[:,0]<xlim) & (XY_joined_dir_b[:,2]>0) & (XY_joined_dir_b[:,2]<zlim) )[0]
    Ibk_dir = np.where( (np.abs(XY_joined_dir_b[:,1]-ylim)<1e-10) & (XY_joined_dir_b[:,0]>0) & (XY_joined_dir_b[:,0]<xlim) & (XY_joined_dir_b[:,2]>0) & (XY_joined_dir_b[:,2]<zlim) )[0]

    Id_dir = np.where( (np.abs(XY_joined_dir_b[:,2])<1e-10) & (XY_joined_dir_b[:,0]>0) & (XY_joined_dir_b[:,0]<xlim) & (XY_joined_dir_b[:,1]>0) & (XY_joined_dir_b[:,1]<ylim) )[0]
    Iu_dir = np.where( (np.abs(XY_joined_dir_b[:,2]-zlim)<1e-10) & (XY_joined_dir_b[:,0]>0) & (XY_joined_dir_b[:,0]<xlim) & (XY_joined_dir_b[:,1]>0) & (XY_joined_dir_b[:,1]<ylim) )[0]

    Ic_dir = np.where( (np.abs(XY_joined_dir_i[:,dir]-xc)<1e-10))[0]

    Ibox_joined_dir = np.append(Il_dir,If_dir)
    Ibox_joined_dir = np.append(Ibox_joined_dir,Id_dir)
    Ibox_joined_dir = np.append(Ibox_joined_dir,Iu_dir)
    Ibox_joined_dir = np.append(Ibox_joined_dir,Ibk_dir)
    Ibox_joined_dir = np.append(Ibox_joined_dir,Ir_dir)

    ctr = 0
    Ijl = np.arange(ctr,ctr+len(Il_dir))
    ctr+= len(Il_dir)
    Ijf = np.arange(ctr,ctr+len(If_dir))
    ctr+= len(If_dir)
    Ijd = np.arange(ctr,ctr+len(Id_dir))
    ctr+= len(Id_dir)
    Iju = np.arange(ctr,ctr+len(Iu_dir))
    ctr+= len(Iu_dir)
    Ijb = np.arange(ctr,ctr+len(Ibk_dir))
    ctr+= len(Ibk_dir)
    Ijr = np.arange(ctr,ctr+len(Ir_dir))


    Dxx = Dx@Dx
    Dyy = Dy@Dy
    Dzz = Dz@Dz

    L_joined_dir = -np.kron(np.kron(Dxx,np.identity(njy)),np.identity(njz))-np.kron(np.kron(np.identity(njx),Dyy),np.identity(njz))-np.kron(np.kron(np.identity(njx),np.identity(njy)),Dzz)-kh*kh*np.kron(np.kron(np.identity(njx),np.identity(njy)),np.identity(njz))
    return L_joined_dir,Ijl,Ijf,Ijd,Iju,Ijb,Ijr,Ic_dir,Ibox_joined_dir,Ii_dir,Ib_dir,XY_joined_dir



def interp_ops(px,py,pz,scl_x,scl_y,scl_z):


    """
    Construct interpolation operators from joined dofs to unjoined dofs

    ----------------------------------------------------------------------------------------
    | 'joined' refers to (attributes of) block with the dimensions of two 1x1 blocks glued |
    | together BUT                                                                         |
    | with the DOFS on the faces along the glued dimension NOT two tiles of 2-d cheb pts,  |
    | but instead one 2-d cheb grid                                                        |
    ----------------------------------------------------------------------------------------

    INPUT:  - (px,py,pz)            :    polynomial orders for each 1x1 block in the 2x1 set-up
            - (scl_x,scl_y,scl_z)   :    length scales of the 1x1 blocks
    
    OUTPUT: - Interp_x,Interp_y,Interp_z    : Interpolation operators associated to (px,scl_x) etc


    DESCRIPTION:
                        
                                              Interp                         
    |* *  *   *   *  * *|* *  *   *   *  * *| <-------------------------------- |* *  *   *    *    *    *    *   *  * *|
            (scl,p)             (scl,p)                                                         (2*scl,3*p/2)
    

    Interp is the map that takes ~3*p/2 cheb grid sample on the interval [0,2*scl] 
    and interpolates underlying poly to the two pieces of p cheb pts, on [0,scl] and [scl,2*scl] respectively 

    """



    # sorry for this notation, it sucks
    # x2 = 2 sets of p-1 cheb pts stuck next to each other
    # xpts2 = 1 set op pjoined-1 cheb pts on the double-wide interval

    Dx,xpts = spectral.cheb(px)
    Dy,ypts = spectral.cheb(py)
    Dz,zpts = spectral.cheb(pz)

    Dx = - 2*Dx/scl_x
    Dy = - 2*Dy/scl_y
    Dz = - 2*Dz/scl_z

    xpts = ((xpts[::-1]+1)/2)*scl_x
    ypts = ((ypts[::-1]+1)/2)*scl_y
    zpts = ((zpts[::-1]+1)/2)*scl_z


    nx = len(xpts)
    ny = len(ypts)
    nz = len(zpts)


    x2 = np.append(xpts[1:nx-1],scl_x+xpts[1:nx-1])
    y2 = np.append(ypts[1:ny-1],scl_y+ypts[1:ny-1])
    z2 = np.append(zpts[1:nz-1],scl_z+zpts[1:nz-1])

    nx2 = len(x2)
    ny2 = len(y2)
    nz2 = len(z2)


    px_joined = (3*px)//2
    py_joined = (3*py)//2
    pz_joined = (3*pz)//2

    px_joined = px_joined-px_joined%2
    py_joined = py_joined-py_joined%2
    pz_joined = pz_joined-pz_joined%2


    Dx2,xpts2 = spectral.cheb(px_joined)
    Dy2,ypts2 = spectral.cheb(py_joined)
    Dz2,zpts2 = spectral.cheb(pz_joined)
    xpts2 = (1+xpts2[::-1])*scl_x
    ypts2 = (1+ypts2[::-1])*scl_y
    zpts2 = (1+zpts2[::-1])*scl_z


    Dx2 = - Dx2/scl_x
    Dy2 = - Dy2/scl_y
    Dz2 = - Dz2/scl_z

    nxpts2 = len(xpts2)
    nypts2 = len(ypts2)
    nzpts2 = len(zpts2)


    xpts2 = xpts2[1:nxpts2-1]
    ypts2 = ypts2[1:nypts2-1]
    zpts2 = zpts2[1:nzpts2-1]

    nxpts2 = len(xpts2)
    nypts2 = len(ypts2)
    nzpts2 = len(zpts2)


    # E is evaluation from cheb coeffs to points
    # Interpolation map is now E_x2*E_xpts2^-1
    # you can check that this is correct by chasing diagrams:
    #       E_x2        E_xpts2
    # x2 <----- coeffs ---------> xpts2
    #   <-------------------------|
    #           Interp_x

    E_x2 = np.zeros(shape = (nx2,nxpts2))
    E_y2 = np.zeros(shape = (ny2,nypts2))
    E_z2 = np.zeros(shape = (nz2,nzpts2))
    
    E_xpts2 = np.zeros(shape = (nxpts2,nxpts2))
    E_ypts2 = np.zeros(shape = (nypts2,nypts2))
    E_zpts2 = np.zeros(shape = (nzpts2,nzpts2))


    for indcoeff in range(nxpts2):
        ci = np.zeros(shape = (nxpts2,))
        ci[indcoeff] = 1.
        Ti = chebpoly.Chebyshev(ci,domain=[0,2*scl_x])
        E_x2[:,indcoeff] = Ti(x2)
        E_xpts2[:,indcoeff] = Ti(xpts2)
    for indcoeff in range(nypts2):
        ci = np.zeros(shape = (nypts2,))
        ci[indcoeff] = 1.
        Ti = chebpoly.Chebyshev(ci,domain=[0,2*scl_y])
        E_y2[:,indcoeff] = Ti(y2)
        E_ypts2[:,indcoeff] = Ti(ypts2)
    for indcoeff in range(nzpts2):
        ci = np.zeros(shape = (nzpts2,))
        ci[indcoeff] = 1.
        Ti = chebpoly.Chebyshev(ci,domain=[0,2*scl_z])
        E_z2[:,indcoeff] = Ti(z2)
        E_zpts2[:,indcoeff] = Ti(zpts2)



    Interp_x = np.linalg.solve(E_xpts2.T,E_x2.T).T
    Interp_y = np.linalg.solve(E_ypts2.T,E_y2.T).T
    Interp_z = np.linalg.solve(E_zpts2.T,E_z2.T).T
    return Interp_x,Interp_y,Interp_z

def XYU(dir,px,py,pz,scl_x,scl_y,scl_z):


    """
    Construct unjoined DOFs

    ----------------------------------------------------------------------------------------
    | 'unjoined' refers to (attributes of) two 1x1 blocks stuck together along a direction |
    | with their touching face DOFs identified                                             |
    ----------------------------------------------------------------------------------------

    INPUT:  - dir (in [0,1,2])      :    whether two blocks are stacked together in x-dir (0), y-dir (1), z-dir (2)
            - (px,py,pz)            :    polynomial orders for each 1x1 block in the 2x1 set-up
            - (scl_x,scl_y,scl_z)   :    length scales of the 1x1 blocks
    
    OUTPUT: - XYu                       :   spatial coordinates of the unjoined DOFs
            - Iul,Iuf,Iud,Iuu,Iub,Iur   :   faces (left, front, down, up, back, right) as subsets of the boundary


    NOTE: edges of the faces are removed

    """


    Dx,xpts = spectral.cheb(px)
    Dy,ypts = spectral.cheb(py)
    Dz,zpts = spectral.cheb(pz)

    Dx = - 2*Dx/scl_x
    Dy = - 2*Dy/scl_y
    Dz = - 2*Dz/scl_z

    xpts = ((xpts[::-1]+1)/2)*scl_x
    ypts = ((ypts[::-1]+1)/2)*scl_y
    zpts = ((zpts[::-1]+1)/2)*scl_z


    nx = len(xpts)
    ny = len(ypts)
    nz = len(zpts)

    xpts = xpts[1:nx-1]
    ypts = ypts[1:ny-1]
    zpts = zpts[1:nz-1]

    

    if dir == 0:
        xpts = np.append(xpts,scl_x+xpts)
    if dir == 1:
        ypts = np.append(ypts,scl_y+ypts)
    if dir == 2:
        zpts = np.append(zpts,scl_z+zpts)
    nx = len(xpts)
    ny = len(ypts)
    nz = len(zpts)

    #left
    XYul = np.zeros(shape = (ny*nz,3))
    XYul[:,1] = np.kron(ypts,np.ones_like(zpts))
    XYul[:,2] = np.kron(np.ones_like(ypts),zpts)
    


    #front
    XYuf = np.zeros(shape = (nx*nz,3))
    XYuf[:,0] = np.kron(xpts,np.ones_like(zpts))
    XYuf[:,2] = np.kron(np.ones_like(xpts),zpts)

    # down

    XYud = np.zeros(shape = (nx*ny,3))
    XYud[:,0] = np.kron(xpts,np.ones_like(ypts))
    XYud[:,1] = np.kron(np.ones_like(xpts),ypts)

    # up
    XYuu = XYud+scl_z*np.array([0,0,1])
    if dir == 2:
        XYuu = XYuu+scl_z*np.array([0,0,1])


    # back
    XYub = XYuf+scl_y*np.array([0,1,0])
    if dir == 1:
        XYub = XYub+scl_y*np.array([0,1,0])

    # right
    XYur = XYul+scl_x*np.array([1,0,0])
    if dir == 0:
        XYur = XYur+scl_x*np.array([1,0,0])

    XYu=np.zeros(shape=(0,3))
    XYu = np.append(XYul,XYuf,axis=0)
    XYu = np.append(XYu,XYud,axis=0)
    XYu = np.append(XYu,XYuu,axis=0)
    XYu = np.append(XYu,XYub,axis=0)
    XYu = np.append(XYu,XYur,axis=0)
    ctr = 0
    Iul = np.arange(ctr,ctr+XYul.shape[0])
    ctr+= XYul.shape[0]
    Iuf = np.arange(ctr,ctr+XYuf.shape[0])
    ctr+= XYuf.shape[0]
    Iud = np.arange(ctr,ctr+XYud.shape[0])
    ctr+= XYud.shape[0]
    Iuu = np.arange(ctr,ctr+XYuu.shape[0])
    ctr+= XYuu.shape[0]
    Iub = np.arange(ctr,ctr+XYub.shape[0])
    ctr+= XYub.shape[0]
    Iur = np.arange(ctr,ctr+XYur.shape[0])
    return XYu,Iul,Iuf,Iud,Iuu,Iub,Iur

def global_dofs(tiling,px,py,pz,Lx,Ly,Lz):
    """
    Construct global reduced DOFs

    ----------------------------------------------------------------------------------------
    | 'reduced' the union of all cuboid boundaries in the provided tiling i.e.             |
    |  the DOFs interior to the cuboids are removed                                        |
    ----------------------------------------------------------------------------------------

    INPUT:  - tiling (1x3 int array)    :   the tiling defining the underlying non-overlapping domain decomp
            - (px,py,pz)                :    polynomial orders for each 1x1 block
            - (Lx,Ly,Lz)                :    length scales of the global domain
    
    OUTPUT: - XYtot                     :   spatial coordinates of the global DOFs
            - md_vec                    :   int vec with length the number of cuboid faces
                                            'missing direction vector', equals 2 for an xy face, 1 for an xz face, 0 for a yz face
            - b_vec                     :   bool vec with length the number of cuboid faces
                                            'boundary vec', 1 for a face on the global boundary, zero otherwise
            -nxy,nyz,nxz                :   number of points per face of each cuboid
                                            for example, the bottom and top face have nxy points etc.
            -indx_vec,indy_vec,indz_vec :   int vecs with length the number of cuboid faces
                                            correspond to the loop indexes at which face is added



    """
    Lx0 = tiling[0]
    Ly0 = tiling[1]
    Lz0 = tiling[2]

    scl_z = Lz/Lz0
    scl_y = Ly/Ly0
    scl_x = Lx/Lx0
    _,xpts = spectral.cheb(px)
    _,ypts = spectral.cheb(py)
    _,zpts = spectral.cheb(pz)
    xpts = ((xpts[::-1]+1)/2)*scl_x
    ypts = ((ypts[::-1]+1)/2)*scl_y
    zpts = ((zpts[::-1]+1)/2)*scl_z


    nx = len(xpts)
    ny = len(ypts)
    nz = len(zpts)

    xpts = xpts[1:nx-1]
    ypts = ypts[1:ny-1]
    zpts = zpts[1:nz-1]

    xy = np.zeros(shape = ((nx-2)*(ny-2),3))
    yz = np.zeros(shape = ((ny-2)*(nz-2),3))
    xz = np.zeros(shape = ((nx-2)*(nz-2),3))
    
    xy[:,0] = np.kron(xpts,np.ones_like(ypts))
    xy[:,1] = np.kron(np.ones_like(xpts),ypts)

    yz[:,1] = np.kron(ypts,np.ones_like(zpts))
    yz[:,2] = np.kron(np.ones_like(ypts),zpts)

    xz[:,0] = np.kron(xpts,np.ones_like(zpts))
    xz[:,2] = np.kron(np.ones_like(xpts),zpts)
    
    nxy = xy.shape[0]
    nyz = yz.shape[0]
    nxz = xz.shape[0]
    md_vec=np.zeros(shape=(0,),dtype = np.int8)
    b_vec=np.zeros(shape=(0,),dtype = np.bool)
    indx_vec=np.zeros(shape=(0,),dtype = np.int64)
    indy_vec=np.zeros(shape=(0,),dtype = np.int64)
    indz_vec=np.zeros(shape=(0,),dtype = np.int64)
    
    XYtot = np.zeros(shape=(0,3))
    for indx in range(tiling[0]+1):
        for indy in range(tiling[1]):
            for indz in range(tiling[2]):
            
                XYtot = np.append(XYtot,yz+np.array([indx*scl_x,indy*scl_y,indz*scl_z]),axis=0)
                md_vec=np.append(md_vec,0)
                b_bool = (indx == 0) | (indx == tiling[0])
                b_vec = np.append(b_vec,b_bool)
                indx_vec=np.append(indx_vec,indx)
                indy_vec=np.append(indy_vec,indy)
                indz_vec=np.append(indz_vec,indz)
                
        for indy in range(tiling[1]+1):
            for indz in range(tiling[2]):
                if indx<tiling[0]:
                    XYtot = np.append(XYtot,xz+np.array([indx*scl_x,indy*scl_y,indz*scl_z]),axis=0)
                    md_vec=np.append(md_vec,1)
                    b_bool = (indy == 0) | (indy == tiling[1])
                    b_vec = np.append(b_vec,b_bool)
                    indx_vec=np.append(indx_vec,indx)
                    indy_vec=np.append(indy_vec,indy)
                    indz_vec=np.append(indz_vec,indz)
                    
            for indz in range(tiling[2]+1):
                if indy<tiling[1] and indx<tiling[0]:
                    XYtot = np.append(XYtot,xy+np.array([indx*scl_x,indy*scl_y,indz*scl_z]),axis=0)
                    md_vec=np.append(md_vec,2)
                    b_bool = (indz == 0) | (indz == tiling[2])
                    b_vec = np.append(b_vec,b_bool)
                    indx_vec=np.append(indx_vec,indx)
                    indy_vec=np.append(indy_vec,indy)
                    indz_vec=np.append(indz_vec,indz)
                    
    return XYtot,md_vec,b_vec,nxy,nyz,nxz,indx_vec,indy_vec,indz_vec

def construct_SOMS(nxy,nyz,nxz,md_vec,b_vec,XYtot,tiling,indx_vec,indy_vec,indz_vec,uXY,Sx,Sy,Sz):
    """
    Construct global SOMS system

    
    INPUT:  - nxy,nyz,nxz               :   number of points for the xy-faces (down and up) etc.
            - md_vec                    :   vec indication direction of each face (xy=2,xz=1,yz=0)
            - b_vec                     :   vec indicating if corresponding face is on global boundary
            - XYtot                     :   spatial locations of global DOFs
            - tiling                    :   underlying non-overlapping domain decomp tiling
            - indx_vec,indy_vec,indz_vec:   hard to explain, sorry
            - uXY                       :   solution on XYtot, for testing purposes
            - Sx,Sy,Sz                  :   Local solve-and-restrict maps, one for each direction
    
    OUTPUT: - Stot                      :   total S system



    """
    ctr = 0
    max_block_err = 0.
    nFYZ = tiling[1]*tiling[2]*nyz
    nFXZ = tiling[2]*nxz
    nFXY = (tiling[2]+1)*nxy

    Stot = np.identity(XYtot.shape[0])
    for indxyz in range(len(md_vec)):
        source = np.zeros(shape=(0,),dtype = np.int64)
        match md_vec[indxyz]:
            case 2:
                target = np.arange(ctr,ctr+nxy)
                source = np.zeros(shape = (0,),dtype=np.int64)
        
                step_up     = nxy
                step_down   = nxy
                step_bk  = (tiling[2]+1-indz_vec[indxyz])*nxy+(indz_vec[indxyz]-1)*nxz
                step_front  = (indz_vec[indxyz])*nxy+(tiling[2]+1-indz_vec[indxyz])*nxz

                step_right =  (tiling[1]-indy_vec[indxyz])*tiling[2]*nxz+(tiling[2]+1)*(tiling[1]-indy_vec[indxyz]-1)*nxy+ nxy*(tiling[2]+1-indz_vec[indxyz])+(indz_vec[indxyz]-1)*nyz + (indy_vec[indxyz])*tiling[2]*nyz
                start_left =  ctr+step_right - tiling[2]*tiling[1]*nyz - (tiling[2]+1)*tiling[1]*nxy - (tiling[1]+1)*tiling[2]*nxz

                source = np.append(source,np.arange(start_left,start_left+2*nyz))
                source = np.append(source,np.arange(ctr-step_front,ctr-step_front+2*nxz))
                source = np.append(source,np.arange(ctr-step_down,ctr-step_down+nxy))
                source = np.append(source,np.arange(ctr+step_up,ctr+step_up+nxy))
                source = np.append(source,np.arange(ctr+step_bk,ctr+step_bk+2*nxz))
                source = np.append(source,np.arange(ctr+step_right,ctr+step_right+2*nyz))
                ctr += nxy

                
                
                if not b_vec[indxyz]:
                    Stot[np.ix_(target,source)]=-Sz
                    ub_loc = uXY[source]
                    uc_loc = uXY[target]
                    max_block_err = max(max_block_err,np.linalg.norm(uc_loc-Sz@ub_loc)/np.linalg.norm(uc_loc))
                
                

            case 1:
                target = np.arange(ctr,ctr+nxz)
                # woops, mixed up front and back
                step_front = nxz*tiling[2] + nxy*(tiling[2]+1)
                step_back = step_front
                start_left1 = indx_vec[indxyz]*(nFYZ+(tiling[1]+1)*nFXZ+tiling[1]*nFXY)+tiling[2]*nyz*(indy_vec[indxyz]-1)+nyz*indz_vec[indxyz]
                start_left2 = indx_vec[indxyz]*(nFYZ+(tiling[1]+1)*nFXZ+tiling[1]*nFXY)+tiling[2]*nyz*indy_vec[indxyz]+nyz*indz_vec[indxyz]
                
                start_right1 = start_left1+ (nFYZ+(tiling[1]+1)*nFXZ+tiling[1]*nFXY)
                start_right2 = start_left2+ (nFYZ+(tiling[1]+1)*nFXZ+tiling[1]*nFXY)

                start_down1 = indx_vec[indxyz]*(nFYZ+(tiling[1]+1)*nFXZ+tiling[1]*nFXY)+nFYZ+indy_vec[indxyz]*nFXZ+ (indy_vec[indxyz]-1)*nFXY +indz_vec[indxyz]*nxy
                start_down2 = start_down1 + nFXY+nFXZ

                start_up1 = start_down1+nxy
                start_up2 = start_up1 + nFXY+nFXZ



                source = np.append(source,np.arange(start_left1,start_left1+nyz))
                source = np.append(source,np.arange(start_left2,start_left2+nyz))
                source = np.append(source,np.arange(ctr-step_back,ctr-step_back+nxz))
                
                source = np.append(source,np.arange(start_down1,start_down1+nxy))
                source = np.append(source,np.arange(start_down2,start_down2+nxy))
                source = np.append(source,np.arange(start_up1,start_up1+nxy))
                source = np.append(source,np.arange(start_up2,start_up2+nxy))
                source = np.append(source,np.arange(ctr+step_front,ctr+step_front+nxz))
                source = np.append(source,np.arange(start_right1,start_right1+nyz))
                source = np.append(source,np.arange(start_right2,start_right2+nyz))
                
                ctr += nxz
                
            
                if not b_vec[indxyz]:
                    Stot[np.ix_(target,source)]=-Sy
                    ub_loc = uXY[source]
                    uc_loc = uXY[target]
                    max_block_err = max(max_block_err,np.linalg.norm(uc_loc-Sy@ub_loc)/np.linalg.norm(uc_loc))
                    

            case 0:
                target = np.arange(ctr,ctr+nyz)
                step_right = nyz*tiling[2]*tiling[1] + nxy*(tiling[2]+1)*tiling[1] + nxz*(tiling[1]+1)*tiling[2]
                step_left = step_right
                
                
                start_front1 = (indx_vec[indxyz]-1)*(nFYZ+(tiling[1]+1)*nFXZ+tiling[1]*nFXY)+nFYZ+indy_vec[indxyz]*nFXZ+ (indy_vec[indxyz])*nFXY +indz_vec[indxyz]*nxz
                start_front2 = start_front1+nFYZ+tiling[1]*nFXY + (tiling[1]+1)*nFXZ
                
                start_back1 = start_front1 + nFXY + nFXZ
                start_back2 = start_front2 + nFXY + nFXZ
                
                start_down1 = (indx_vec[indxyz]-1)*(nFYZ+(tiling[1]+1)*nFXZ+tiling[1]*nFXY)+nFYZ+(indy_vec[indxyz]+1)*nFXZ+ (indy_vec[indxyz])*nFXY +indz_vec[indxyz]*nxy
                start_down2 = start_down1+(tiling[1])*nFXY + (tiling[1]+1)*nFXZ + nFYZ
                start_up1 = start_down1+nxy
                start_up2 = start_down2+nxy
                source = np.append(source,np.arange(ctr-step_left,ctr-step_left+nyz))
                source = np.append(source,np.arange(start_front1,start_front1+nxz))
                source = np.append(source,np.arange(start_front2,start_front2+nxz))
                source = np.append(source,np.arange(start_down1,start_down1+nxy))
                source = np.append(source,np.arange(start_down2,start_down2+nxy))
                source = np.append(source,np.arange(start_up1,start_up1+nxy))
                source = np.append(source,np.arange(start_up2,start_up2+nxy))
                source = np.append(source,np.arange(start_back1,start_back1+nxz))
                source = np.append(source,np.arange(start_back2,start_back2+nxz))
                source = np.append(source,np.arange(ctr+step_right,ctr+step_right+nyz))
                
                

                #print(source)
                ctr += nyz
                
                if not b_vec[indxyz]:
                    Stot[np.ix_(target,source)]=-Sx
                    ub_loc = uXY[source]
                    uc_loc = uXY[target]
                    max_block_err = max(max_block_err,np.linalg.norm(uc_loc-Sx@ub_loc)/np.linalg.norm(uc_loc))
    print("max_block_err = ",max_block_err)
    return Stot

def local_S(dir,px,py,pz,scl_x,scl_y,scl_z):
    """
    Construct local SOMS system

    
    INPUT:  - dir                       :   direction in {0,1,2} in which blocks are stacked/joined
            - (px,py,pz)                :   polynomial orders for each 1x1 block
            - (scl_x,scl_y,scl_z)       :   length scales of 1x1 blocks
    
    OUTPUT: - S_dir                     :   local S-block in direction dir



    """
    Dx,xpts = spectral.cheb(px)
    Dy,ypts = spectral.cheb(py)
    Dz,zpts = spectral.cheb(pz)
    nx = len(xpts)
    ny = len(ypts)
    nz = len(zpts)
    L_joined_dir,Ijl,Ijf,Ijd,Iju,Ijb,Ijr,Ic_dir,Ibox_joined_dir,Ii_dir,Ib_dir,XY_joined_dir = L_op(dir,px,py,pz,scl_x,scl_y,scl_z,kh)
    XY_joined_dir_i = XY_joined_dir[Ii_dir,:]
    XY_joined_dir_b = XY_joined_dir[Ib_dir,:]

    Lii_joined_dir = L_joined_dir[Ii_dir,:][:,Ii_dir]
    Lib_joined_dir = L_joined_dir[Ii_dir,:][:,Ib_dir]
    Lib_joined_dir_box = Lib_joined_dir[:,Ibox_joined_dir]


    # fiddle with interp since kronecker is not respected by global disc
    # inv_inter_xy is interpolation for the two xy-faces etc.

    Interp_x,Interp_y,Interp_z=interp_ops(px,py,pz,scl_x,scl_y,scl_z)
    if dir == 0:
        inv_inter_xy = np.kron(np.linalg.pinv(Interp_x),np.identity(ny-2))
        inv_inter_yz = np.identity((ny-2)*(nz-2))
        inv_inter_xz = np.kron(np.linalg.pinv(Interp_x),np.identity(nz-2))
    if dir == 1:
        iIy = np.linalg.pinv(Interp_y)
        inv_inter_xy_1 = np.kron(np.identity(nx-2),iIy[:,:(ny-2)])
        inv_inter_xy_2 = np.kron(np.identity(nx-2),iIy[:,(ny-2):])
        inv_inter_xy = np.append(inv_inter_xy_1,inv_inter_xy_2,axis=1)
        inv_inter_yz = np.kron(np.linalg.pinv(Interp_y),np.identity(nz-2))
        inv_inter_xz = np.identity((nx-2)*(nz-2))


    if dir == 2:
        iIz = np.linalg.pinv(Interp_z)
        inv_inter_xy = np.identity((nx-2)*(ny-2))
        inv_inter_yz1 = np.kron(np.identity(ny-2),iIz[:,:(nz-2)])
        inv_inter_yz2 = np.kron(np.identity(ny-2),iIz[:,(nz-2):])
        inv_inter_yz = np.append(inv_inter_yz1,inv_inter_yz2,axis=1)
        inv_inter_xz1 = np.kron(np.identity(nx-2),iIz[:,:(nz-2)])
        inv_inter_xz2 = np.kron(np.identity(nx-2),iIz[:,(nz-2):])
        inv_inter_xz = np.append(inv_inter_xz1,inv_inter_xz2,axis=1)
    XYu,Iul,Iuf,Iud,Iuu,Iub,Iur = XYU(dir,px,py,pz,scl_x,scl_y,scl_z)
    
    # C_dir is total inverse interpolation matrix, i.e. over the 6 faces


    C_dir = np.zeros( shape = ( len(Ibox_joined_dir) , XYu.shape[0] ) )

    C_dir[np.ix_(Ijl,Iul)] = inv_inter_yz
    C_dir[np.ix_(Ijf,Iuf)] = inv_inter_xz
    C_dir[np.ix_(Ijd,Iud)] = inv_inter_xy
    C_dir[np.ix_(Iju,Iuu)] = inv_inter_xy
    C_dir[np.ix_(Ijb,Iub)] = inv_inter_xz
    C_dir[np.ix_(Ijr,Iur)] = inv_inter_yz

    S_dir = -(np.linalg.solve(Lii_joined_dir,Lib_joined_dir[:,Ibox_joined_dir]@C_dir))[Ic_dir,:]
    return S_dir




Lx = 1.
Ly = 1.
Lz = 1.

pvec = np.array([4,6,8,10,12],dtype = np.int64)
condvec = np.zeros(shape = pvec.shape)
Nvec = np.zeros(shape = pvec.shape)
for indp in range(len(pvec)):
    p = pvec[indp]
    px = p
    py = p
    pz = p



    tiling = [4,4,4] #non-overlapping (!!!) tiling
    #tiling[dir] = 3
    Lx0 = tiling[0]
    Ly0 = tiling[1]
    Lz0 = tiling[2]

    scl_z = Lz/Lz0
    scl_y = Ly/Ly0
    scl_x = Lx/Lx0

    XYtot,md_vec,b_vec,nxy,nyz,nxz,indx_vec,indy_vec,indz_vec = global_dofs(tiling,px,py,pz,Lx,Ly,Lz)



    uXY = bc(XYtot,kh)
    S_x = local_S(0,px,py,pz,scl_x,scl_y,scl_z)
    S_y = local_S(1,px,py,pz,scl_x,scl_y,scl_z)
    S_z = local_S(2,px,py,pz,scl_x,scl_y,scl_z)
    Stot = construct_SOMS(nxy,nyz,nxz,md_vec,b_vec,XYtot,tiling,indx_vec,indy_vec,indz_vec,uXY,S_x,S_y,S_z)
    print("SOMS construction DONE")
    print("data = ",Stot.nbytes/1e6)
    Ib = np.where((np.abs(XYtot[:,0])<1e-10)|(np.abs(XYtot[:,0]-Lx)<1e-10) | (np.abs(XYtot[:,1])<1e-10)|(np.abs(XYtot[:,1]-Ly)<1e-10)|(np.abs(XYtot[:,2])<1e-10)|(np.abs(XYtot[:,2]-Lz)<1e-10))[0]
    Ii = [i for i in range(XYtot.shape[0]) if not i in Ib]
    ui = bc(XYtot[Ii,:],kh)
    ub = bc(XYtot[Ib,:],kh)
    print("solving:")
    Sii = Stot[Ii,:][:,Ii]
    uhat = -np.linalg.solve(Sii,Stot[Ii,:][:,Ib]@ub)

    print("S err final = ",np.linalg.norm(uhat-ui)/np.linalg.norm(ui))
    condS = np.linalg.cond(Sii)
    print("condS = ",condS)
    N= px*py*pz*tiling[0]*tiling[1]*tiling[2]
    print("N = ",N)
    Nvec[indp] = N
    condvec[indp] = condS

condfit = (1+np.log(Nvec))**3
condfit *= condvec[-1]/condfit[-1]
plt.figure(1)
plt.loglog(Nvec,condvec)
plt.loglog(Nvec,condfit,linestyle='dashed')
plt.show()