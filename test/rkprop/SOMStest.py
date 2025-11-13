
import numpy as np
import solver.spectral.spectralSolver as spectral
import numpy.polynomial.chebyshev as chebpoly
import matplotlib.pyplot as plt


def bc(p):
    r = np.sqrt(((p[:,0]+1.)**2)+((p[:,1]+1.)**2))
    
    return np.log(r)/(2*np.pi)

p0 = 64
nby_vec=[1,2,4,8]
eMat = np.zeros(shape = (p0,len(nby_vec)))
eAbsMat = np.zeros(shape = (p0,len(nby_vec)))
mu_nn_vec = np.zeros(shape = (len(nby_vec),))
for ind_nby in range(len(nby_vec)):
    nby = nby_vec[ind_nby]
    tiling = [nby,2*nby+1] # tiling is wrong way around, fix later

    Lx = 2.
    Ly = 1.

    Lx0 = (1+.5*(tiling[1]-1))
    Ly0 = (1+.5*(tiling[0]-1))

    scl_y = Ly/Ly0
    scl_x = Lx/Lx0

    p = p0/nby
    p = int(p)
    p = p-p%2+1
    print("p = ",p)

    Dx,xpts = spectral.cheb(p)
    Dy,ypts = spectral.cheb(p)

    xpts = ((xpts[::-1]+1)/2)*scl_x
    ypts = ((ypts[::-1]+1)/2)*scl_y

    Dx = (-2*Dx)/scl_x
    Dy = (-2*Dy)/scl_y

    nx = len(xpts)
    ny = len(ypts)


    XY_elem = np.zeros(shape=(nx*ny,2))
    XY_elem[:,0] = np.kron(xpts,np.ones_like(ypts))
    XY_elem[:,1] = np.kron(np.ones_like(xpts),ypts)


    E = np.zeros(shape = (nx*ny,nx*ny))

    for indcoeff in range(nx*ny):
        i = indcoeff%ny
        j = indcoeff//ny
        
        ci = np.zeros(shape = (nx,))
        cj = np.zeros(shape = (ny,))
        ci[i] = 1.
        cj[j] = 1.
        Ti = chebpoly.Chebyshev(ci,domain=[0,scl_x])
        Tj = chebpoly.Chebyshev(cj,domain=[0,scl_y])
        E[:,indcoeff] = Ti(XY_elem[:,0])*Tj(XY_elem[:,1])

    # create interior points

    Ii = np.where((XY_elem[:,0]>0) & (XY_elem[:,0]<scl_x) & (XY_elem[:,1]>0) & (XY_elem[:,1]<scl_y) )[0]
    Ib = np.where((np.abs(XY_elem[:,0])<1e-10) | (np.abs(XY_elem[:,0]-scl_x)<1e-10) | (np.abs(XY_elem[:,1])<1e-10) | (np.abs(XY_elem[:,1]-scl_y)<1e-10) )[0]


    XY_elem_i = XY_elem[Ii,:]
    XY_elem_b = XY_elem[Ib,:]
    print("scl_x = ",scl_x)
    print("scl_y = ",scl_y)
    Il = np.where( (np.abs(XY_elem_b[:,0])<1e-10) & (XY_elem_b[:,1]>0) & (XY_elem_b[:,1]<scl_y) )[0]
    Ir = np.where( (np.abs(XY_elem_b[:,0]-scl_x)<1e-10) & (XY_elem_b[:,1]>0) & (XY_elem_b[:,1]<scl_y) )[0]
    Id = np.where( (np.abs(XY_elem_b[:,1])<1e-10) & (XY_elem_b[:,0]>0) & (XY_elem_b[:,0]<scl_x) )[0]
    Iu = np.where( (np.abs(XY_elem_b[:,1]-scl_y)<1e-10) & (XY_elem_b[:,0]>0) & (XY_elem_b[:,0]<scl_x) )[0]

    Ibox = np.append(Id,Il)
    Ibox = np.append(Ibox,Iu)
    Ibox = np.append(Ibox,Ir)
    XYbox = XY_elem_b[Ibox,:]



    XYcross = np.zeros(shape = (nx+ny-4,2))
    XYcross[:nx-2,0]=xpts[1:nx-1]
    XYcross[:nx-2,1]=.5*scl_y

    XYcross[nx-2:,0]=.5*scl_x
    XYcross[nx-2:,1]=ypts[1:ny-1]



    ## build interpolation operator
    # Interp_cross = PHI_cross@(E^-1)
    # PHI_cross is the evaluation matrix from coeff to cross

    PHI_cross = np.zeros(shape = (nx+ny-4,XY_elem.shape[0]))

    for ij in range(PHI_cross.shape[1]):
        i = ij%ny
        j = ij//ny
        
        ci = np.zeros(shape = (nx,))
        cj = np.zeros(shape = (ny,))
        ci[i] = 1.
        cj[j] = 1.
        Ti = chebpoly.Chebyshev(ci,domain=[0,scl_x])
        Tj = chebpoly.Chebyshev(cj,domain=[0,scl_y])
        PHI_cross[:,ij] = Ti(XYcross[:,0])*Tj(XYcross[:,1])


    InterpCross = np.linalg.solve(E.T,PHI_cross.T).T


    # differential operator & solve

    Dxx = Dx@Dx
    Dyy = Dy@Dy
    L = np.kron(Dxx,np.identity(ny))+np.kron(np.identity(nx),Dyy)
    Lii = L[Ii,:][:,Ii]
    Lib = L[Ii,:][:,Ib]

    Lib_box = Lib[:,Ibox]

    S0 = np.zeros(shape = (nx*ny,len(Ibox)))
    S0[Ii,:] = -np.linalg.solve(Lii,Lib_box)

    Ibox_glob = [Ib[i] for i in Ibox]

    S0[Ibox_glob,:] = np.identity(len(Ibox))

    S = InterpCross@S0


    Il = np.where(np.abs(XYbox[:,0])<1e-10)[0]
    Ir = np.where(np.abs(XYbox[:,0]-scl_x)<1e-10)[0]
    Id = np.where(np.abs(XYbox[:,1])<1e-10)[0]
    Iu = np.where(np.abs(XYbox[:,1]-scl_y)<1e-10)[0]



    nxx = nx-2
    nyy = ny-2

    xx = np.zeros(shape = (nxx,2))
    yy = np.zeros(shape = (nxx,2))
    xx[:,0] = xpts[1:nx-1]
    yy[:,1] = ypts[1:ny-1]




    XYtot = np.zeros(shape=(0,2))
    XYtot_b = np.zeros(shape=(0,2))

    for indj in range(tiling[1]):
        for indi in range(tiling[0]):
            xxloc = xx + np.array([.5*(indj)*scl_x,.5*(indi+1)*scl_y])
            yyloc = yy + np.array([.5*(indj+1)*scl_x,.5*(indi)*scl_y])
            xxd = xx + np.array([.5*(indj)*scl_x,.5*(indi)*scl_y])
            xxu = xx + np.array([.5*(indj)*scl_x,(.5*indi+1)*scl_y])
            yyl = yy + np.array([.5*(indj)*scl_x,.5*(indi)*scl_y])
            yyr = yy + np.array([(.5*indj+1)*scl_x,.5*(indi)*scl_y])
            XYtot = np.append(XYtot,xxloc,axis=0)
            XYtot = np.append(XYtot,yyloc,axis=0)
            
            if indi == 0:
                XYtot_b = np.append(XYtot_b,xxd,axis=0)
            if indj == 0:
                XYtot_b = np.append(XYtot_b,yyl,axis=0)
            
            if indj == tiling[1]-1:
                XYtot_b = np.append(XYtot_b,yyr,axis=0)
            if indi == tiling[0]-1:
                XYtot_b = np.append(XYtot_b,xxu,axis=0)
            #plt.figure(1)
            #plt.scatter(XYtot[:,0],XYtot[:,1])
            #plt.scatter(XYtot_b[:,0],XYtot_b[:,1])
            #plt.axis('equal')
            #plt.show()
    ncross = nxx+nyy

    Stot_ii = np.identity(XYtot.shape[0])
    Stot_ib = np.zeros(shape = (XYtot.shape[0],XYtot_b.shape[0]))
    bdr_start = 0
    for indj in range(tiling[1]):
        for indi in range(tiling[0]):
            D = indj*ncross*(tiling[0])
            D1 = ncross*(tiling[0])
            start = indi*ncross + D
            target_loc = np.arange(start,start+ncross)
            source_loc_ud = np.zeros(shape=(0,),dtype = np.int32)
            source_loc_lr = np.zeros(shape=(0,),dtype = np.int32)
            
            if indi == 0:
                Stot_ib[np.ix_(np.arange(start,start+ncross),np.arange(bdr_start,bdr_start+nxx))]=-S[:,Id]
                bdr_start+=nxx
            
            if indj == 0:
                Stot_ib[np.ix_(np.arange(start,start+ncross),np.arange(bdr_start,bdr_start+nyy))]=-S[:,Il]
                bdr_start+=nyy
            
            if indj == tiling[1]-1:
                Stot_ib[np.ix_(np.arange(start,start+ncross),np.arange(bdr_start,bdr_start+nyy))]=-S[:,Ir]
                bdr_start+=nyy
            
            if indi == tiling[0]-1:
                Stot_ib[np.ix_(np.arange(start,start+ncross),np.arange(bdr_start,bdr_start+nxx))]=-S[:,Iu]
                bdr_start+=nxx

            if indj>0:
                source_loc_lr = np.append(source_loc_lr,np.arange(start-D1+ncross//2,start-D1+ncross))
                Stot_ii[np.ix_(np.arange(start,start+ncross),np.arange(start-D1+ncross//2,start-D1+ncross))]=-S[:,Il]
            if indi>0:
                source_loc_ud = np.append(source_loc_ud,np.arange(start-ncross,start-ncross//2))
                Stot_ii[np.ix_(target_loc,np.arange(start-ncross,start-ncross//2))]=-S[:,Id]
            if indi<tiling[0]-1:
                source_loc_ud = np.append(source_loc_ud,np.arange(start+ncross,start+ncross+ncross//2))
                Stot_ii[np.ix_(target_loc,np.arange(start+ncross,start+ncross+ncross//2))]=-S[:,Iu]
            if indj<tiling[1]-1:
                source_loc_lr = np.append(source_loc_lr,np.arange(start+D1+ncross//2,start+D1+ncross))
                Stot_ii[np.ix_(target_loc,np.arange(start+D1+ncross//2,start+D1+ncross))]=-S[:,Ir]



            #plt.figure(1)
            #plt.scatter(XYtot[:,0],XYtot[:,1])
            #plt.scatter(XYtot[target_loc,0],XYtot[target_loc,1])
            #plt.scatter(XYtot[source_loc_lr,0],XYtot[source_loc_lr,1])
            #plt.scatter(XYtot[source_loc_ud,0],XYtot[source_loc_ud,1])
            #plt.legend(['tot','target','lr','ud'])
    plt.figure(2)
    plt.spy(Stot_ii)
    
    plt.figure(3)
    plt.spy(Stot_ii,precision=1e-8)
    plt.show()

    ui = bc(XYtot)
    ub = bc(XYtot_b)
    uhat = -np.linalg.solve(Stot_ii,Stot_ib@ub)

    print("norm sol = ",np.linalg.norm(ui-uhat)/np.linalg.norm(ui))
    print("cond S = ",np.linalg.cond(Stot_ii))

    

    IIl = np.where(np.abs(XYtot_b[:,0])<1e-10)[0]
    IIc = np.where(np.abs(XYtot[:,0]-Lx/2)<1e-10)[0]

    interfaceMap = -(np.linalg.solve(Stot_ii,Stot_ib[:,IIl]))[IIc,:]
    print("shape interfaceMap = ",interfaceMap.shape)
    [e,V] = np.linalg.eig(interfaceMap)
    eAbs = np.sort(abs(e))
    eAbs = eAbs[::-1]
    eAbsMat[:,ind_nby]=eAbs
    eMat[:,ind_nby]=e
    [U,s,V] = np.linalg.svd(interfaceMap)
    s = np.sort(s)
    mu_nn = np.linalg.norm(s-e,np.inf)
    mu_nn_vec[ind_nby] = mu_nn


plt.figure(4)
plt.semilogy(eAbsMat)
plt.legend(['3','5','9','17'])

plt.figure(5)
plt.scatter(np.real(eMat[:,0]),np.imag(eMat[:,0]))
plt.scatter(np.real(eMat[:,1]),np.imag(eMat[:,1]))
plt.scatter(np.real(eMat[:,2]),np.imag(eMat[:,2]))
plt.scatter(np.real(eMat[:,3]),np.imag(eMat[:,3]))
plt.figure(6)
plt.semilogy(mu_nn_vec)
plt.show()