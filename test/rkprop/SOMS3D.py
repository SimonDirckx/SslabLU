import numpy as np
import solver.spectral.spectralSolver as spectral
import numpy.polynomial.chebyshev as chebpoly
import matplotlib.pyplot as plt
import scipy.linalg as sclinalg
import time


def SOMS_solver(px,py,pz,nbx,nby,nbz,Lx=1,Ly=1,Lz=1,kh=0.):
    tiling = [nbx,nby,nbz] #non-overlapping (!!!) tiling
    Lx0 = tiling[0]
    Ly0 = tiling[1]
    Lz0 = tiling[2]

    scl_x = Lx/Lx0
    scl_y = Ly/Ly0    
    scl_z = Lz/Lz0

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

    nxpts2 = len(xpts2)
    nypts2 = len(ypts2)
    nzpts2 = len(zpts2)

    ##########################
    # FORM DOFS AND SYSTEM
    ##########################

    XYtot = np.zeros(shape=(0,3))
    xy = np.zeros(shape = ((nx-2)*(ny-2),3))
    xz = np.zeros(shape = ((nx-2)*(nz-2),3))
    yz = np.zeros(shape = ((ny-2)*(nz-2),3))
    xy[:,0] = np.kron(xpts[1:nx-1],np.ones_like(ypts[1:ny-1]))
    xy[:,1] = np.kron(np.ones_like(xpts[1:nx-1]),ypts[1:ny-1])
    xz[:,0] = np.kron(xpts[1:nx-1],np.ones_like(zpts[1:nz-1]))
    xz[:,2] = np.kron(np.ones_like(xpts[1:nx-1]),zpts[1:nz-1])
    yz[:,1] = np.kron(ypts[1:ny-1],np.ones_like(zpts[1:nz-1]))
    yz[:,2] = np.kron(np.ones_like(ypts[1:ny-1]),zpts[1:nz-1])

    plnszX = (ny-2)*(nz-2)*(tiling[1])*(tiling[2])
    plnszY = (nx-2)*(nz-2)*tiling[0]*tiling[2]
    plnszZ = (nx-2)*(ny-2)*tiling[0]*tiling[1]

    for indx in range(tiling[0]+1):
        for indy in range(tiling[1]):
            for indz in range(tiling[2]):        
                XYtot = np.append(XYtot,yz+np.array([indx*scl_x,indy*scl_y,indz*scl_z]),axis=0)
    STRTy = XYtot.shape[0]
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(XYtot[:,0], XYtot[:,1], XYtot[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    for indy in range(tiling[1]+1):
        for indx in range(tiling[0]):
            for indz in range(tiling[2]):        
                XYtot = np.append(XYtot,xz+np.array([indx*scl_x,indy*scl_y,indz*scl_z]),axis=0)
    STRTz = XYtot.shape[0]
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(XYtot[:,0], XYtot[:,1], XYtot[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    for indz in range(tiling[2]+1):
        for indx in range(tiling[0]):
            for indy in range(tiling[1]):        
                XYtot = np.append(XYtot,xy+np.array([indx*scl_x,indy*scl_y,indz*scl_z]),axis=0)
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(XYtot[:,0], XYtot[:,1], XYtot[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    Stot = np.identity(XYtot.shape[0])

    for indx in range(1,tiling[0]):
        for indy in range(tiling[1]):
            for indz in range(tiling[2]):
                sources = indx*plnszX+indy*nby*(ny-2)*(nz-2)+indz*(ny-2)*(nz-2)+np.arange(0,(ny-2)*(nz-2))
                targetsl = sources-plnszX
                targetsr = sources+plnszX
                targetsd1 = STRTz+indz*plnszZ+indy*(nx-2)*(ny-2)+(indx-1)*nbx*(nx-2)*(ny-2)+np.arange(0,(nx-2)*(ny-2))
                targetsd2 = STRTz+indz*plnszZ+indy*(nx-2)*(ny-2)+indx*nbx*(nx-2)*(ny-2)+np.arange(0,(nx-2)*(ny-2))
                targetsd = np.append(targetsd1,targetsd2)
                targetsu = targetsd+plnszZ
                targetsf1 = STRTy+indy*plnszY+indz*(nx-2)*(nz-2)+(indx-1)*nbx*(nx-2)*(nz-2)+np.arange(0,(nx-2)*(nz-2))
                targetsf2 = STRTy+indy*plnszY+indz*(nx-2)*(nz-2)+(indx)*nbx*(nx-2)*(nz-2)+np.arange(0,(nx-2)*(nz-2))
                targetsf = np.append(targetsf1,targetsf2)
                targetsb = targetsf+plnszY
                #targetsf = 
                #targetsb = 
                #targetsu = 
                #targetsr = 

                #Stot[np.ix_(targets,sources)] = -S_x
    for indy in range(1,tiling[1]):
        for indx in range(tiling[0]):
            for indz in range(tiling[2]):
                sources = STRTy+indy*plnszY+indx*nbx*(nx-2)*(nz-2)+indz*(nx-2)*(nz-2)+np.arange(0,(ny-2)*(nz-2))
                targetsf = sources-plnszY
                targetsb = sources+plnszY
                targetsd1 = STRTz+indz*plnszZ+(indy-1)*(ny-2)*(nx-2)+indx*nbx*(nx-2)*(ny-2)+np.arange(0,(nx-2)*(ny-2))
                targetsd2 = STRTz+indz*plnszZ+indy*(ny-2)*(nx-2)+indx*nbx*(nx-2)*(ny-2)+np.arange(0,(nx-2)*(ny-2))
                targetsd = np.append(targetsd1,targetsd2)
                targetsu = targetsd+plnszZ
                targetsl1 = indx*plnszX+indz*(ny-2)*(nz-2)+(indy-1)*nby*(ny-2)*(nz-2)+np.arange(0,(ny-2)*(nz-2))
                targetsl2 = indx*plnszX+indz*(ny-2)*(nz-2)+(indy)*nby*(ny-2)*(nz-2)+np.arange(0,(ny-2)*(nz-2))
                targetsl = np.append(targetsl1,targetsl2)
                fig = plt.figure(1)
                ax = fig.add_subplot(projection='3d')
                ax.scatter(XYtot[sources,0], XYtot[sources,1], XYtot[sources,2])
                #ax.scatter(XYtot[targetsl,0], XYtot[targetsl,1], XYtot[targetsl,2])
                #ax.scatter(XYtot[targetsr,0], XYtot[targetsr,1], XYtot[targetsr,2])
                ax.scatter(XYtot[targetsd,0], XYtot[targetsd,1], XYtot[targetsd,2])
                ax.scatter(XYtot[targetsu,0], XYtot[targetsu,1], XYtot[targetsu,2])
                ax.scatter(XYtot[targetsf,0], XYtot[targetsf,1], XYtot[targetsf,2])
                ax.scatter(XYtot[targetsb,0], XYtot[targetsb,1], XYtot[targetsb,2])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                #plt.legend(['source','l','r','d','u','f','b'])
                plt.show()
    ##################################
    #       JOINED BLOCK SET-UP
    ##################################

    XY_x,Ii_x,Ib_x,Ibox_x,Ic_x = joined(xpts2,ypts,zpts)
    XY_y,Ii_y,Ib_y,Ibox_y,Ic_y = joined(ypts2,xpts,zpts)
    XY_z,Ii_z,Ib_z,Ibox_z,Ic_z = joined(zpts2,xpts,ypts)

    Dxx = Dx@Dx
    Dxx_joined = Dx2@Dx2
    Dyy = Dy@Dy
    Dyy_joined = Dy2@Dy2
    Dzz = Dz@Dz
    Dzz_joined = Dz2@Dz2

    L_x = -np.kron(np.kron(Dxx_joined,np.identity(ny)),np.identity(nz))-np.kron(np.kron(np.identity(nxpts2),Dyy),np.identity(nz))-np.kron(np.kron(np.identity(nxpts2),np.identity(ny)),Dzz)-kh*kh*np.kron(np.kron(np.identity(nxpts2),np.identity(ny)),np.identity(nz))
    L_y = -np.kron(np.kron(Dyy_joined,np.identity(nx)),np.identity(nz))-np.kron(np.kron(np.identity(nypts2),Dxx),np.identity(nz))-np.kron(np.kron(np.identity(nypts2),np.identity(nx)),Dzz)-kh*kh*np.kron(np.kron(np.identity(nypts2),np.identity(nx)),np.identity(nz))
    L_z = -np.kron(np.kron(Dzz_joined,np.identity(nx)),np.identity(ny))-np.kron(np.kron(np.identity(nzpts2),Dxx),np.identity(ny))-np.kron(np.kron(np.identity(nzpts2),np.identity(nx)),Dyy)-kh*kh*np.kron(np.kron(np.identity(nzpts2),np.identity(nx)),np.identity(ny))

    Lii_x = L_x[Ii_x,:][:,Ii_x]
    Lib_x = L_x[Ii_x,:][:,Ib_x]
    
    Lii_y = L_y[Ii_y,:][:,Ii_y]
    Lib_y = L_y[Ii_y,:][:,Ib_y]

    Lii_z = L_z[Ii_z,:][:,Ii_z]
    Lib_z = L_z[Ii_z,:][:,Ib_z]
    

    ##################################
    #     INTERPOLATION OPERATORS
    ##################################

    # E is evaluation

    C_x = gluing_mat(xpts,xpts2,ny,nz)
    C_y = gluing_mat(ypts,ypts2,nx,nz)
    C_z = gluing_mat(zpts,zpts2,nx,ny)
    
    
    S_x = -(np.linalg.solve(Lii_x,Lib_x[:,Ibox_x]@C_x))[Ic_x,:]
    S_y = -(np.linalg.solve(Lii_y,Lib_y[:,Ibox_y]@C_y))[Ic_y,:]
    S_z = -(np.linalg.solve(Lii_z,Lib_z[:,Ibox_z]@C_z))[Ic_z,:]


    
    
        
        
    
            

    for indx in range(1,tiling[0]):
        for indy in range(1,tiling[1]+1):
            start2 = indx*(tiling[1]+1)*(nx-2)+indx*tiling[1]*(ny-2) + (indy-1)*(ny-2)
            targets2 = np.arange(start2,start2+(ny-2))
            sources2 = np.zeros(shape=(0,2),dtype=np.int32)
            startl = start2-(tiling[1]+1)*(nx-2) - tiling[1]*(ny-2)
            sources2 = np.append(sources2,np.arange(startl,startl+(ny-2)))

            startd1 = start2-(tiling[1]-indy+2)*(nx-2) - (indy-1)*(ny-2)
            sources2 = np.append(sources2,np.arange(startd1,startd1+(nx-2)))

            startd2 = start2+(indy-1)*(nx-2) + (tiling[1]-indy+1)*(ny-2)
            sources2 = np.append(sources2,np.arange(startd2,startd2+(nx-2)))
            

            startu1 = start2-(tiling[1]-indy+1)*(nx-2) - (indy-1)*(ny-2)
            sources2 = np.append(sources2,np.arange(startu1,startu1+(nx-2)))

            startu2 = start2+(indy)*(nx-2) + (tiling[1]-indy+1)*(ny-2)
            sources2 = np.append(sources2,np.arange(startu2,startu2+(nx-2)))

            startr = start2+(tiling[1]+1)*(nx-2) + tiling[1]*(ny-2)
            sources2 = np.append(sources2,np.arange(startr,startr+(ny-2)))
            Stot[np.ix_(targets2,sources2)] = -S_x


    Ib = np.where((np.abs(XYtot[:,0])<1e-14) | (np.abs(XYtot[:,0]-Lx)<1e-14) | (np.abs(XYtot[:,1])<1e-14) | (np.abs(XYtot[:,1]-Ly)<1e-14))[0]
    Ii = [i for i in range(XYtot.shape[0]) if not i in Ib]

    return Stot,XYtot,Ii,Ib

def joined(x,y):
    nx = len(x)
    ny = len(y)

    sclx = x[-1]-x[0]
    scly = y[-1]-y[0]

    XY = np.zeros(shape=(nx*ny,2))
    XY[:,0] = np.kron(x,np.ones_like(y))
    XY[:,1] = np.kron(np.ones_like(x),y)

    Ii = np.where((XY[:,0]>0) & (XY[:,0]<sclx) & (XY[:,1]>0) & (XY[:,1]<scly) )[0]
    Ib = np.where((np.abs(XY[:,0])<1e-10) | (np.abs(XY[:,0]-sclx)<1e-10) | (np.abs(XY[:,1])<1e-10) | (np.abs(XY[:,1]-scly)<1e-10) )[0]

    XYi = XY[Ii,:]
    XYb = XY[Ib,:]


    Il = np.where( (np.abs(XYb[:,0])<1e-10) & (XYb[:,1]>0) & (XYb[:,1]<scly) )[0]
    Ir = np.where( (np.abs(XYb[:,0]-sclx)<1e-10) & (XYb[:,1]>0) & (XYb[:,1]<scly) )[0]
    Id = np.where( (np.abs(XYb[:,1])<1e-10) & (XYb[:,0]>0) & (XYb[:,0]<sclx) )[0]
    Iu = np.where( (np.abs(XYb[:,1]-scly)<1e-10) & (XYb[:,0]>0) & (XYb[:,0]<sclx) )[0]
    Ic = np.where( (np.abs(XYi[:,0]-.5*sclx)<1e-10))[0]


    return XY,Ii,Ib,Il,Ir,Id,Iu,Ic
def gluing_mat(x,xhat,ny):
    nx = len(x)
    nxhat = len(xhat)
    sclx = x[-1]-x[0]
    x2 = np.append(x[1:nx-1],sclx+x[1:nx-1])
    nx2 = len(x2)
    xhat = xhat[1:nxhat-1]
    nxhat = len(xhat)
    #E_x2 = np.zeros(shape = (nx2,nxhat))
    E_x2 = chebpoly.chebvander((x2-sclx)/sclx,nxhat-1).T
    #E_xhat = np.zeros(shape = (nxhat,nxhat))
    E_xhat = chebpoly.chebvander((xhat-sclx)/sclx,nxhat-1).T

    #for indcoeff in range(nxhat):
    #    ci = np.zeros(shape = (nxhat,))
    #    ci[indcoeff] = 1.
    #    Ti = chebpoly.Chebyshev(ci,domain=[0,2*sclx])
    #    E_x2[:,indcoeff] = Ti(x2)
    #    E_xhat[:,indcoeff] = Ti(xhat)


    [U,s,V] = np.linalg.svd(E_x2.T)
    k = sum(s>1e-13*s[0])
    Uk = U[:,:k]
    Vk = V[:k,:].T
    sk = s[:k]
    Sk = np.diag(sk**(-1))
    cc = ((E_xhat.T@Vk)@Sk)@Uk.T
    C = np.zeros(shape = (2*len(xhat)+2*(ny-2),4*(nx-2)+2*(ny-2)))

    C[np.ix_(np.arange(0,ny-2),np.arange(0,ny-2))] = np.identity(ny-2)
    C[np.ix_(np.arange(ny-2,ny-2+len(xhat)),np.arange(ny-2,ny-2+2*(nx-2)))] = cc
    C[np.ix_(np.arange(ny-2+len(xhat),ny-2+2*len(xhat)),np.arange(ny-2+2*(nx-2),ny-2+4*(nx-2)))] = cc
    C[np.ix_(np.arange(ny-2+2*len(xhat),2*(ny-2)+2*len(xhat)),np.arange(ny-2+4*(nx-2),2*(ny-2)+4*(nx-2)))] = np.identity(ny-2)


    return C