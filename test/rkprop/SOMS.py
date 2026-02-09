import numpy as np
import solver.spectral.spectralSolver as spectral
import numpy.polynomial.chebyshev as chebpoly
import matplotlib.pyplot as plt
import scipy.linalg as sclinalg
import time


def SOMS_solver(px,py,nbx,nby,kh=0.,Lx=1,Ly=1):
    tiling = [nbx,nby] #non-overlapping (!!!) tiling
    Lx0 = tiling[0]
    Ly0 = tiling[1]

    scl_y = Ly/Ly0
    scl_x = Lx/Lx0
    Dx,xpts = spectral.cheb(px)
    Dy,ypts = spectral.cheb(py)

    Dx = - 2*Dx/scl_x
    Dy = - 2*Dy/scl_y

    xpts = ((xpts[::-1]+1)/2)*scl_x
    ypts = ((ypts[::-1]+1)/2)*scl_y

    nx = len(xpts)
    ny = len(ypts)


    px_joined = (3*px)//2
    py_joined = (3*py)//2

    px_joined = px_joined-px_joined%2
    py_joined = py_joined-py_joined%2

    
    Dx2,xpts2 = spectral.cheb(px_joined)
    Dy2,ypts2 = spectral.cheb(py_joined)
    xpts2 = (1+xpts2[::-1])*scl_x
    ypts2 = (1+ypts2[::-1])*scl_y


    Dx2 = - Dx2/scl_x
    Dy2 = - Dy2/scl_y

    nxpts2 = len(xpts2)
    nypts2 = len(ypts2)
    ##################################
    #       JOINED BLOCK SET-UP
    ##################################



    #################
    # JOINED IN XDIR


    XY_joined_x = np.zeros(shape=(nxpts2*ny,2))
    XY_joined_x[:,0] = np.kron(xpts2,np.ones_like(ypts))
    XY_joined_x[:,1] = np.kron(np.ones_like(xpts2),ypts)

    Ii_x = np.where((XY_joined_x[:,0]>0) & (XY_joined_x[:,0]<2*scl_x) & (XY_joined_x[:,1]>0) & (XY_joined_x[:,1]<scl_y) )[0]
    Ib_x = np.where((np.abs(XY_joined_x[:,0])<1e-10) | (np.abs(XY_joined_x[:,0]-2*scl_x)<1e-10) | (np.abs(XY_joined_x[:,1])<1e-10) | (np.abs(XY_joined_x[:,1]-scl_y)<1e-10) )[0]

    XY_joined_x_i = XY_joined_x[Ii_x,:]
    XY_joined_x_b = XY_joined_x[Ib_x,:]

    Il_x = np.where( (np.abs(XY_joined_x_b[:,0])<1e-10) & (XY_joined_x_b[:,1]>0) & (XY_joined_x_b[:,1]<scl_y) )[0]
    Ir_x = np.where( (np.abs(XY_joined_x_b[:,0]-2*scl_x)<1e-10) & (XY_joined_x_b[:,1]>0) & (XY_joined_x_b[:,1]<scl_y) )[0]
    Id_x = np.where( (np.abs(XY_joined_x_b[:,1])<1e-10) & (XY_joined_x_b[:,0]>0) & (XY_joined_x_b[:,0]<2*scl_x) )[0]
    Iu_x = np.where( (np.abs(XY_joined_x_b[:,1]-scl_y)<1e-10) & (XY_joined_x_b[:,0]>0) & (XY_joined_x_b[:,0]<2*scl_x) )[0]

    Ic_x = np.where( (np.abs(XY_joined_x_i[:,0]-scl_x)<1e-10))[0]

    Ibox_joined_x = np.append(Il_x,Id_x)
    Ibox_joined_x = np.append(Ibox_joined_x,Iu_x)
    Ibox_joined_x = np.append(Ibox_joined_x,Ir_x)
    


    #################
    # JOINED IN YDIR

    XY_joined_y = np.zeros(shape=(nypts2*nx,2))
    XY_joined_y[:,0] = np.kron(xpts,np.ones_like(ypts2))
    XY_joined_y[:,1] = np.kron(np.ones_like(xpts),ypts2)


    Ii_y = np.where((XY_joined_y[:,0]>0) & (XY_joined_y[:,0]<scl_x) & (XY_joined_y[:,1]>0) & (XY_joined_y[:,1]<2*scl_y) )[0]
    Ib_y = np.where((np.abs(XY_joined_y[:,0])<1e-10) | (np.abs(XY_joined_y[:,0]-scl_x)<1e-10) | (np.abs(XY_joined_y[:,1])<1e-10) | (np.abs(XY_joined_y[:,1]-2*scl_y)<1e-10) )[0]

    XY_joined_y_i = XY_joined_y[Ii_y,:]
    XY_joined_y_b = XY_joined_y[Ib_y,:]


    Il_y = np.where( (np.abs(XY_joined_y_b[:,0])<1e-10) & (XY_joined_y_b[:,1]>0) & (XY_joined_y_b[:,1]<2*scl_y) )[0]
    Ir_y = np.where( (np.abs(XY_joined_y_b[:,0]-scl_x)<1e-10) & (XY_joined_y_b[:,1]>0) & (XY_joined_y_b[:,1]<2*scl_y) )[0]
    Id_y = np.where( (np.abs(XY_joined_y_b[:,1])<1e-10) & (XY_joined_y_b[:,0]>0) & (XY_joined_y_b[:,0]<scl_x) )[0]
    Iu_y = np.where( (np.abs(XY_joined_y_b[:,1]-2*scl_y)<1e-10) & (XY_joined_y_b[:,0]>0) & (XY_joined_y_b[:,0]<scl_x) )[0]
    Ic_y = np.where( (np.abs(XY_joined_y_i[:,1]-scl_y)<1e-10))[0]

    Ibox_joined_y = np.append(Il_y,Id_y)
    Ibox_joined_y = np.append(Ibox_joined_y,Iu_y)
    Ibox_joined_y = np.append(Ibox_joined_y,Ir_y)
    

    Dxx = Dx@Dx
    Dxx_joined = Dx2@Dx2
    Dyy = Dy@Dy
    Dyy_joined = Dy2@Dy2

    L_joined_x = -np.kron(Dxx_joined,np.identity(ny))-np.kron(np.identity(nxpts2),Dyy)-kh*kh*np.kron(np.identity(nxpts2),np.identity(ny))
    L_joined_y = -np.kron(Dxx,np.identity(nypts2))-np.kron(np.identity(nx),Dyy_joined)-kh*kh*np.kron(np.identity(nx),np.identity(nypts2))

    Lii_joined_x = L_joined_x[Ii_x,:][:,Ii_x]
    Lib_joined_x = L_joined_x[Ii_x,:][:,Ib_x]
    
    Lii_joined_y = L_joined_y[Ii_y,:][:,Ii_y]
    Lib_joined_y = L_joined_y[Ii_y,:][:,Ib_y]
    

    ##################################
    #     INTERPOLATION OPERATORS
    ##################################

    # E is evaluation


    x2 = np.append(xpts[1:nx-1],scl_x+xpts[1:nx-1])
    y2 = np.append(ypts[1:ny-1],scl_y+ypts[1:ny-1])

    nx2 = len(x2)
    ny2 = len(y2)

    xpts2 = xpts2[1:nxpts2-1]
    ypts2 = ypts2[1:nypts2-1]

    nxpts2 = len(xpts2)
    nypts2 = len(ypts2)

    E_x2 = np.zeros(shape = (nx2,nxpts2))
    E_y2 = np.zeros(shape = (ny2,nypts2))
    E_xpts2 = np.zeros(shape = (nxpts2,nxpts2))
    E_ypts2 = np.zeros(shape = (nypts2,nypts2))


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
    Interp_x = np.linalg.solve(E_xpts2.T,E_x2.T).T 
    Interp_y = np.linalg.solve(E_ypts2.T,E_y2.T).T 

    C_x = np.zeros(shape = (2*len(xpts2)+2*(ny-2),4*(nx-2)+2*(ny-2)))
    C_y = np.zeros(shape = (2*len(ypts2)+2*(nx-2),4*(ny-2)+2*(nx-2)))

    C_x[np.ix_(np.arange(0,ny-2),np.arange(0,ny-2))] = np.identity(ny-2)
    C_x[np.ix_(np.arange(ny-2,ny-2+len(xpts2)),np.arange(ny-2,ny-2+2*(nx-2)))] = np.linalg.pinv(Interp_x)
    C_x[np.ix_(np.arange(ny-2+len(xpts2),ny-2+2*len(xpts2)),np.arange(ny-2+2*(nx-2),ny-2+4*(nx-2)))] = np.linalg.pinv(Interp_x)
    C_x[np.ix_(np.arange(ny-2+2*len(xpts2),2*(ny-2)+2*len(xpts2)),np.arange(ny-2+4*(nx-2),2*(ny-2)+4*(nx-2)))] = np.identity(ny-2)

    C_y[np.ix_(np.arange(0,len(ypts2)),np.arange(0,2*(ny-2)))] = np.linalg.pinv(Interp_y)
    C_y[np.ix_(np.arange(len(ypts2),len(ypts2)+(nx-2)),np.arange(2*(ny-2),2*(ny-2)+(nx-2)))] = np.identity(nx-2)
    C_y[np.ix_(np.arange(len(ypts2)+(nx-2),len(ypts2)+2*(nx-2)),np.arange(2*(ny-2)+(nx-2),2*(ny-2)+2*(nx-2)))] = np.identity(nx-2)
    C_y[np.ix_(np.arange(len(ypts2)+2*(nx-2),2*len(ypts2)+2*(nx-2)),np.arange(2*(ny-2)+2*(nx-2),4*(ny-2)+2*(nx-2)))] = np.linalg.pinv(Interp_y)


    S_x = -(np.linalg.solve(Lii_joined_x,Lib_joined_x[:,Ibox_joined_x]@C_x))[Ic_x,:]
    S_y = -(np.linalg.solve(Lii_joined_y,Lib_joined_y[:,Ibox_joined_y]@C_y))[Ic_y,:]

    ##########################
    # FORM DOFS AND SYSTEM
    ##########################

    XYtot = np.zeros(shape=(0,2))
    xyloc = np.zeros(shape = (0,2))
    xx = np.zeros(shape = (nx-2,2))
    yy = np.zeros(shape = (ny-2,2))
    xx[:,0] = xpts[1:nx-1]
    yy[:,1] = ypts[1:ny-1]

    for indx in range(tiling[0]):
        
        for indy in range(tiling[1]):        
            XYtot = np.append(XYtot,yy+np.array([indx*scl_x,indy*scl_y]),axis=0)
    
        for indy in range(tiling[1]+1):
            XYtot = np.append(XYtot,xx+np.array([indx*scl_x,indy*scl_y]),axis=0)
            
    for indy in range(tiling[1]):        
            XYtot = np.append(XYtot,yy+np.array([tiling[0]*scl_x,indy*scl_y]),axis=0)
        
        
    Stot = np.identity(XYtot.shape[0])

    for indx in range(1,tiling[0]+1):
        for indy in range(1,tiling[1]):
            start1 = indx*(tiling[1]*(ny-2)) + (indx-1)*(tiling[1]+1)*(nx-2) + indy*(nx-2)
            targets1 = np.arange(start1,start1+nx-2)
            sources1 = np.zeros(shape=(0,2),dtype=np.int32)

            startl = start1-indy*(nx-2)-(tiling[1]-indy+1)*(ny-2)
            sources1 = np.append(sources1,np.arange(startl,startl+2*(ny-2)))
            startd = start1-(nx-2)
            sources1 = np.append(sources1,np.arange(startd,startd+(nx-2)))
            startu = start1+(nx-2)
            sources1 = np.append(sources1,np.arange(startu,startu+(nx-2)))
            startr = start1+(tiling[1]+1-indy)*(nx-2)+(indy-1)*(ny-2)
            sources1 = np.append(sources1,np.arange(startr,startr+2*(ny-2)))
            Stot[np.ix_(targets1,sources1)] = -S_y
            

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

    #Sii = Stot[Ii,:][:,Ii]
    #Sib = Stot[Ii,:][:,Ib]
    return Stot,XYtot,Ii,Ib