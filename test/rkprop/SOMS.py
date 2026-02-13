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


    px_joined = 2*px-2#(3*px)//2
    py_joined = 2*py-2#(3*py)//2

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

    XY_hor,Ii_hor,Ib_hor,Il_hor,Ir_hor,Id_hor,Iu_hor,Ic_hor = joined(xpts2,ypts)
    XY_ver,Ii_ver,Ib_ver,Il_ver,Ir_ver,Id_ver,Iu_ver,Ic_ver = joined(ypts2,xpts)

    Dxx = Dx@Dx
    Dxx_joined = Dx2@Dx2
    Dyy = Dy@Dy
    Dyy_joined = Dy2@Dy2

    L_hor = -np.kron(Dxx_joined,np.identity(ny))-np.kron(np.identity(nxpts2),Dyy)-kh*kh*np.kron(np.identity(nxpts2),np.identity(ny))
    L_ver = -np.kron(Dyy_joined,np.identity(nx))-np.kron(np.identity(nypts2),Dxx)-kh*kh*np.kron(np.identity(nypts2),np.identity(nx))

    Lii_hor = L_hor[Ii_hor,:][:,Ii_hor]
    Lib_hor = L_hor[Ii_hor,:][:,Ib_hor]
    
    Lii_ver = L_ver[Ii_ver,:][:,Ii_ver]
    Lib_ver = L_ver[Ii_ver,:][:,Ib_ver]
    

    ##################################
    #     INTERPOLATION OPERATORS
    ##################################

    # E is evaluation

    C_hor = gluing_mat(xpts,xpts2,ny)
    C_ver = gluing_mat(ypts,ypts2,nx)

    Ibox_hor = np.append(Il_hor,Id_hor)
    Ibox_hor = np.append(Ibox_hor,Iu_hor)
    Ibox_hor = np.append(Ibox_hor,Ir_hor)

    Ibox_ver = np.append(Il_ver,Id_ver)
    Ibox_ver = np.append(Ibox_ver,Iu_ver)
    Ibox_ver = np.append(Ibox_ver,Ir_ver)
    
    
    
    S_x = -(np.linalg.solve(Lii_hor,Lib_hor[:,Ibox_hor]@C_hor))[Ic_hor,:]
    S_y = -(np.linalg.solve(Lii_ver,Lib_ver[:,Ibox_ver]@C_ver))[Ic_ver,:]
    ##########################
    # FORM DOFS AND SYSTEM
    ##########################

    XYtot = np.zeros(shape=(0,2))
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
            sources1l = np.arange(startl,startl+2*(ny-2))
            startd = start1-(nx-2)
            sources1d = np.arange(startd,startd+(nx-2))
            startu = start1+(nx-2)
            sources1u = np.arange(startu,startu+(nx-2))
            startr = start1+(tiling[1]+1-indy)*(nx-2)+(indy-1)*(ny-2)
            sources1r = np.arange(startr,startr+2*(ny-2))
            
            sources1 = np.append(sources1,sources1d)
            sources1 = np.append(sources1,sources1l)
            sources1 = np.append(sources1,sources1r)
            sources1 = np.append(sources1,sources1u)
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


    [U,s,V] = np.linalg.svd(E_xhat)
    k = sum(s>1e-15*s[0])
    Uk = U[:,:k].T
    Vk = V[:k,:].T
    sk = s[:k]
    Sk = np.diag(sk**(-1))
    Interp = (Vk@(Sk@(Uk@(E_x2)))).T
    #Interp = np.linalg.solve(E_xhat.T,E_x2.T).T
    #Interp = np.linalg.solve(E_xhat,E_x2).T
    C = np.zeros(shape = (2*len(xhat)+2*(ny-2),4*(nx-2)+2*(ny-2)))

    C[np.ix_(np.arange(0,ny-2),np.arange(0,ny-2))] = np.identity(ny-2)
    C[np.ix_(np.arange(ny-2,ny-2+len(xhat)),np.arange(ny-2,ny-2+2*(nx-2)))] = np.linalg.pinv(Interp,rcond = 1e-14)
    C[np.ix_(np.arange(ny-2+len(xhat),ny-2+2*len(xhat)),np.arange(ny-2+2*(nx-2),ny-2+4*(nx-2)))] = np.linalg.pinv(Interp,rcond = 1e-14)
    C[np.ix_(np.arange(ny-2+2*len(xhat),2*(ny-2)+2*len(xhat)),np.arange(ny-2+4*(nx-2),2*(ny-2)+4*(nx-2)))] = np.identity(ny-2)


    return C