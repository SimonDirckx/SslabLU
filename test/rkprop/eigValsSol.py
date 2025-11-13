import numpy as np

import torch
import jax.numpy as jnp
import geometry.geom_2D.square as square
import matplotlib.pyplot as plt

import solver.spectral.spectralSolver as spectral
import solver.spectral.geom as spectralGeom


import solver.stencil.stencilSolver as stencil
import solver.stencil.geom as stencilGeom

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
from solver.hpsmultidomain.hpsmultidomain import pdo as pdoalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsGeomalt

from solver.spectralmultidomain.hps import hps_multidomain as hps
from solver.spectralmultidomain.hps import pdo as pdo
import solver.hpsmultidomain.hpsmultidomain.geom as hpsGeom
kh = 10.1
def eigvals_compute(disc,ny,xr,n_boxes_y = 2,k=1):

    # set up problem
    bnds = np.array([[0,0],[xr,1]])
    match disc:
        case 'hpsalt':
            geom = hpsGeomalt.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

            def c11(p):
                return torch.ones_like(p[:,0])
            def c22(p):
                return torch.ones_like(p[:,0])
            def c(p):
                return -kh*kh*torch.ones_like(p[:,0])
            diff_op = pdoalt.PDO_2d(c11=c11,c22=c22)#,c=c)
        case 'hps':
            geom = hpsGeom.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[...,1])*np.sinh(np.pi*p[...,0])

            def c11(p):
                return jnp.ones_like(p[...,0])
            def c22(p):
                return jnp.ones_like(p[...,0])
            def c(p):
                return -kh*kh*jnp.ones_like(p[...,0])
            diff_op = pdo.PDO2d(c11=c11,c22=c22,c=c)
            p = ny//n_boxes_y
            n_boxes_x = (int)(n_boxes_y*xr)
            ax = .5*xr/n_boxes_x
            ay = .5*1/n_boxes_y
            solver = hps.HPSMultidomain(diff_op,geom,np.array([ax,ay]),p)
            XX = solver.XX
            XXi = XX[solver._Ji,:]
            XXb = XX[solver._Jx,:]
            plt.figure(1)
            plt.scatter(XXi[:,0],XXi[:,1])
            plt.show()
            x,w0 = np.polynomial.legendre.leggauss(p)
            w = np.kron(np.ones(shape=(n_boxes_y,)),w0)
        case 'stencil':
            geom = stencilGeom.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

            def c11(p):
                return np.ones_like(p[:,0])
            def c22(p):
                return np.ones_like(p[:,0])
            def c(p):
                return -kh*kh*np.ones_like(p[:,0])
            diff_op = pdo.PDO2d(c11=c11,c22=c22,c=c)
            nx = (int)(xr*ny)
            nx = nx - nx%2 + 1
            solver = stencil.stencilSolver(diff_op,geom,[nx,ny])
            w = np.ones(shape=(ny-2,))
            XXi = solver.XXi
            XXb = solver.XXb

        case 'spectral':
            geom = spectralGeom.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

            def c11(p):
                return np.ones_like(p[:,0])
            def c22(p):
                return np.ones_like(p[:,0])
            def c(p):
                return -kh*kh*np.ones_like(p[:,0])
            diff_op = pdo.PDO2d(c11=c11,c22=c22,c=c)
            py = ny
            px = (int)(xr*ny)
            px = px + px%2
            solver = spectral.spectralSolver(diff_op,geom,[px,py])
            x,w0 = spectral.clenshaw_curtis_compute(py+1)
            w = np.sqrt(w0[1:len(w0)-1]/2)
            XXi = solver.XXi
            XXb = solver.XXb

    # set up S map

    xc = (xl+xr)/2

    Aii = np.array(solver._Aii.todense())
    print("condT = ",np.linalg.cond(Aii))
    Aib = np.array(solver._Aix.todense())
    #plt.figure(1)
    #plt.spy(Aii)
    #plt.show()
    
    xcr = .25
    Ic = np.where(np.abs(XXi[:,0]-xc)<1e-10)[0]
    Icr = np.where(np.abs(XXi[:,1]-xcr)<1e-10)[0]
    Il = np.where((np.abs(XXb[:,0]-xl)<1e-10)&(XXb[:,1]>0)&(XXb[:,1]<1))[0]
    Id = np.where((np.abs(XXb[:,1])<1e-10))[0]
    nx = len(Id)

    S = -(np.linalg.solve(Aii,Aib[:,Il]))[Ic,:]
    S = np.array(S)

    SW = np.diag(w)@S@np.diag(1./w)
    [e,V] = np.linalg.eig(SW)
    Isort = np.argsort(np.abs(e))
    e = e[Isort[::-1]]
    V = V[:,Isort[::-1]]
    
    XXl = XXb[Il,:]
    XXc = XXi[Ic,:]
    XXcr = XXi[Icr,:]
    f = np.sin(np.pi*k*XXl[:,1])
    factor = (np.sinh(xc*k*np.pi)/(np.sinh(xr*k*np.pi)))
    g = np.sin(np.pi*k*XXc[:,1])*factor
    print("norm g = ",np.linalg.norm(g,ord=np.inf))
        #plt.figure(1)
        #plt.plot(XXl[:,1],S@f)
        #plt.plot(XXc[:,1],g)
        #plt.legend(['Sf','g'])
        #plt.show()
    err = np.linalg.norm(S@f-g,ord=np.inf)/np.linalg.norm(g,ord=np.inf)
    f = f/np.linalg.norm(f)
    ip = f.T@V[:,k]
    guess = np.sin(np.pi*XXl[:,1])
    guess = guess/np.linalg.norm(guess)

    


    plt.figure(1)
    #plt.plot(XXl[:,1],f)
    plt.plot(XXl[:,1],V[:,0])
    plt.plot(XXl[:,1],V[:,1])
    plt.plot(XXl[:,1],V[:,2])
    #plt.plot(XXl[:,1],guess)
    #plt.plot(XXl[:,1],SW@guess)
    #plt.plot(XXl[:,1],S@guess)
    plt.legend(['v0','v1','v2'])
    #plt.plot(XXl[:,1],V[:,6])
    #plt.plot(XXl[:,1],V[:,7])
    #plt.plot(XXl[:,1],V[:,8])
    #plt.plot(XXl[:,1],V[:,9])
    #plt.legend(['f','0','1','2','3','4'])
    for i in range(5):
        

        #find freq. mode
        f = np.sin(np.pi*(i+1)*XXl[:,1])
        IP = f.T@V
        ind = np.argmax(np.abs(IP))
        v = V[:,ind]
        print("ind = ",ind)
        ub = np.zeros(shape = (Aib.shape[1],))
        ub[Il] = V[:,ind]
        ui = -np.linalg.solve(Aii,Aib@ub)

        alph = np.sqrt(abs(((i+1)**2)*(np.pi**2)-kh**2))
        if ((i+1)**2)*(np.pi**2)-kh**2>0:

            CMAT = np.zeros(shape = (2,2))
            CMAT[0,0] = 1.
            CMAT[0,1] = 1.

            CMAT[1,0] = np.exp(-2*alph)
            CMAT[1,1] = np.exp(2*alph)
            rhs = np.array([1,0.])
            coeffs = np.linalg.solve(CMAT,rhs)
            utest = coeffs[0]*np.exp(-alph*XXcr[:,0])+coeffs[1]*np.exp(alph*XXcr[:,0])
        else:
            CMAT = np.zeros(shape = (2,2))
            CMAT[0,0] = 1.
            CMAT[0,1] = 1.

            CMAT[1,0] = np.cos(2*alph)
            CMAT[1,1] = np.sin(2*alph)

            rhs = np.array([1,0.])
            coeffs = np.linalg.solve(CMAT,rhs)
            utest = coeffs[0]*np.cos(alph*XXcr[:,0])+coeffs[1]*np.sin(alph*XXcr[:,0])
        print("coeffs are : ",coeffs)
        
        ucr = ui[Icr]
        #ucr /= np.linalg.norm(ucr)
        N = XXcr.shape[0]
        Ny = XXl.shape[0]
        utest = np.real(utest)*np.max(ucr)/np.max(utest)
        utest /= np.linalg.norm(utest)
        ip = ucr.T@utest
        utest*=ip
        plt.figure(0)
        plt.plot(ucr)
        plt.plot(utest)
        #plt.plot(np.imag(utest))
        plt.legend(['ucr','rutest'])
        plt.show()

    print(ip)
    err_V = np.linalg.norm(V[:,k]-f/ip)
    print(err_V)
    return e,nx,err,err_V

ny = 8*16
xl = 0.
xr = 1.
xc = (xl+xr)/2

#e_hps_2,nx,_,_ = eigvals_compute('hps',ny,xr,1)
#print("nx = ", nx)

#e_hps_4,nx,_,_ = eigvals_compute('hps',ny,xr,2)
#print("nx = ", nx)
#e_hps_8,nx,_,_ = eigvals_compute('hps',ny,xr,4)
#print("nx = ", nx)
#e_hps_16,nx,_,_ = eigvals_compute('hps',ny,xr,8)
#print("nx = ", nx)

e_stencil,nx,_,_ = eigvals_compute('hps',ny,xr,8)
print("nx = ", nx)
#e_spectral,nx,_,_ = eigvals_compute('spectral',ny,xr)
#print("nx = ", nx)
#etest = np.zeros(shape = (ny,))
#for k in range(1,ny+1):
#    etest[k-1] = max([np.sinh(xc*k*np.pi)/np.sinh(xr*k*np.pi),1e-20])


#plt.figure(1)
#plt.semilogy(np.abs(e_stencil))
#plt.semilogy(np.abs(e_spectral))
#plt.semilogy(np.abs(e_hps_2))
#plt.semilogy(np.abs(e_hps_4))
#plt.semilogy(np.abs(e_hps_8))
#plt.semilogy(np.abs(e_hps_16))
#plt.semilogy(np.abs(etest))
#plt.legend(['stencil','spectral','hps2','hps4','hps8','hps16','exact'])
#plt.show()
'''
ny_vec = [24,32,40,48,56,64,72,80]
err_hps_2_vec = np.zeros(shape = (len(ny_vec),))
err_hps_4_vec = np.zeros(shape = (len(ny_vec),))
err_hps_8_vec = np.zeros(shape = (len(ny_vec),))
err_hps_16_vec = np.zeros(shape = (len(ny_vec),))

err_hps_2_V_vec = np.zeros(shape = (len(ny_vec),))
err_hps_4_V_vec = np.zeros(shape = (len(ny_vec),))
err_hps_8_V_vec = np.zeros(shape = (len(ny_vec),))
err_hps_16_V_vec = np.zeros(shape = (len(ny_vec),))


for ind_ny in range(len(ny_vec)):
    ny = ny_vec[ind_ny]
    e_hps_2,nx,err_hps_2,err_hps_2_V = eigvals_compute('hps',ny,xr,1)
    e_hps_4,nx,err_hps_4,err_hps_4_V = eigvals_compute('hps',ny,xr,2)
    e_hps_8,nx,err_hps_8,err_hps_8_V = eigvals_compute('hps',ny,xr,4)
    e_hps_16,nx,err_hps_16,err_hps_16_V = eigvals_compute('hps',ny,xr,8)
    err_hps_2_vec[ind_ny] = err_hps_2
    err_hps_4_vec[ind_ny] = err_hps_4
    err_hps_8_vec[ind_ny] = err_hps_8
    err_hps_16_vec[ind_ny] = err_hps_16
    err_hps_2_V_vec[ind_ny] = err_hps_2_V
    err_hps_4_V_vec[ind_ny] = err_hps_4_V
    err_hps_8_V_vec[ind_ny] = err_hps_8_V
    err_hps_16_V_vec[ind_ny] = err_hps_16_V
plt.figure(2)
plt.semilogy(ny_vec,err_hps_2_vec)
plt.semilogy(ny_vec,err_hps_4_vec)
plt.semilogy(ny_vec,err_hps_8_vec)
plt.semilogy(ny_vec,err_hps_16_vec)
plt.legend(['2','4','8','16'])
plt.show()

plt.figure(3)
plt.semilogy(ny_vec,err_hps_2_V_vec)
plt.semilogy(ny_vec,err_hps_4_V_vec)
plt.semilogy(ny_vec,err_hps_8_V_vec)
plt.semilogy(ny_vec,err_hps_16_V_vec)
plt.legend(['2','4','8','16'])
plt.show()
'''

