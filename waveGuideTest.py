import numpy as np
import matplotlib.pyplot as plt

def bfield_crystal_waveguide(xx,kh):
    
    mag   = 0.930655
    width = 2500; 
    
    b = np.zeros(shape=(xx.shape[0],))
    
    dist = 0.04
    x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
    y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
    
    # box of points [x0,x1] x [y0,y1]
    for x in np.arange(x0,x1,dist):
        for y in np.arange(y0,y1,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * np.exp(-width * xx_sq_c)

    # box of points [x0,x1] x [y0,y2]
    for x in np.arange(x2,x3,dist):
        for y in np.arange(y0,y2-0.5*dist,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * np.exp(-width * xx_sq_c)
            
    # box of points [x0,x3] x [y2,y3]
    for x in np.arange(x0,x3,dist):
        for y in np.arange(y2,y3,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * np.exp(-width * xx_sq_c)    
    
    kh_fun = -kh**2 * (1 - b)
    return kh_fun#kh_fun.unsqueeze(-1)
n=500
xpts = np.linspace(0.,1.,n)
ypts = np.linspace(0.,1.,n)
xx=np.zeros(shape=(n*n,2))
nwaves = 24.623521102434587
kh = (nwaves+0.03)*2*np.pi+1.8
[X,Y] = np.meshgrid(xpts,ypts)
ind = 0
for i in range(n):
    x=xpts[i]
    for j in range(n):
        xx[ind,:] = [x,ypts[j]]
        ind+=1
print("REACHED")

b = bfield_crystal_waveguide(xx,nwaves)
b = np.reshape(b,(n,n))

plt.figure(1)
plt.imshow(b,interpolation='spline16')
plt.colorbar()
plt.show()