import numpy as np
import geometry.geom_3D.twistedTorus as twisted
import matplotlib.pyplot as plt
import matplotlib.tri as tri

'''
nx = 50
ny = 50
nz = 50

xpts = np.linspace(-1,1,nx)
ypts = np.linspace(-1,1,ny)
zpts = np.linspace(-1,1,nz)

XY = np.zeros(shape=(nx*ny,3))
YZ = np.zeros(shape=(ny*nz,3))
XZ = np.zeros(shape=(nx*nz,3))

XY[:,0] = np.kron(xpts,np.ones_like(ypts))
XY[:,1] = np.kron(np.ones_like(xpts),ypts)

YZ[:,1] = np.kron(ypts,np.ones_like(zpts))
YZ[:,2] = np.kron(np.ones_like(ypts),zpts)

XZ[:,0] = np.kron(xpts,np.ones_like(zpts))
XZ[:,2] = np.kron(np.ones_like(ypts),zpts)


ZZ_XY = np.zeros(shape=XY.shape)
ZZ_YZ = np.zeros(shape=YZ.shape)
ZZ_XZ = np.zeros(shape=XZ.shape)

ZZ_XY[:,0] = twisted.z1(XY,False)
ZZ_XY[:,1] = twisted.z2(XY,False)
ZZ_XY[:,2] = twisted.z3(XY,False)

ZZ_XZ[:,0] = twisted.z1(XZ,False)
ZZ_XZ[:,1] = twisted.z2(XZ,False)
ZZ_XZ[:,2] = twisted.z3(XZ,False)

ZZ_YZ[:,0] = twisted.z1(YZ,False)
ZZ_YZ[:,1] = twisted.z2(YZ,False)
ZZ_YZ[:,2] = twisted.z3(YZ,False)


YY_XY = np.zeros(shape=XY.shape)
YY_YZ = np.zeros(shape=YZ.shape)
YY_XZ = np.zeros(shape=XZ.shape)

YY_XY[:,0] = twisted.y1(ZZ_XY,False)
YY_XY[:,1] = twisted.y2(ZZ_XY,False)
YY_XY[:,2] = twisted.y3(ZZ_XY,False)

YY_XZ[:,0] = twisted.y1(ZZ_XZ,False)
YY_XZ[:,1] = twisted.y2(ZZ_XZ,False)
YY_XZ[:,2] = twisted.y3(ZZ_XZ,False)

YY_YZ[:,0] = twisted.y1(ZZ_YZ,False)
YY_YZ[:,1] = twisted.y2(ZZ_YZ,False)
YY_YZ[:,2] = twisted.y3(ZZ_YZ,False)



XX=np.random.uniform(size=(100,3),low=-1,high=1)

ZZ=np.zeros(shape = XX.shape)
ZZ[:,0] = twisted.z1(XX,False)
ZZ[:,1] = twisted.z2(XX,False)
ZZ[:,2] = twisted.z3(XX,False)

YY=np.zeros(shape = ZZ.shape)
YY[:,0] = twisted.y1(ZZ,False)
YY[:,1] = twisted.y2(ZZ,False)
YY[:,2] = twisted.y3(ZZ,False)


print("y err = ",np.linalg.norm(YY-XX,ord=np.inf))
twisted.check_param()



fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(XY[:,0],XY[:,1],XY[:,2],label = 'XY')
ax.scatter(YZ[:,0],YZ[:,1],YZ[:,2],label = 'YZ')
ax.scatter(XZ[:,0],XZ[:,1],XZ[:,2],label = 'XZ')
plt.legend()
plt.axis('equal')

fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.scatter(ZZ_XY[:,0],ZZ_XY[:,1],ZZ_XY[:,2],label = 'XY')
ax.scatter(ZZ_YZ[:,0],ZZ_YZ[:,1],ZZ_YZ[:,2],label = 'YZ')
ax.scatter(ZZ_XZ[:,0],ZZ_XZ[:,1],ZZ_XZ[:,2],label = 'XZ')
plt.legend()
plt.axis('equal')

fig = plt.figure(3)
ax = fig.add_subplot(projection='3d')
ax.scatter(YY_XY[:,0],YY_XY[:,1],YY_XY[:,2],label = 'XY')
ax.scatter(YY_YZ[:,0],YY_YZ[:,1],YY_YZ[:,2],label = 'YZ')
ax.scatter(YY_XZ[:,0],YY_XZ[:,1],YY_XZ[:,2],label = 'XZ')
plt.legend()
plt.axis('equal')

plt.show()
'''
nx=200
ny=200

xpts = np.linspace(-4,4,nx)
ypts = np.linspace(-4,4,ny)

ZZ = np.zeros(shape=(nx*ny,3))
ZZ[:,0] = np.kron(xpts,np.ones_like(ypts))
ZZ[:,1] = np.kron(np.ones_like(xpts),ypts)

sliceYY = np.zeros(shape=ZZ.shape)
sliceYY[:,0] = twisted.y1(ZZ,False)
sliceYY[:,1] = twisted.y2(ZZ,False)
sliceYY[:,2] = twisted.y3(ZZ,False)


I = np.where( (sliceYY[:,0]>=twisted.bnds[0][0]) & (sliceYY[:,0]<=twisted.bnds[1][0]) & (sliceYY[:,1]>=twisted.bnds[0][1]) & (sliceYY[:,1]<=twisted.bnds[1][1]) & (sliceYY[:,2]>=twisted.bnds[0][2]) & (sliceYY[:,2]<=twisted.bnds[1][2]) )[0]

sliceZZ = np.zeros(shape=(len(I),3))
sliceZZ[:,0] = twisted.z1(sliceYY[I,:],False)
sliceZZ[:,1] = twisted.z2(sliceYY[I,:],False)
sliceZZ[:,2] = twisted.z3(sliceYY[I,:],False)

triang = tri.Triangulation(sliceZZ[:,0],sliceZZ[:,1])

tri0 = triang.triangles

q1 = (sliceZZ[tri0[:,0],:]+sliceZZ[tri0[:,1],:])/2.
q2 = (sliceZZ[tri0[:,1],:]+sliceZZ[tri0[:,2],:])/2.
q3 = (sliceZZ[tri0[:,2],:]+sliceZZ[tri0[:,0],:])/2.


yy1 = np.zeros(shape = q1.shape)
yy2 = np.zeros(shape = q2.shape)
yy3 = np.zeros(shape = q3.shape)

yy1[:,0] = twisted.y1(q1,False)
yy1[:,1] = twisted.y2(q1,False)
yy1[:,2] = twisted.y3(q1,False)

yy2[:,0] = twisted.y1(q2,False)
yy2[:,1] = twisted.y2(q2,False)
yy2[:,2] = twisted.y3(q2,False)

yy3[:,0] = twisted.y1(q3,False)
yy3[:,1] = twisted.y2(q3,False)
yy3[:,2] = twisted.y3(q3,False)


b1 = (yy1[:,0]<twisted.bnds[0][0]) | (yy1[:,0]>twisted.bnds[1][0]) | (yy1[:,1]<twisted.bnds[0][1]) | (yy1[:,1]>twisted.bnds[1][1]) | (yy1[:,2]<twisted.bnds[0][2]) | (yy1[:,2]>twisted.bnds[1][2])
b2 = (yy2[:,0]<twisted.bnds[0][0]) | (yy2[:,0]>twisted.bnds[1][0]) | (yy2[:,1]<twisted.bnds[0][1]) | (yy2[:,1]>twisted.bnds[1][1]) | (yy2[:,2]<twisted.bnds[0][2]) | (yy2[:,2]>twisted.bnds[1][2])
b3 = (yy3[:,0]<twisted.bnds[0][0]) | (yy3[:,0]>twisted.bnds[1][0]) | (yy3[:,1]<twisted.bnds[0][1]) | (yy3[:,1]>twisted.bnds[1][1]) | (yy3[:,2]<twisted.bnds[0][2]) | (yy3[:,2]>twisted.bnds[1][2])


mask = (b1&b2)|(b1&b3)|(b2&b3)
triang.set_mask(mask)

plt.figure(5)
plt.triplot(triang, 'b-', lw=1, zorder=3, label='inner')
plt.axis('equal')
plt.show()