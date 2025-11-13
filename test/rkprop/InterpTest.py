
import numpy as np
import solver.spectral.spectralSolver as spectral
import numpy.polynomial.chebyshev as chebpoly
import matplotlib.pyplot as plt

def bc(p):
    return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

p = 51
Dx,xpts = spectral.cheb(p)
Dy,ypts = spectral.cheb(p)

xpts = (xpts[::-1]+1)/2
ypts = (ypts[::-1]+1)/2

Dx = -2*Dx
Dy = -2*Dy

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
    Ti = chebpoly.Chebyshev(ci,domain=[0,1])
    Tj = chebpoly.Chebyshev(cj,domain=[0,1])
    E[:,indcoeff] = Ti(XY_elem[:,0])*Tj(XY_elem[:,1])

# create interior points

Ii = np.where((XY_elem[:,0]>0) & (XY_elem[:,0]<1) & (XY_elem[:,1]>0) & (XY_elem[:,1]<1.) )[0]
Ib = np.where((XY_elem[:,0]==0) | (XY_elem[:,0]==1) | (XY_elem[:,1]==0) | (XY_elem[:,1]==1) )[0]


XY_elem_i = XY_elem[Ii,:]
XY_elem_b = XY_elem[Ib,:]

Il = np.where((XY_elem_b[:,0]==0) & (XY_elem_b[:,1]>0) & (XY_elem_b[:,1]<1.) )[0]
Ir = np.where((XY_elem_b[:,0]==1) & (XY_elem_b[:,1]>0) & (XY_elem_b[:,1]<1.) )[0]
Id = np.where((XY_elem_b[:,1]==0) & (XY_elem_b[:,0]>0) & (XY_elem_b[:,0]<1.) )[0]
Iu = np.where((XY_elem_b[:,1]==1) & (XY_elem_b[:,0]>0) & (XY_elem_b[:,0]<1.) )[0]

Ibox = np.append(Id,Il)
Ibox = np.append(Ibox,Iu)
Ibox = np.append(Ibox,Ir)
XYbox = XY_elem_b[Ibox,:]



XYcross = np.zeros(shape = (nx+ny-4,2))
XYcross[:nx-2,0]=xpts[1:nx-1]
XYcross[:nx-2,1]=.5

XYcross[nx-2:,0]=.5
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
    Ti = chebpoly.Chebyshev(ci,domain=[0,1])
    Tj = chebpoly.Chebyshev(cj,domain=[0,1])
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


nxx = nx-2
nyy = ny-2
XYtot = np.zeros(shape=(0,2))

# compute total DOFs

box_dof_sets = []
ctr_new = 0 # ctr for new DOF sets
for j in range(3):
    for i in range(3):
        dof_sets = []
        localDOFS = np.zeros(shape=(0,),dtype=np.int32)
        if i<2:
            localDOFS = Id
            dof_sets = dof_sets+[ctr_new]
            ctr_new = ctr_new+1
        else:
            ctr_old = box_dof_sets[(i-2)+j*3][2]
            dof_sets = dof_sets+[ctr_old]

        if j<2:
            localDOFS = np.append(localDOFS,Il,axis=0)    
            dof_sets = dof_sets+[ctr_new]
            ctr_new = ctr_new+1
        else:
            ctr_old = box_dof_sets[i+(j-2)*3][3]
            dof_sets = dof_sets+[ctr_old]
        dof_sets = dof_sets+[ctr_new,ctr_new+1]
        ctr_new = ctr_new+2
        box_dof_sets+=[dof_sets]
        localDOFS = np.append(localDOFS,Iu,axis=0)
        localDOFS = np.append(localDOFS,Ir,axis=0)

        XYtot = np.append(XYtot,XY_elem_b[localDOFS,:]+np.array([j*.5,i*.5]),axis = 0)
Stot = np.zeros(shape=(XYtot.shape[0],XYtot.shape[0]))
Stot = np.identity(XYtot.shape[0])
print(box_dof_sets)
F = bc(XYtot)

for j in range(3):
    for i in range(3):
        dof_set = box_dof_sets[i+j*3][0]
        localDOFS_down_start = box_dof_sets[i+j*3][0]*nxx
        localDOFS_left_start = box_dof_sets[i+j*3][1]*nxx
        localDOFS_up_start = box_dof_sets[i+j*3][2]*nxx
        localDOFS_right_start = box_dof_sets[i+j*3][3]*nxx


        range_down= np.arange(localDOFS_down_start,localDOFS_down_start+nxx)
        range_left=  np.arange(localDOFS_left_start,localDOFS_left_start+nyy)
        range_up=    np.arange(localDOFS_up_start,localDOFS_up_start+nxx)
        range_right= np.arange(localDOFS_right_start,localDOFS_right_start+nyy)
        
        total_source = np.append(range_down,range_left)
        total_source = np.append(total_source,range_up)
        total_source = np.append(total_source,range_right)

        if i<2:
            target_lr_dofs_start = box_dof_sets[i+1+j*3][0]*nxx
        else:
            target_lr_dofs_start = box_dof_sets[i-1+j*3][2]*nxx
        if j<2:
            target_ud_dofs_start = box_dof_sets[i+(j+1)*3][1]*nxx
        else:
            target_ud_dofs_start = box_dof_sets[i+(j-1)*3][3]*nxx


        floc = F[total_source]
        

        range_lr = np.arange(target_lr_dofs_start,target_lr_dofs_start+nxx)
        range_ud = np.arange(target_ud_dofs_start,target_ud_dofs_start+nxx)
        total_target = np.append(range_lr,range_ud)

        uloc = F[total_target]

        sf = S@floc
        print("uloc err = ",np.linalg.norm(uloc-sf)/np.linalg.norm(uloc))

        for i0 in range(len(total_target)):
            for j0 in range(len(total_source)):
                Stot[total_target[i0],total_source[j0]] = -S[i0,j0]


        #plt.figure(1)
        #plt.scatter(XYtot[:,0],XYtot[:,1])
        #plt.scatter(XYtot[range_lr,0],XYtot[range_lr,1])
        #plt.scatter(XYtot[range_ud,0],XYtot[range_ud,1])
        #plt.legend(['tot','lr','ud'])
        #plt.show()
#uvec = np.zeros(shape=(XYtot.shape[0],))


Ii = np.where((XYtot[:,0]>0) & (XYtot[:,0]<2) & (XYtot[:,1]>0) & (XYtot[:,1]<2))[0]
Ib = np.where((XYtot[:,0]==0) | (XYtot[:,0]==2) | (XYtot[:,1]==0) | (XYtot[:,1]==2))[0]

Stot_ii = Stot[Ii,:][:,Ii]
Stot_ib = Stot[Ii,:][:,Ib]
Fb = F[Ib]
Fi = F[Ii]
Fhat = -np.linalg.solve(Stot_ii,Stot_ib@Fb)

print(np.linalg.norm(Fhat-Fi)/np.linalg.norm(Fi))
print(np.linalg.cond(Stot_ii))
plt.figure(1)
plt.spy(Stot_ii)
plt.show()

print("shape XYtot = ",XYtot.shape)
print("nxx = ",nxx)
plt.figure(1)
plt.scatter(XYtot[:,0],XYtot[:,1])
plt.show()
Iq = np.unique(XYtot,axis = 0,return_index=True)[1]
print("len(Iq) = ",len(Iq))

