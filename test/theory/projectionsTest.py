import numpy as np
from solver.stencil.stencilSolver import stencilSolver as stencil
import pdo.pdo as pdo
import geometry.standardGeometries as stdGeom
import geometry.slabGeometry as slabGeom
import matplotlib.pyplot as plt
def c11(p):
    f = np.zeros(shape=(p.shape[0],))
    for i in range(p.shape[0]):
        f[i] = .1+p[i,0]#.1+(np.sin(np.pi*p[i,0])*np.sin(np.pi*p[i,0]))
    return f
def c22(p):
    return np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22)#,None,c1)
def l2g(p):
    return p
Om0 = stdGeom.Box([[0,0],[.001,1]])
Om=slabGeom.boxSlab(l2g,Om0.bnds,Om0)
ord=[129,65]

solver = stencil(Lapl, Om, ord)
A = solver.Aii

Nslabs = 4
Ivec = []
Ir = []
Ib=[]
Ii = []
Is = []
H = Om0.bnds[1][0]/Nslabs
for j in range(Nslabs):
    for i in range(solver.XXi.shape[0]):
        x=solver.XXi[i,0]
        if (x<(j+1)*H) and (x>j*H):
            Ii+=[i]
        if (np.abs(x-(j+1)*H)<1e-13):
            Is+=[i]
            if j%2:
                Ir+=[i]
            else:
                Ib+=[i]
#Ig = list(set(Ir).union(Ib))
#print(len(Ir),'//',len(Ib),'//',len(Ig))
#IrinIg=[]
#IbinIg=[]
#for i in range(len(Ig)):
#    if Ig[i] in Ir:
#        IrinIg+=[i]
#    if Ig[i] in Ib:
#        IbinIg+=[i]
#Igc = list(set(I)-set(Ig))
#Omsub0 = stdGeom.Box([[0,0],[H,1]])
#Omsub=slabGeom.boxSlab(l2g,Omsub0.bnds,Omsub0)
#solverSub = stencil(Lapl, Omsub, [(513-1)//Nslabs,10])

#Asub = solverSub._A
#XXsub = solverSub._XX
#for i in range(XXsub.shape[0]):
#        x=XXsub[i,0]
#        if (x<(j+1)*H) and (x>j*H):
#            I+=[i]
#        if (np.abs(x-j*H)<1e-13):
#            if j%2:
#                Ir+=[i]
#            else:
#                Ib+=[i]


#print(Ivec)
Aii = A.todense()
print("sym A: ",np.linalg.norm(Aii-Aii.T)/np.linalg.norm(Aii))
Aii_adj = Aii.T
reA = .5*(Aii+Aii_adj)
print("sym A: ",np.linalg.norm(reA-reA.T)/np.linalg.norm(reA))
T       = Aii[Is,:][:,Is]-Aii[Is,:][:,Ii]@np.linalg.solve(Aii[Ii,:][:,Ii],Aii[Ii,:][:,Is])
T_adj   = Aii_adj[Is,:][:,Is]-Aii_adj[Is,:][:,Ii]@np.linalg.solve(Aii_adj[Ii,:][:,Ii],Aii_adj[Ii,:][:,Is])


ny = ord[1]-2
Ir=[]
Ib=[]
for i in range(T.shape[0]):
    if (i//ny)%2==0:
        Ir+=[i]
    else: 
        Ib+=[i]
I = Ir+Ib
T= T[I,:][:,I]
T_adj= T_adj[I,:][:,I]
reT = np.array(.5*(T+T.T))
print("sym T : ",np.linalg.norm(reT-reT.T)/np.linalg.norm(reT))

plt.figure(1)
plt.spy(T,1e-10,markersize=1)
#plt.show()
#T=.5*(T+T.T)


S=np.zeros(shape=T.shape)
S[0:len(Ir),:][:,0:len(Ir)]=np.identity(len(Ir))
S[0:len(Ir),:][:,len(Ir):]=np.linalg.solve(T[0:len(Ir),0:len(Ir)],T[0:len(Ir),:][:,len(Ir):])
S[len(Ir):,:][:,len(Ir):]=np.identity(len(Ib))
S[len(Ir):,:][:,0:len(Ir)]=np.linalg.solve(T[len(Ir):,len(Ir):],T[len(Ir):,:][:,0:len(Ir)])

reS = .5*(S+S.T)


Trr = np.array(reT[0:len(Ir),:][:,0:len(Ir)])
Tbb = np.array(reT[len(Ir):len(I),:][:,len(Ir):len(I)])

Trb = reT[0:len(Ir),:][:,len(Ir):len(I)]
Tbr = reT[len(Ir):len(I),:][:,0:len(Ir)]
err0= np.linalg.norm(reS@reT-reT@reS,ord=2)
print('err0 = ',err0)

###################
# projection check
###################
Pr = np.zeros(S.shape)
Pr[0:len(Ir),:] = reS[0:len(Ir),:]
err1 = np.linalg.norm(Pr.T@T-T@Pr,ord=2)
err2 = np.linalg.norm(Pr@Pr-Pr)
print('err1 = ',err1)
print('err2 = ',err2)
print('nrm Pr',np.linalg.norm(Pr))


###########################
#   ip check
###########################

SS=np.zeros(shape=T.shape)
SS[0:len(Ir),:][:,0:len(Ir)]=np.identity(len(Ir))
SS[0:len(Ir),:][:,len(Ir):]=np.linalg.solve(reT[0:len(Ir),0:len(Ir)],reT[0:len(Ir),:][:,len(Ir):])
SS[len(Ir):,:][:,len(Ir):]=np.identity(len(Ib))
SS[len(Ir):,:][:,0:len(Ir)]=np.linalg.solve(reT[len(Ir):,len(Ir):],reT[len(Ir):,:][:,0:len(Ir)])

mx = 0.
mx2 = 0.
boolSum =0
for i in range(1000):
    u = np.random.standard_normal(size=(SS.shape[0],))
    ur=np.array(u[0:len(Ir)])
    ub=np.array(u[len(Ir):])
    ip1 = u.T@(reT@np.linalg.solve(SS,u))
    ip2 = u.T@(reT@np.linalg.solve(reS,u))
    ip3 = ur.T@(Trr@ur)+ub.T@(Tbb@ub)
    mx = np.max([mx,np.abs(ip1-ip3)/np.abs(ip1)])
    mx2 = np.max([mx2,np.abs(ip1-ip2)/np.abs(ip1)])
    boolSum+=(ip2<2*ip3)
print("mx = ",mx)
print("mx2 = ",mx2)
print("boolSum",boolSum)