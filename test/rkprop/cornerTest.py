import solver.spectralmultidomain.hps as hps
import solver.spectralmultidomain.hps.pdo as pdo
import solver.hpsmultidomain.hpsmultidomain as hps_alt
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo_alt
import numpy as np
import multislab.oms as oms
import geometry.geom_2D.square as square
import jax.numpy as jnp
import torch
import matplotlib.pyplot as plt
jax_avail = False
torch_avail = not jax_avail
hpsalt = not jax_avail
if jax_avail:
    def c11(p):
        return jnp.ones_like(p[...,0])
    def c22(p):
        return jnp.ones_like(p[...,0])
    def bc(p):
        return jnp.sin(jnp.pi*p[...,1])*jnp.sinh(jnp.pi*p[...,0]) 
    diff_op = pdo.PDO2d(c11=c11,c22=c22)
else:
    def c11(p):
        return torch.ones_like(p[:,0])
    def c22(p):
        return torch.ones_like(p[:,0])
    def bc(p):
        return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])
    diff_op = pdo_alt.PDO_2d(c11=c11,c22=c22)

a=np.array([.25,.25])
p=4
method = 'hps'
p_disc = p
if hpsalt:
    method = 'hpsalt'
    p_disc += 2

opts=oms.solverWrap.solverOptions(method,[p_disc,p_disc],a)


geom    = np.array([[0.,0.],[1.,1.]])
slab_i  = oms.slab(geom,lambda p : square.gb(p,jax_avail,torch_avail))
solver  = oms.solverWrap.solverWrapper(opts)
solver.construct(geom,diff_op)
Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)

plt.figure(1)
plt.scatter(XXi[:,0],XXi[:,1])
plt.scatter(XXb[:,0],XXb[:,1])
plt.axis('equal')
plt.show()
XXfull = solver.XXfull
plt.figure(1)
plt.scatter(XXfull[:,0],XXfull[:,1])
plt.scatter(XXi[:,0],XXi[:,1])
plt.scatter(XXb[:,0],XXb[:,1])
plt.legend(['full','i','b'])
plt.axis('equal')
plt.show()


g = np.zeros(shape=(XXb.shape[0],))
g[Igb] = bc(XXb[Igb,:])
g[Il] = bc(XXb[Il,:])
g[Ir] = bc(XXb[Ir,:])
g=g[:,np.newaxis]
print("g shape = ",g.shape)
if torch_avail:
    uu = solver.solver.solve_dir_full(torch.from_numpy(g))
    uu=uu.numpy().flatten()
else:
    uu = solver.solver.solve_dir_full(g)
    uu=uu.flatten()
uuhat = bc(XXfull)

print("uu shape = ",uu.shape)
print("XXfull shape = ",XXfull.shape)
print("uuerr = ",np.linalg.norm(uu-uuhat,ord=np.inf))