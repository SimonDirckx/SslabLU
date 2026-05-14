import numpy as np
from matAssembly.HBS.slabTree import slabTree
from SOMSmerge import InterfaceMap, merge_S
import SOMSmerge
import SOMSmergeTest
from SOMSmergeTest import interface_map as build_imap, bc, bc_helmholtz
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

# ---------------------------------------------------------------------------
# Grid and tree
# ---------------------------------------------------------------------------
time_vec = np.zeros((0,))
time_vec_full_stencil = np.zeros((0,))
tree_time_vec = np.zeros((0,))
err_vec = np.zeros((0,))
err_vec_full_stencil = np.zeros((0,))
hvec = np.zeros((0,))
Nvec = np.zeros((0,))
kh = 5.
if kh>0.:
    bc_loc = lambda p : bc_helmholtz(p,kh)
else:
    bc_loc = lambda p : bc(p)
timeTree = False


for k in range(9,12):
    
    N = 2**k + 1
    Nvec = np.append(Nvec,N**2)
    print("NDOFS = ",N*N)
    Nx = N; Ny = N
    p = 4
    leaf_size = 2**p
    n_levels = 2*(k-p)

    xpts = np.linspace(0,1,Nx); ypts = np.linspace(0,1,Ny)
    XY = np.zeros((len(xpts)*len(ypts),2))
    XY[:,0] = np.kron(xpts,np.ones_like(ypts))
    XY[:,1] = np.kron(np.ones_like(xpts),ypts)
    if timeTree:
        ttree = 0
        for iter in range(5):
            tic=time.time()
            tree = slabTree(XY, quad=False, min_leaf_size=1, max_level=n_levels)
            leaves = tree.get_leaves()
            ttree+=time.time()-tic
        ttree/=5
        print("tree done in ",ttree)
        tree_time_vec = np.append(tree_time_vec,ttree)
    else:
        tree = slabTree(XY, quad=False, min_leaf_size=1, max_level=n_levels)
        leaves = tree.get_leaves()
    print(f"Tree: {tree}")
    b0 = tree.get_node(leaves[0]).bounds
    dx = b0[1]-b0[0]; dy = b0[3]-b0[2]
    h  = 1.0/(Nx-1); nx = round(dx/h)+1; ny = round(dy/h)+1
    print(f"Leaf: dx={dx:.4f} dy={dy:.4f}  nx={nx} ny={ny}")
    hvec = np.append(hvec,h)
    
    tic = time.time()
    _reg = {}
    SOMSmerge.interface_map = lambda a, b: _reg.get((a,b), None)
    compute_errs = False
    errors = []
    S_hor = build_imap(nx, ny, dx, dy, 'horizontal',kh)
    S_ver = build_imap(nx, ny, dx, dy, 'vertical',kh)
    for (a, b, ori) in tree.adjacency[-1]:
        ba = tree.get_node(a).bounds; bb = tree.get_node(b).bounds
        if ori == 'horizontal':
            left, right = (a,b) if ba[0] < bb[0] else (b,a)
            _reg[(left, right)] = S_hor
        else:
            bot, top = (a,b) if ba[2] < bb[2] else (b,a)
            _reg[(bot, top)] = S_ver
        if compute_errs:
                a_node = tree.get_node(a)
                b_node = tree.get_node(b)
                if ori == 'horizontal':
                    XYb = np.vstack([XY[a_node.bDOFs['left']],
                                    XY[a_node.bDOFs['down']], XY[b_node.bDOFs['down']],
                                    XY[a_node.bDOFs['up']],   XY[b_node.bDOFs['up']],
                                    XY[b_node.bDOFs['right']]])
                    XYc = XY[a_node.bDOFs['right']]
                    err = np.max(np.abs(S_hor.S @ bc_loc(XYb) - bc_loc(XYc)))
                else:
                    XYb = np.vstack([XY[a_node.bDOFs['left']],  XY[b_node.bDOFs['left']],
                                    XY[a_node.bDOFs['down']],
                                    XY[b_node.bDOFs['up']],
                                    XY[a_node.bDOFs['right']], XY[b_node.bDOFs['right']]])
                    XYc = XY[a_node.bDOFs['up']]

                    err = np.max(np.abs(S_ver.S @ bc_loc(XYb) - bc_loc(XYc)))
                errors.append(err)
    if compute_errs:
        print("err over leaves = ",max(errors))
    

    
    compute_errs = False
    print("level ",tree.nlevels-1," done.")
    for level in range(tree.nlevels-2, 0, -1):
        errors = []
        for (tau_idx, sig_idx, dir) in tree.adjacency[level]:
            tau_node = tree.get_node(tau_idx)
            sig_node = tree.get_node(sig_idx)

            imap = merge_S(tau_idx, sig_idx, dir, tree)
            _reg[(tau_idx, sig_idx)] = imap

            if compute_errs:
                if dir == 'horizontal':
                    if tau_node.bounds[0] > sig_node.bounds[0]:
                        tau_node, sig_node = sig_node, tau_node
                    XYb = np.vstack([XY[tau_node.bDOFs['left']],
                                    XY[tau_node.bDOFs['down']], XY[sig_node.bDOFs['down']],
                                    XY[tau_node.bDOFs['up']],   XY[sig_node.bDOFs['up']],
                                    XY[sig_node.bDOFs['right']]])
                    XYc = XY[tau_node.bDOFs['right']]
                else:
                    if tau_node.bounds[2] > sig_node.bounds[2]:
                        tau_node, sig_node = sig_node, tau_node
                    XYb = np.vstack([XY[tau_node.bDOFs['left']],  XY[sig_node.bDOFs['left']],
                                    XY[tau_node.bDOFs['down']],
                                    XY[sig_node.bDOFs['up']],
                                    XY[tau_node.bDOFs['right']], XY[sig_node.bDOFs['right']]])
                    XYc = XY[tau_node.bDOFs['up']]

                if XYb.shape[0] != imap.S.shape[1]:
                    print(f"  L{level} {dir:10s} ({tau_idx:2d},{sig_idx:2d}): "
                        f"SHAPE MISMATCH XYb={XYb.shape[0]} imap_cols={imap.S.shape[1]}")
                    continue
                if XYc.shape[0] != imap.S.shape[0]:
                    print(f"  L{level} {dir:10s} ({tau_idx:2d},{sig_idx:2d}): "
                        f"IFACE MISMATCH XYc={XYc.shape[0]} imap_rows={imap.S.shape[0]}")
                    continue

                err = np.max(np.abs(imap.S @ bc_loc(XYb) - bc_loc(XYc)))
                errors.append(err)
                
        if compute_errs:
            print(f"  --> Level {level}: max err over level ={max(errors):.2e}\n")
        print("level ",level," done.")
    

    root = tree.get_node(0)
    XYb = np.vstack([   XY[root.bDOFs['left']],
                        XY[root.bDOFs['down']],
                        XY[root.bDOFs['up']],
                        XY[root.bDOFs['right']]])
    b_root = bc_loc(XYb)
    tree.set_u(root,b_root)
    tree.solve(root,interface_map=lambda t,s:SOMSmerge.interface_map(t,s),dir='vertical')
    leaves = tree.get_leaves()
    err = 0.
    for node in tree.get_nodes_level(tree.nlevels-1):
        XYb = np.vstack([   XY[node.bDOFs['left']],
                            XY[node.bDOFs['down']],
                            XY[node.bDOFs['up']],
                            XY[node.bDOFs['right']]])
        uhat_leaf = node.u
        u_node = bc_loc(XYb)
        err = max(err,np.linalg.norm(u_node-uhat_leaf,np.inf))
    print("max err over nodes = ",err)
    err_vec=np.append(err_vec,err)
    time_vec = np.append(time_vec,time.time()-tic)
    print("elapsed_time = ",time.time()-tic)
    tic = time.time()
    L = SOMSmergeTest.sparse_helmholtz_stencil(Nx,Ny,h,h,kh)
    Ii = np.where((XY[:,0]>0)&(XY[:,0]<1)&(XY[:,1]>0)&(XY[:,1]<1))[0]
    Ib = np.where((XY[:,0]==0)|(XY[:,0]==1)|(XY[:,1]==0)|(XY[:,1]==1))[0]

    Lii = L[Ii,:][:,Ii]
    Lib = L[Ii,:][:,Ib]
    ub = bc_loc(XY[Ib,:])
    ui = bc_loc(XY[Ii,:])
    lu = splinalg.splu(Lii)
    uhat = -lu.solve(Lib@ub)
    print("err of full stencil sol = ",np.linalg.norm(ui-uhat,np.inf))
    err_vec_full_stencil = np.append(err_vec_full_stencil,np.linalg.norm(ui-uhat,np.inf))
    time_vec_full_stencil = np.append(time_vec_full_stencil,time.time()-tic)

fac = time_vec[0]/(Nvec[0]**(1.5))
fac2 = time_vec[0]/(Nvec[0]**(1))
fac3 = time_vec[0]/(Nvec[0]*np.log(Nvec[0]))
plt.figure(1)
plt.loglog(Nvec,time_vec,label = 'time')
#plt.loglog(Nvec,time_vec_full_stencil,label = 'time stencil')
plt.loglog(Nvec,fac*Nvec**(1.5),label='O(N^{1.5})',linestyle='--')
plt.loglog(Nvec,fac3*Nvec*np.log(Nvec),label='O(NlogN)',linestyle='--')
plt.legend()
plt.savefig('timeplot')
plt.show()

fach = 1.1*err_vec[-1]/(hvec[-1]**2)
plt.figure(2)
plt.loglog(Nvec,err_vec,label = 'err_domino')
plt.loglog(Nvec,err_vec,label = 'err_stencil')
plt.loglog(Nvec,fach*(hvec**2),label='O(h^2)',linestyle='--')
plt.legend()
plt.savefig('errplot')


if timeTree:
    fac =  tree_time_vec[0]/(Nvec[0]**(1.5))
    fac2 = tree_time_vec[0]/(Nvec[0]**(1))
    fac3 = tree_time_vec[0]/(Nvec[0]*np.log(Nvec[0]))

    plt.figure(3)
    plt.loglog(Nvec,tree_time_vec,label = 'time')
    plt.loglog(Nvec,fac*Nvec**(1.5),label='O(N^{1.5})')
    plt.loglog(Nvec,fac2*Nvec**(1),label='O(N)')
    plt.loglog(Nvec,fac3*Nvec*np.log(Nvec),label='O(NlogN)')
    plt.legend()

#plt.show()

'''
starts at k=6
time_vec = np.array([   0.12151765823364258,
                        0.5961408615112305,
                        4.538544654846191,
                        26.474758625030518,
                        146.1588625907898,
                        635.6061873435974,
                        1775.2131607532501,
                        6299.312522888184
                     ])
'''