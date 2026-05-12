import numpy as np
from matAssembly.HBS.slabTree import slabTree
from SOMSmerge import InterfaceMap, merge_S
import SOMSmerge
from SOMSmergeTest import interface_map as build_imap, bc, bc_helmholtz
import time
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
# Grid and tree
# ---------------------------------------------------------------------------
time_vec = np.zeros((0,))
Nvec = np.zeros((0,))
kh = 10.15
if kh>0.:
    bc_loc = lambda p : bc_helmholtz(p,kh)
else:
    bc_loc = lambda p : bc(p)

for k in range(7,12):
    
    N = 2**k + 1
    Nvec = np.append(Nvec,N**2)
    print("NDOFS = ",N*N)
    Nx = N; Ny = N
    p = 3
    leaf_size = 2**p
    n_levels = 2*(k-p)

    xpts = np.linspace(0,1,Nx); ypts = np.linspace(0,1,Ny)
    XY = np.zeros((len(xpts)*len(ypts),2))
    XY[:,0] = np.kron(xpts,np.ones_like(ypts))
    XY[:,1] = np.kron(np.ones_like(xpts),ypts)

    tic=time.time()
    tree = slabTree(XY, quad=False, min_leaf_size=1, max_level=n_levels)
    leaves = tree.get_leaves()
    
    print(f"Tree: {tree}")
    b0 = tree.get_node(leaves[0]).bounds
    dx = b0[1]-b0[0]; dy = b0[3]-b0[2]
    h  = 1.0/(Nx-1); nx = round(dx/h)+1; ny = round(dy/h)+1
    print(f"Leaf: dx={dx:.4f} dy={dy:.4f}  nx={nx} ny={ny}")
    print("tree done in ",time.time()-tic)
    
    tic = time.time()
    _reg = {}
    SOMSmerge.interface_map = lambda a, b: _reg.get((a,b), None)
    compute_errs = False
    errors = []
    for (a, b, ori) in tree.adjacency[-1]:
        ba = tree.get_node(a).bounds; bb = tree.get_node(b).bounds
        if ori == 'horizontal':
            left, right = (a,b) if ba[0] < bb[0] else (b,a)
            S_hor = build_imap(nx, ny, dx, dy, 'horizontal',kh)
    
            _reg[(left, right)] = S_hor
        else:
            bot, top = (a,b) if ba[2] < bb[2] else (b,a)
            S_ver = build_imap(nx, ny, dx, dy, 'vertical',kh)
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

    
    compute_errs = True
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
    time_vec = np.append(time_vec,time.time()-tic)
    print("elapsed_time = ",time.time()-tic)
fac = time_vec[0]/(Nvec[0]**(1.5))
fac2 = time_vec[0]/(Nvec[0]**(1))
fac3 = time_vec[0]/(Nvec[0]*np.log(Nvec[0]))
plt.figure(1)
plt.loglog(Nvec,time_vec,label = 'time')
plt.loglog(Nvec,fac*Nvec**(1.5),label='O(N^{1.5})')
plt.loglog(Nvec,fac2*Nvec**(1),label='O(N)')
plt.loglog(Nvec,fac3*Nvec*np.log(Nvec),label='O(NlogN)')
plt.legend()
plt.show()