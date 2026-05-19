"""
Reminder: S maps have the form S = Solve ∘ I_seg→full.
At leaf level: segmented -> corner-excluded interface.
Above leaf level: segmented -> segmented.

SOMSHtest_quad.py — hierarchical build test for quad tree.
"""
import numpy as np
from matAssembly.HBS.slabTree import slabTree
from SOMSmerge_quad import InterfaceMap, merge_S
import SOMSmerge_quad as SOMSmerge
from SOMSmergeTest import interface_map as build_imap, bc,bc_helmholtz
import time
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
kh  = 10.

if kh>0.:
    bc_loc = lambda p:bc_helmholtz(p,kh)
else:
    bc_loc = lambda p:bc(p)
time_vec = np.zeros((0,))
Nvec = np.zeros((0,))
compute_errs = True
for k in range(7,12):
    
    N   = 2**k + 1
    Nvec = np.append(Nvec,N**2)
    p   = 4
    n_levels = k - p

    print(f"N={N}, k={k}, p={p}, n_levels={n_levels}")

    xpts = np.linspace(0, 1, N); ypts = np.linspace(0, 1, N)
    XY   = np.zeros((N*N, 2))
    XY[:,0] = np.kron(xpts, np.ones_like(ypts))
    XY[:,1] = np.kron(np.ones_like(xpts), ypts)

    tic  = time.time()
    tree = slabTree(XY, quad=True, min_leaf_size=1, max_level=n_levels)
    leaves = tree.get_leaves()
    print(f"Tree: {tree}  ({time.time()-tic:.2f}s)")

    b0 = tree.get_node(leaves[0]).bounds
    dx = b0[1]-b0[0]; dy = b0[3]-b0[2]
    h  = 1.0/(N-1);  nx = round(dx/h)+1;  ny = round(dy/h)+1
    print(f"Leaf: dx={dx:.5f} dy={dy:.5f}  nx={nx} ny={ny}")

    XX    = XY
    tol   = 1e-10

    # ---------------------------------------------------------------------------
    # Registry
    # ---------------------------------------------------------------------------
    _reg = {}
    SOMSmerge.interface_map = lambda a, b: _reg.get((a, b), None)

    # ---------------------------------------------------------------------------
    # Leaf-level S maps (segmented input, corner-excluded output)
    # ---------------------------------------------------------------------------
    S_hor = build_imap(nx, ny, dx, dy, 'horizontal', kh)
    S_ver = build_imap(nx, ny, dx, dy, 'vertical',   kh)
    print(f"S_hor: {S_hor.S.shape}, S_ver: {S_ver.S.shape}")

    def pair_bnd_pts(a_node, b_node, ori):
        """Segmented boundary pts for a leaf pair, from bDOFs."""
        if ori == 'horizontal':
            if a_node.bounds[0] > b_node.bounds[0]: a_node, b_node = b_node, a_node
            return np.vstack([XX[a_node.bDOFs['left'],  :],
                            XX[a_node.bDOFs['down'],  :],
                            XX[b_node.bDOFs['down'],  :],
                            XX[a_node.bDOFs['up'],    :],
                            XX[b_node.bDOFs['up'],    :],
                            XX[b_node.bDOFs['right'], :]])
        else:
            if a_node.bounds[2] > b_node.bounds[2]: a_node, b_node = b_node, a_node
            return np.vstack([XX[a_node.bDOFs['left'],  :],
                            XX[b_node.bDOFs['left'],  :],
                            XX[a_node.bDOFs['down'],  :],
                            XX[b_node.bDOFs['up'],    :],
                            XX[a_node.bDOFs['right'], :],
                            XX[b_node.bDOFs['right'], :]])

    def pair_iface_pts(a_node, b_node, ori):
        """Corner-excluded interface pts for a leaf pair."""
        if ori == 'horizontal':
            if a_node.bounds[0] > b_node.bounds[0]: a_node, b_node = b_node, a_node
            return XX[a_node.bDOFs['right'], :]
        else:
            if a_node.bounds[2] > b_node.bounds[2]: a_node, b_node = b_node, a_node
            return XX[a_node.bDOFs['up'], :]

    # ---------------------------------------------------------------------------
    # Register leaf-level maps and verify accuracy
    # ---------------------------------------------------------------------------
    errors_leaf = []
    for (ai, bi, ori) in tree.adjacency[-1]:
        an = tree.get_node(ai); bn = tree.get_node(bi)
        S  = S_hor if ori == 'horizontal' else S_ver
        if ori == 'horizontal':
            left, right = (ai, bi) if an.bounds[0] < bn.bounds[0] else (bi, ai)
        else:
            left, right = (ai, bi) if an.bounds[2] < bn.bounds[2] else (bi, ai)
        _reg[(left, right)] = S

        bnd = pair_bnd_pts(an, bn, ori)
        ifc = pair_iface_pts(an, bn, ori)
        if compute_errs:
            if bnd.shape[0] == S.S.shape[1] and ifc.shape[0] == S.S.shape[0]:
                err = np.max(np.abs(S.S @ bc_loc(bnd) - bc_loc(ifc)))
                errors_leaf.append(err)
    if compute_errs:
        print(f"Leaf level: {len(errors_leaf)} pairs, max err = {max(errors_leaf):.2e}")

    # ---------------------------------------------------------------------------
    # Hierarchical merge — bottom-up
    # ---------------------------------------------------------------------------
    tic = time.time()
    for level in range(tree.nlevels-2, 0, -1):
        errors = []; mismatches = 0
        for (tau_idx, sig_idx, dir) in tree.adjacency[level]:
            imap = merge_S(tau_idx, sig_idx, dir, tree)
            _reg[(tau_idx, sig_idx)] = imap

            # Test accuracy using stored iface_pts and bnd_pts from merge_S
            bnd = imap.bnd_pts if hasattr(imap, 'bnd_pts') else None
            ifc = imap.iface_pts if hasattr(imap, 'iface_pts') else None
            if compute_errs:
                if bnd is not None and ifc is not None and \
                imap.S.shape[1] == len(bnd) and imap.S.shape[0] == len(ifc):
                    err = np.max(np.abs(imap.S @ bc_loc(bnd) - bc_loc(ifc)))
                    errors.append(err)
                else:
                    mismatches += 1

        msg = f"Level {level}: {len(errors)} pairs, max err = {max(errors):.2e}" \
            if errors else f"Level {level}: no testable pairs"
        if mismatches: msg += f"  ({mismatches} shape mismatches)"
        print(msg)

    print(f"\nTotal build time: {time.time()-tic:.2f}s")
    time_vec=np.append(time_vec,time.time()-tic)
fac = time_vec[0]/(Nvec[0]**(1.5))
fac2 = time_vec[0]/(Nvec[0]**(1))
fac3 = time_vec[0]/(Nvec[0]*np.log(Nvec[0]))
plt.figure(1)
plt.loglog(Nvec,time_vec,label = 'time')
plt.loglog(Nvec,fac*Nvec**(1.5),label='O(N^{1.5})',linestyle='--')
plt.loglog(Nvec,fac3*Nvec*np.log(Nvec),label='O(NlogN)',linestyle='--')
plt.legend()
plt.savefig('timeplot')
plt.show()