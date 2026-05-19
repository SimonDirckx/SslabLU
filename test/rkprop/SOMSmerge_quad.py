"""
Reminder: S maps have the form S = Solve ∘ I_seg→full.
At leaf level: input segmented, output corner-excluded interface.
Above leaf level: input segmented, output segmented (the output segments
are the bDOFs of the children on the shared interface, already split at
the midpoint by the tree structure).

SOMSmerge_quad.py — hierarchical S-map merge for quad trees.

For a merge of (tau, sig):
  - tau and sig each have 4 children: SW, SE, NW, NE
  - There are 10 adjacent child pairs: 8 sibling + 2 cross
  - The equilibrium system routes each pair's boundary columns to either:
      f_ext: outer boundary of tau∪sig (segmented bDOFs of outer children)
      u:     internal + target interface DOFs
  - Cross-points (corners of 4 children) are added to u_int with stencil eqns
  - Schur complement eliminates u_int, yielding S_merged: f_ext -> u_tgt

The segmented outer boundary f_ext is built from the bDOFs of the outermost
children, exactly as in the 8-box test.

The target interface u_tgt is the shared boundary between tau and sig,
represented as bDOFs of the children facing the interface — already segmented.
"""
import numpy as np


class InterfaceMap:
    """Maps segmented outer boundary of tau∪sig to segmented shared interface."""
    def __init__(self, S: np.ndarray, seg_starts, iface_pts=None):
        self.S = S
        self.seg_starts = list(seg_starts)
        self.iface_pts  = iface_pts   # (n_tgt,2) coordinates of output DOFs

    def seg_slice(self, k: int) -> slice:
        end = self.S.shape[1] if k == len(self.seg_starts)-1 \
              else self.seg_starts[k+1]
        return slice(self.seg_starts[k], end)

    def seg_len(self, k: int) -> int:
        sl = self.seg_slice(k)
        return sl.stop - sl.start


def interface_map(a, b):
    raise NotImplementedError


def _get_bnd_pts(node, side, XX, col_x, col_y):
    """Return coordinate array for a node's bDOFs on one side."""
    return XX[node.bDOFs[side], :]


def merge_S(tau_idx, sig_idx, direction, tree):
    """
    Merge adjacent quad-tree nodes tau and sig via Schur complement.

    Each child pair's S map takes segmented boundary as input (via coordinate
    matching using the tree's bDOFs) and outputs corner-excluded interface DOFs.
    The merged S map takes the segmented outer boundary of tau∪sig as input
    and outputs the segmented shared interface (bDOFs of the children on the
    shared boundary, split at the cross-point).

    Parameters
    ----------
    tau_idx, sig_idx : int
    direction : 'horizontal' (tau LEFT of sig) or 'vertical' (tau BELOW sig)
    tree : slabTree (quad=True)
    """
    tau_node = tree.get_node(tau_idx)
    sig_node = tree.get_node(sig_idx)

    # Children: SW=0, SE=1, NW=2, NE=3
    a, b, c, d = tau_node.children
    e, f, g, h = sig_node.children


    # ------------------------------------------------------------------
    # 1.  Target interface: the bDOFs of the children facing the shared side.
    #     This gives the segmented output (two sub-segments split at midpoint).
    #
    #     Horizontal (tau LEFT of sig):
    #       u_tgt_b = tau_se.bDOFs['right'] = tau_ne.bDOFs['right']  <- no, it's:
    #       u_tgt_b = tau_se.bDOFs['right']   (y in lower half)
    #       u_tgt_t = tau_ne.bDOFs['right']   (y in upper half)
    #     Vertical (tau BELOW sig):
    #       u_tgt_l = tau_nw.bDOFs['up']      (x in left half)
    #       u_tgt_r = tau_ne.bDOFs['up']      (x in right half)
    # ------------------------------------------------------------------
    if direction == 'horizontal':
        tgt_b_idx = tau_se.bDOFs['right']
        tgt_t_idx = tau_ne.bDOFs['right']
        tgt_idxs  = np.concatenate([tgt_b_idx, tgt_t_idx])
        tgt_pts   = XX[tgt_idxs, :]
    else:
        tgt_l_idx = tau_nw.bDOFs['up']
        tgt_r_idx = tau_ne.bDOFs['up']
        tgt_idxs  = np.concatenate([tgt_l_idx, tgt_r_idx])
        tgt_pts   = XX[tgt_idxs, :]

    # ------------------------------------------------------------------
    # 2.  Internal interface DOFs: all child-pair interfaces that will be
    #     Schur-eliminated.
    #
    #     Sibling pairs within tau:  SW-SE (H), NW-NE (H), SW-NW (V), SE-NE (V)
    #     Sibling pairs within sig:  same 4 pairs
    #     Each pair's interface = bDOFs of the left/bottom child's right/up side.
    #
    #     Cross-points: the 4+1 interior corners (corners of 4 children each).
    #     For horizontal merge: cross-points at the intersections of all split lines
    #     within tau and sig, plus the cross-point on the shared boundary
    #     (which is in tgt but already excluded by the [1:-1] of bDOFs).
    # ------------------------------------------------------------------
    # Sibling interfaces
    int_parts = []

    # Within tau
    int_parts.append(XX[tau_sw.bDOFs['right'], :])   # SW-SE horizontal
    int_parts.append(XX[tau_nw.bDOFs['right'], :])   # NW-NE horizontal
    int_parts.append(XX[tau_sw.bDOFs['up'],    :])   # SW-NW vertical
    int_parts.append(XX[tau_se.bDOFs['up'],    :])   # SE-NE vertical
    # Within sig
    int_parts.append(XX[sig_sw.bDOFs['right'], :])   # SW-SE horizontal
    int_parts.append(XX[sig_nw.bDOFs['right'], :])   # NW-NE horizontal
    int_parts.append(XX[sig_sw.bDOFs['up'],    :])   # SW-NW vertical
    int_parts.append(XX[sig_se.bDOFs['up'],    :])   # SE-NE vertical

    # Cross-points within tau and sig (centre of each 2x2 block)
    # Each is at the intersection of the two split lines, so not in any bDOFs
    def cross_pt(node):
        xmid = 0.5*(node.bounds[0]+node.bounds[1])
        ymid = 0.5*(node.bounds[2]+node.bounds[3])
        return np.array([[xmid, ymid]])

    tau_cp = cross_pt(tau_node)
    sig_cp = cross_pt(sig_node)

    u_int_pts_list = int_parts + [tau_cp, sig_cp]
    u_int_pts = np.vstack(u_int_pts_list)

    # ------------------------------------------------------------------
    # 3.  Outer boundary f_ext: segmented bDOFs of the outer-facing children.
    #
    #     Horizontal merge (tau LEFT, sig RIGHT):
    #       left:   tau_sw.left (bottom half) + tau_nw.left (top half)
    #       bot_tau: tau_sw.down + tau_se.down   (split at x_mid of tau)
    #       bot_sig: sig_sw.down + sig_se.down
    #       top_tau: tau_nw.up  + tau_ne.up
    #       top_sig: sig_nw.up  + sig_ne.up
    #       right:  sig_se.right (bottom half) + sig_ne.right (top half)
    #
    #     Vertical merge (tau BELOW, sig ABOVE):
    #       left_tau: tau_sw.left + tau_nw.left
    #       left_sig: sig_sw.left + sig_nw.left
    #       bot:    tau_sw.down + tau_se.down
    #       top:    sig_nw.up  + sig_ne.up
    #       right_tau: tau_se.right + tau_ne.right
    #       right_sig: sig_se.right + sig_ne.right
    # ------------------------------------------------------------------
    if direction == 'horizontal':
        ext_parts_pts = [
            XX[tau_sw.bDOFs['left'],  :],   # left bottom
            XX[tau_nw.bDOFs['left'],  :],   # left top
            XX[tau_sw.bDOFs['down'],  :],   # bot tau-left
            XX[tau_se.bDOFs['down'],  :],   # bot tau-right
            XX[sig_sw.bDOFs['down'],  :],   # bot sig-left
            XX[sig_se.bDOFs['down'],  :],   # bot sig-right
            XX[tau_nw.bDOFs['up'],    :],   # top tau-left
            XX[tau_ne.bDOFs['up'],    :],   # top tau-right
            XX[sig_nw.bDOFs['up'],    :],   # top sig-left
            XX[sig_ne.bDOFs['up'],    :],   # top sig-right
            XX[sig_se.bDOFs['right'], :],   # right bottom
            XX[sig_ne.bDOFs['right'], :],   # right top
        ]
    else:
        ext_parts_pts = [
            XX[tau_sw.bDOFs['left'],  :],   # left tau-bottom
            XX[tau_nw.bDOFs['left'],  :],   # left tau-top
            XX[sig_sw.bDOFs['left'],  :],   # left sig-bottom
            XX[sig_nw.bDOFs['left'],  :],   # left sig-top
            XX[tau_sw.bDOFs['down'],  :],   # bot left
            XX[tau_se.bDOFs['down'],  :],   # bot right
            XX[sig_nw.bDOFs['up'],    :],   # top left
            XX[sig_ne.bDOFs['up'],    :],   # top right
            XX[tau_se.bDOFs['right'], :],   # right tau-bottom
            XX[tau_ne.bDOFs['right'], :],   # right tau-top
            XX[sig_se.bDOFs['right'], :],   # right sig-bottom
            XX[sig_ne.bDOFs['right'], :],   # right sig-top
        ]

    f_ext_pts = np.vstack(ext_parts_pts)
    ext_sizes = [len(p) for p in ext_parts_pts]

    # ------------------------------------------------------------------
    # 4.  Build lookup dicts and assemble system
    # ------------------------------------------------------------------
    u_all_pts = np.vstack([u_int_pts, tgt_pts])
    n_int = len(u_int_pts); n_tgt = len(tgt_pts)
    n_all = n_int + n_tgt; n_ext = len(f_ext_pts)

    def ckey(p):
        return (round(float(p[0])/tol)*tol, round(float(p[1])/tol)*tol)

    ext_lkp = {ckey(p): i for i, p in enumerate(f_ext_pts)}
    all_lkp = {ckey(p): i for i, p in enumerate(u_all_pts)}

    A = np.zeros((n_all, n_ext))
    M = np.zeros((n_all, n_all))

    def assemble(S, bnd_pts, iface_pts):
        rows = []; rmask = []
        for p in iface_pts:
            idx = all_lkp.get(ckey(p))
            rows.append(idx); rmask.append(idx is not None)
        rows   = [r for r in rows if r is not None]
        S_rows = S[rmask, :]
        ext_c=[]; ext_t=[]; int_c=[]; int_t=[]
        for col, p in enumerate(bnd_pts):
            k  = ckey(p)
            ei = ext_lkp.get(k)
            if ei is not None:
                ext_c.append(col); ext_t.append(ei)
            else:
                ai = all_lkp.get(k)
                if ai is not None:
                    int_c.append(col); int_t.append(ai)
        if ext_c: A[np.ix_(rows, ext_t)] += S_rows[:, ext_c]
        if int_c: M[np.ix_(rows, int_t)] += S_rows[:, int_c]

    # ------------------------------------------------------------------
    # 5.  Child pair S maps and their boundary/interface DOFs.
    #
    #     Each child pair's S map takes segmented boundary (from bDOFs
    #     of that pair's outer children) as input, assembled by coordinate
    #     matching against ext_lkp and all_lkp.
    #
    #     The boundary of each child pair = union of its 6 boundary segments.
    #     We retrieve these from the registered S maps which store iface_pts.
    # ------------------------------------------------------------------
    # All 10 child pairs: 8 sibling + 2 cross
    if direction == 'horizontal':
        cross_pairs = [
            (tau_se.index, sig_sw.index, 'horizontal'),
            (tau_ne.index, sig_nw.index, 'horizontal'),
        ]
    else:
        cross_pairs = [
            (tau_nw.index, sig_sw.index, 'vertical'),
            (tau_ne.index, sig_se.index, 'vertical'),
        ]

    sibling_pairs = [
        (tau_sw.index, tau_se.index, 'horizontal'),
        (tau_nw.index, tau_ne.index, 'horizontal'),
        (tau_sw.index, tau_nw.index, 'vertical'),
        (tau_se.index, tau_ne.index, 'vertical'),
        (sig_sw.index, sig_se.index, 'horizontal'),
        (sig_nw.index, sig_ne.index, 'horizontal'),
        (sig_sw.index, sig_nw.index, 'vertical'),
        (sig_se.index, sig_ne.index, 'vertical'),
    ]

    for (p_idx, q_idx, pd) in sibling_pairs + cross_pairs:
        imap = interface_map(p_idx, q_idx)
        if imap is None:
            imap = interface_map(q_idx, p_idx)
            if imap is not None: p_idx, q_idx = q_idx, p_idx
        if imap is None:
            continue

        # bnd_pts = segmented boundary of pair (p,q), from their bDOFs
        p_node = tree.get_node(p_idx); q_node = tree.get_node(q_idx)
        if pd == 'horizontal':
            # p LEFT of q
            if p_node.bounds[0] > q_node.bounds[0]:
                p_node, q_node = q_node, p_node
            # Segmented boundary: [left(p), down(p), down(q), up(p), up(q), right(q)]
            bnd_pts = np.vstack([
                XX[p_node.bDOFs['left'],  :],
                XX[p_node.bDOFs['down'],  :],
                XX[q_node.bDOFs['down'],  :],
                XX[p_node.bDOFs['up'],    :],
                XX[q_node.bDOFs['up'],    :],
                XX[q_node.bDOFs['right'], :],
            ])
            # iface = p_node's right side (corner-excluded) = q_node's left side
            iface_pts = XX[p_node.bDOFs['right'], :]
        else:
            # p BELOW q
            if p_node.bounds[2] > q_node.bounds[2]:
                p_node, q_node = q_node, p_node
            bnd_pts = np.vstack([
                XX[p_node.bDOFs['left'],  :],
                XX[q_node.bDOFs['left'],  :],
                XX[p_node.bDOFs['down'],  :],
                XX[q_node.bDOFs['up'],    :],
                XX[p_node.bDOFs['right'], :],
                XX[q_node.bDOFs['right'], :],
            ])
            iface_pts = XX[p_node.bDOFs['up'], :]

        # Check shape match
        if imap.S.shape[1] != len(bnd_pts):
            continue

        assemble(imap.S, bnd_pts, iface_pts)

    # ------------------------------------------------------------------
    # 6.  Stencil equations for internal cross-points.
    # ------------------------------------------------------------------
    #for cp in [tau_cp[0], sig_cp[0]]:
    #    cp_row = all_lkp.get(ckey(cp))
    #    if cp_row is None:
    #        continue
    #    # find h from the nearest bDOF
    #    for dxy in [(h, 0), (-h, 0), (0, h), (0, -h)]:
    #        nb = cp + np.array(dxy)
    #        k  = ckey(nb)
    #        ei = ext_lkp.get(k)
    #        if ei is not None:      A[cp_row, ei] += 0.25
    #        else:
    #            ai = all_lkp.get(k)
    #            if ai is not None:  M[cp_row, ai] += 0.25

    # ------------------------------------------------------------------
    # 7.  Schur complement
    # ------------------------------------------------------------------
    I_minus_M = np.eye(n_all) - M
    M_uu = I_minus_M[:n_int, :n_int];  M_ut = I_minus_M[:n_int, n_int:]
    M_tu = I_minus_M[n_int:, :n_int];  M_tt = I_minus_M[n_int:, n_int:]
    B_u  = A[:n_int, :];                B_t  = A[n_int:,  :]

    X     = np.linalg.solve(M_uu, np.hstack([M_ut, B_u]))
    schur = M_tt - M_tu @ X[:, :n_tgt]
    rhs_t = B_t  - M_tu @ X[:, n_tgt:]
    S_fin = np.linalg.solve(schur, rhs_t)

    # ------------------------------------------------------------------
    # 8.  Build seg_starts for the output (segmented target interface)
    #     and store bnd_pts/iface_pts for use at next level.
    # ------------------------------------------------------------------
    if direction == 'horizontal':
        # Output: [tgt_b(right of SE), tgt_t(right of NE)]
        seg_starts = [0, len(tgt_b_idx)]
    else:
        # Output: [tgt_l(up of NW), tgt_r(up of NE)]
        seg_starts = [0, len(tgt_l_idx)]

    # Build the segmented outer boundary pts and seg_starts for f_ext
    ext_seg_starts = [0]
    for sz in ext_sizes[:-1]:
        ext_seg_starts.append(ext_seg_starts[-1] + sz)

    imap_out = InterfaceMap(S_fin, ext_seg_starts, iface_pts=tgt_pts)
    imap_out.bnd_pts   = f_ext_pts   # segmented outer boundary coords
    imap_out.tgt_seg_starts = seg_starts  # segmentation of the output
    return imap_out