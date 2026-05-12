import numpy as np


class InterfaceMap:
    """
    Maps the 4-segment outer boundary [left, down, up, right] of a merged
    domain to its interface DOFs.
    """
    def __init__(self, S: np.ndarray, seg_starts):
        self.S = S
        self.seg_starts = list(seg_starts)

    def seg_slice(self, k: int) -> slice:
        end = self.S.shape[1] if k == len(self.seg_starts)-1 else self.seg_starts[k+1]
        return slice(self.seg_starts[k], end)

    def seg_len(self, k: int) -> int:
        sl = self.seg_slice(k)
        return sl.stop - sl.start


class Node:
    def __init__(self, children=None):
        self.children = children


def interface_map(a, b):
    raise NotImplementedError


def _outer_of_pair(p_node, q_node, direction):
    """
    Outer boundary indices and 4-segment sizes for an adjacent pair (p, q).
    direction: 'horizontal' (p LEFT of q) or 'vertical' (p BELOW q).
    Returns (outer_idxs, n_l, n_d, n_u, n_r).
    """
    if direction == 'horizontal':
        outer = np.concatenate([p_node.bDOFs['left'],
                                p_node.bDOFs['down'], q_node.bDOFs['down'],
                                p_node.bDOFs['up'],   q_node.bDOFs['up'],
                                q_node.bDOFs['right']])
        n_l = len(p_node.bDOFs['left'])
        n_d = len(p_node.bDOFs['down']) + len(q_node.bDOFs['down'])
        n_u = len(p_node.bDOFs['up'])   + len(q_node.bDOFs['up'])
        n_r = len(q_node.bDOFs['right'])
    else:
        outer = np.concatenate([p_node.bDOFs['left'],  q_node.bDOFs['left'],
                                p_node.bDOFs['down'],
                                q_node.bDOFs['up'],
                                p_node.bDOFs['right'], q_node.bDOFs['right']])
        n_l = len(p_node.bDOFs['left'])  + len(q_node.bDOFs['left'])
        n_d = len(p_node.bDOFs['down'])
        n_u = len(q_node.bDOFs['up'])
        n_r = len(p_node.bDOFs['right']) + len(q_node.bDOFs['right'])
    return outer, n_l, n_d, n_u, n_r


def merge_S(tau_idx, sig_idx, direction, tree):
    """
    Merge adjacent tree nodes tau and sig via Schur complement.

    Parameters
    ----------
    tau_idx, sig_idx : int   Tree node indices.
    direction : str          'horizontal' (tau LEFT) or 'vertical' (tau BELOW).
    tree : slabTree

    Returns
    -------
    InterfaceMap with 4-segment boundary [left, down, up, right] of tau∪sig
    mapping to the tau/sig shared interface.
    """
    tau_node = tree.get_node(tau_idx)
    sig_node = tree.get_node(sig_idx)

    # ------------------------------------------------------------------
    # Outer boundary of tau∪sig and target interface
    # ------------------------------------------------------------------
    outer, n_l, n_d, n_u, n_r = _outer_of_pair(tau_node, sig_node, direction)
    n_ext = len(outer)
    outer_pos = {int(v): k for k, v in enumerate(outer)}

    if direction == 'horizontal':
        tgt_idxs = tau_node.bDOFs['right']   # = sig_node.bDOFs['left']
    else:
        tgt_idxs = tau_node.bDOFs['up']      # = sig_node.bDOFs['down']

    # ------------------------------------------------------------------
    # Internal interfaces: tau.iDOFs (shared between tau's children)
    #                      sig.iDOFs (shared between sig's children)
    # These will be Schur-eliminated.
    # ------------------------------------------------------------------
    int_idxs = np.concatenate([tau_node.iDOFs, sig_node.iDOFs])
    n_int = len(int_idxs)
    n_tgt = len(tgt_idxs)
    n_all = n_int + n_tgt

    all_pos = {}
    for k, v in enumerate(int_idxs):
        all_pos[int(v)] = k
    for k, v in enumerate(tgt_idxs):
        all_pos[int(v)] = n_int + k

    # ------------------------------------------------------------------
    # Collect all adjacent child pairs and their registered imaps.
    # For each pair (p, q) find its direction from bounds.
    # ------------------------------------------------------------------
    children = [c.index for c in tau_node.children] + \
               [c.index for c in sig_node.children]

    tol = 1e-10
    def child_dir(p, q):
        bp = tree.get_node(p).bounds; bq = tree.get_node(q).bounds
        if abs(bp[1] - bq[0]) < tol: return 'horizontal'
        if abs(bq[1] - bp[0]) < tol: return 'horizontal'
        return 'vertical'

    adj_pairs = []   # (p, q, imap, iface_idxs, pair_direction)
    seen = set()
    for i in range(len(children)):
        for j in range(i+1, len(children)):
            p, q = children[i], children[j]
            key = (min(p,q), max(p,q))
            if key in seen: continue
            imap = interface_map(p, q)
            if imap is None:
                imap = interface_map(q, p)
                if imap is not None: p, q = q, p
            if imap is None: continue
            seen.add(key)
            # iface_idxs = interface between p and q = shared boundary
            pd = child_dir(p, q)
            p_node = tree.get_node(p); q_node = tree.get_node(q)
            if pd == 'horizontal':
                iface_idxs = p_node.bDOFs['right']   # = q_node.bDOFs['left']
            else:
                iface_idxs = p_node.bDOFs['up']      # = q_node.bDOFs['down']
            adj_pairs.append((p, q, imap, iface_idxs, pd))

    # ------------------------------------------------------------------
    # Assemble A_ext (n_all x n_ext) and A_int (n_all x n_all).
    #
    # For each pair (p, q) with map S_pq (n_iface x n_bnd):
    #   u_pq = S_pq @ f_pq_bnd
    # where f_pq_bnd is evaluated at the outer boundary of p∪q,
    # and each DOF in that boundary is either in outer_pos (-> A_ext col)
    # or in all_pos (-> A_int col).
    # ------------------------------------------------------------------
    A_ext = np.zeros((n_all, n_ext))
    A_int = np.zeros((n_all, n_all))

    for (p, q, imap, iface_idxs, pd) in adj_pairs:
        p_node = tree.get_node(p); q_node = tree.get_node(q)
        pq_outer, *_ = _outer_of_pair(p_node, q_node, pd)

        S = imap.S   # (n_iface, n_bnd_pq)

        # Row positions in [u_int; u_tgt] for this pair's iface DOFs
        rows = []
        row_mask = []
        for v in iface_idxs:
            pos = all_pos.get(int(v))
            rows.append(pos)
            row_mask.append(pos is not None)
        rows = [r for r in rows if r is not None]
        S_rows = S[row_mask, :]

        # Decompose each boundary column
        for col, v in enumerate(pq_outer):
            v = int(v)
            col_vec = S_rows[:, col]
            if v in outer_pos:
                A_ext[rows, outer_pos[v]] += col_vec
            elif v in all_pos:
                A_int[rows, all_pos[v]]   += col_vec

    # ------------------------------------------------------------------
    # Schur complement solve:
    # (I - A_int) [u_int; u_tgt] = A_ext f_ext
    # Extract the last n_tgt rows -> S_merged
    # ------------------------------------------------------------------
    U_all   = np.linalg.solve(np.eye(n_all) - A_int, A_ext)
    S_final = U_all[n_int:, :]   # (n_tgt, n_ext)

    seg_starts = [0, n_l, n_l+n_d, n_l+n_d+n_u]
    return InterfaceMap(S_final, seg_starts)


# Keep old-style horizontal/vertical merge as thin wrappers for compatibility
def horizontal_merge(tau, sig):
    raise NotImplementedError("Use merge_S with tree access instead")

def vertical_merge(tau, sig):
    raise NotImplementedError("Use merge_S with tree access instead")