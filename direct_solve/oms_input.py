"""Shared OMS input bridge for the three red-black preconditioners.

Only original-level operators are wrapped. Reduced operators are normal HBS
matrices, so neither factor chains nor local PDE solves grow across levels.
"""
from types import SimpleNamespace
import numpy as np
import torch


def _numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _geometric_tree(points, leaf_size):
    """Equal-size binary leaves, recursively split along the longest axis.

    This is the minimal tree protocol used by HBStorch_new: perm_leaf,
    nleaves, nlevels. No padded DOFs are introduced.
    """
    points = np.asarray(points)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('OMS interface_points must be a finite (m, ndim) array')
    if int(leaf_size) < 1:
        raise ValueError('leaf_size must be positive')
    m, leaves = len(points), 1
    while m % (2 * leaves) == 0 and m // (2 * leaves) >= leaf_size:
        leaves *= 2
    depth = leaves.bit_length() - 1

    def split(indices, d):
        if not d:
            return indices
        xyz = points[indices]
        axis = np.argmax(np.ptp(xyz, axis=0))
        indices = indices[np.argsort(xyz[:, axis], kind='stable')]
        mid = len(indices) // 2
        return np.concatenate((split(indices[:mid], d-1),
                               split(indices[mid:], d-1)))

    return SimpleNamespace(perm_leaf=split(np.arange(m), depth),
                           nleaves=leaves, nlevels=depth+1)


class _InputBlock:
    """Natural-order scipy operator exposed in the solver's tree order."""
    def __init__(self, op, perm):
        if np.dtype(op.dtype).kind == 'c':
            raise TypeError('These red-black/HBS implementations support real operators only')
        self.op, self.shape = op, tuple(op.shape)
        self.perm = torch.as_tensor(perm, dtype=torch.int64)
        self._p = np.asarray(perm, dtype=np.int64)

    def _apply(self, X, out=None, alpha=1., beta=0., io_perm=None, transpose=False):
        # Calls arrive in tree order. Scatter to original OMS order on input,
        # gather on output. Keep the same permutation for forward and transpose.
        if io_perm is not None and not np.array_equal(_numpy(io_perm), self._p):
            raise ValueError('Input block called with a different tree ordering')
        values = _numpy(X)
        natural = np.empty_like(values)
        natural[self._p] = values
        f = self.op.rmatmat if transpose else self.op.matmat
        result = np.asarray(f(natural))
        if result.shape != values.shape:
            raise ValueError('OMS block application returned an invalid shape')
        if np.iscomplexobj(result):
            raise TypeError('Complex OMS output is unsupported by the real HBS solver')
        result = torch.as_tensor(np.ascontiguousarray(result[self._p]),
                                 dtype=X.dtype, device=X.device)
        if out is None:
            return result.mul_(alpha)
        # Evaluate before writing out: different slab views may share storage.
        if beta == 0:
            out.copy_(result).mul_(alpha)
        else:
            out.mul_(beta).add_(result, alpha=alpha)
        return out

    def matmat(self, X, **kwargs):
        return self._apply(X, **kwargs)

    def rmatmat(self, X, **kwargs):
        return self._apply(X, transpose=True, **kwargs)


class OMSInputMixin:
    """Public constructor shared by old/new normalized and unnormalized modes.

    Required: rank strategy, built OMS. Optional operator='lu'/'hbs'; default
    follows OMS (LU if retained). tree/quad override tree inference. Otherwise
    reuse an available HBS tree; for LU-only OMS build a binary geometric tree
    from interface_points, with leaf_size default max(64, 4*initial rank).
    All remaining keyword options are passed to the existing solver.
    Construction does not factorize: call factorize() explicitly for timing.
    """
    def __init__(self, rk, oms, *, operator=None, tree=None, quad=None,
                 leaf_size=None, **options):
        if not getattr(oms, '_built', False):
            raise ValueError('Build the OMS with construct_Stot_and_rhstot first')
        sizes = list(oms.ncs)
        if len(sizes) < 2 or any(n != sizes[0] for n in sizes):
            raise ValueError('Red-black solvers require at least two equal-size OMS interfaces')
        m, n = int(sizes[0]), len(sizes)
        conn = np.asarray(oms.connectivity)
        if conn.shape != (n, 2):
            raise ValueError('OMS connectivity must have shape (nSlabs, 2)')
        path = all((int(l) < 0 if i == 0 else int(l) == i-1) and
                   (int(r) < 0 if i == n-1 else int(r) == i+1)
                   for i, (l, r) in enumerate(conn))
        cycle = all(int(l) == (i-1) % n and int(r) == (i+1) % n
                    for i, (l, r) in enumerate(conn))
        if not (path or cycle):
            raise ValueError('OMS connectivity must be an ordered path or cycle')
        if 'cyclic' in options and bool(options.pop('cyclic')) != cycle:
            raise ValueError('cyclic option disagrees with OMS connectivity')
        for i, dofs in enumerate(oms.glob_target_dofs):
            if not np.array_equal(np.asarray(list(dofs)), np.arange(i*m, (i+1)*m)):
                raise ValueError('OMS target DOFs must be contiguous in slab order')
        if len(oms.glob_target_dofs) != n:
            raise ValueError('Incomplete OMS target DOFs')
        rank = self._rank_strategy_type.coerce(rk)
        source = oms._resolve_operator(operator)
        if tree is None:
            if getattr(oms, 'constructHBS', False):
                blocks = oms.hbs_blocks_for(0)
                tree = blocks[0].tree
                if quad is None:
                    quad = bool(blocks[0].quad)
                del blocks
            else:
                if quad:
                    raise ValueError('Automatic LU tree is binary; supply a tree for quad=True')
                points = getattr(oms, 'interface_points', None)
                if points is None:
                    raise ValueError('LU-only OMS needs interface_points: rebuild with the updated oms.py or pass tree=')
                tree = _geometric_tree(_numpy(points),
                                       leaf_size if leaf_size is not None else max(64, 4*rank.rank(0)))
                quad = False
        quad = bool(quad) if quad is not None else False
        p = _numpy(tree.perm_leaf).astype(np.int64, copy=False)
        if p.shape != (m,) or not np.array_equal(np.sort(p), np.arange(m)):
            raise ValueError('Tree permutation must contain every interface DOF exactly once')
        fac, leaves = (4 if quad else 2), int(tree.nleaves)
        count, depth = leaves, 0
        while count > 1 and count % fac == 0:
            count //= fac
            depth += 1
        if leaves < 1 or count != 1 or m % leaves or int(tree.nlevels) != depth+1:
            raise ValueError('Tree must have equal-size leaves and a complete binary/quadtree hierarchy')
        super().__init__(m, rank, tree, quad, cyclic=cycle, **options)
        self.oms, self.input_operator = oms, source
        self._source_storage = oms._S_lu if source == 'lu' else oms._S_hbs
        rows = oms.direct_solver_blocks(source)
        if len(rows) != n or any(len(row) != sum(int(j) >= 0 for j in conn[i])
                                 for i, row in enumerate(rows)):
            raise ValueError('OMS block rows do not match its connectivity')
        cache, self._input_blocks = {}, []
        for row in rows:
            wrapped = []
            for op in row:
                if tuple(op.shape) != (m, m):
                    raise ValueError('All OMS off-diagonal blocks must have shape (m, m)')
                if id(op) not in cache:
                    cache[id(op)] = _InputBlock(op, p)
                wrapped.append(cache[id(op)])
            self._input_blocks.append(wrapped)

    def _check_oms(self):
        current = self.oms._S_lu if self.input_operator == 'lu' else self.oms._S_hbs
        if not getattr(self.oms, '_built', False) or current is not self._source_storage:
            raise RuntimeError('OMS was closed or rebuilt; construct a new OMS-based solver')

    def factorize(self):
        self._check_oms()
        super().factorize(self._input_blocks)
        return self

    def solve(self, rhs):
        self._check_oms()
        return super().solve(rhs)
