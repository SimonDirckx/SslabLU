"""Per-level memoization of immutable operator samples (no Gram matrices).

Keys describe ordered operator expressions and probe column intervals, never
temporary tensor addresses. Cached values live on the host. A containing
interval can serve a smaller request, e.g. coupling probes prefixing diagonal
probes. LRU eviction bounds additional host memory without changing results.
"""
from collections import OrderedDict


class SampleGraph:
    def __init__(self, apply, solve, roots, max_bytes=None):
        if max_bytes is not None and int(max_bytes) < 0:
            raise ValueError('sample_cache_bytes must be nonnegative or None')
        self.apply_op, self.solve_op = apply, solve
        self.roots, self.ops = roots, {}
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.cache = OrderedDict()
        self.bytes = 0
        self.stats = dict(hits=0, apply_calls=0, solve_calls=0,
                          apply_columns=0, solve_columns=0,
                          evictions=0, peak_bytes=0)

    def root(self, transpose=False):
        return ('root', 'Psi' if transpose else 'Om')

    def apply(self, operators, child, transpose=False):
        ids = tuple(id(op) for op in operators)
        for op in operators:
            self.ops[id(op)] = op
        return ('apply', ids, bool(transpose), child)

    def solve(self, operator, child, transpose=False):
        if operator is None:
            return child
        self.ops[id(operator)] = operator
        return ('solve', id(operator), bool(transpose), child)

    def value(self, expr, start=0, stop=None):
        if stop is None:
            stop = self.roots['Om'].shape[1]
        if expr[0] == 'root':
            return self.roots[expr[1]][:, start:stop]
        key = (expr, start, stop)
        hit = key if key in self.cache else next(
            (k for k in self.cache if k[0] == expr and k[1] <= start and stop <= k[2]), None)
        if hit is not None:
            self.stats['hits'] += 1
            result = self.cache[hit][:, start-hit[1]:stop-hit[1]]
            self.cache.move_to_end(hit)
            return result
        kind, ids, transpose, child = expr
        X = self.value(child, start, stop)
        if kind == 'apply':
            result = self.apply_op([self.ops[i] for i in ids], X, T=transpose)
        else:
            # ULV solves are in-place: never overwrite a cached child or probe.
            result = self.solve_op(self.ops[ids], X.clone(),
                                   mode='T' if transpose else 'N')
        operations = len(ids) if kind == 'apply' else 1
        self.stats[kind + '_calls'] += operations
        self.stats[kind + '_columns'] += operations * (stop-start)
        # No GPU-resident cache, even when the old solver uses GPU work arrays.
        result = result.detach().cpu()
        size = result.numel() * result.element_size()
        if self.max_bytes is None or size <= self.max_bytes:
            while self.cache and self.max_bytes is not None and self.bytes+size > self.max_bytes:
                _, old = self.cache.popitem(last=False)
                self.bytes -= old.numel() * old.element_size()
                self.stats['evictions'] += 1
            self.cache[key] = result
            self.bytes += size
            self.stats['peak_bytes'] = max(self.stats['peak_bytes'], self.bytes)
        return result
