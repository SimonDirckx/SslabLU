"""
Approach 1 (pb_fixed_operator.tex, Sec. 5, Algorithm 1): fixed-point iteration
on the OMS interface traces, local nonlinear solves in condensed form.

    R(u) = u - E(u),     E_i(u) = midplane trace of the local nonlinear solve
                                  on double slab i with data u_{i-1}, u_{i+1}
    u_{k+1} = u_k - P R(u_k),     P ~ (I - K(c_ref))^{-1}  (red-black / exact / I)

Everything per slab is reused from the OMS built for -Lap + c_ref with the
'cylhps' fast local solver (CylinderSlabOps): the MUMPS factor (solver_ii),
Aib, the face index sets, and -- through body_reducer.ops -- the leaf operators
for the body-load reduction and the reconstruction.  No new factorizations.
"""
import time
import numpy as np
from types import SimpleNamespace


class InterfaceIteration:
    def __init__(self, OMS, mms, bc, c_of_slab, phi0_of_slab):
        """OMS        : built multislab.oms.oms (keepLU=True, cylhps fast)
        mms        : PBManufactured (f*, g, d_s g, phi*)
        bc         : wall data callable (numpy OMS points -> values)
        c_of_slab  : dict slab -> c_ref at that slab's leaf nodes (as used in set_c)
        phi0_of_slab: dict slab -> initial leaf-grid field (P phi_c)"""
        self.OMS, self.mms = OMS, mms
        self.N = len(OMS.slabList)
        self.gtd = [np.asarray(list(g), dtype=np.int64) if not isinstance(g, slice)
                    else np.arange(g.start, g.stop) for g in OMS.glob_target_dofs]
        self.slabs = []
        for i in range(self.N):
            hnd = OMS._lu_handles[i].get()
            ops = hnd.body_reducer.ops
            info = OMS._slab_info[i]
            W = ops.nodes()
            psi = mms.psi(W)
            self.slabs.append(SimpleNamespace(
                hnd=hnd, ops=ops, info=info, W=W, psi=psi, I=ops.interior_mask(),
                fst=mms.fstar(W), c=np.asarray(c_of_slab[i]), phi=np.asarray(phi0_of_slab[i]).copy(),
                g_wall=np.asarray(bc(info.pts_gb)).reshape(-1),
                L=OMS.connectivity[i][0], R=OMS.connectivity[i][1]))
        self.timers = dict(local=0.0, precond=0.0)

    # ------------------------------------------------------------ pieces
    def traces_of_fields(self):
        """Interface vector from the current local fields (skeleton values of
        each slab's leaf field at its midplane dofs)."""
        u = np.empty(self.OMS.Ntot)
        for i, s in enumerate(self.slabs):
            sk = s.ops.skeleton_from_field(s.phi)
            u[self.gtd[i]] = sk[s.ops.Ii][s.info.Ic]
        return u

    def boundary_data(self, i, u):
        s = self.slabs[i]
        ub = np.zeros(len(s.ops.Ib))
        if s.L >= 0:
            ub[s.info.Il] = u[self.gtd[s.L]]
        if s.R >= 0:
            ub[s.info.Ir] = u[self.gtd[s.R]]
        ub[s.info.Igb] = s.g_wall
        return ub

    def local_solve(self, i, u, tol, maxit):
        """Chord iteration on slab i, warm-started from its stored field.
        Returns (midplane trace E_i, number of chord steps)."""
        s, mms = self.slabs[i], self.mms
        ub = self.boundary_data(i, u)
        x = np.empty(s.ops.ndof)
        x[s.ops.Ib] = ub
        phi = s.phi
        for j in range(maxit):
            h = s.fst - mms.g(s.W, phi, s.psi) + s.c * phi
            _, beta, z = s.ops.reduce_body(h, ub, return_z=True)
            xI = np.asarray(s.hnd.solver_ii @ beta).reshape(-1)
            x[s.ops.Ii] = xI
            phi_new = s.ops.reconstruct(x, z=z)
            inc = np.max(np.abs(phi_new - phi))
            phi = phi_new
            if inc < tol:
                break
        s.phi = phi
        return xI[s.info.Ic], j + 1

    def E(self, u, tol, maxit):
        out = np.empty_like(u)
        steps = []
        for i in range(self.N):
            Ei, k = self.local_solve(i, u, tol, maxit)
            out[self.gtd[i]] = Ei
            steps.append(k)
        return out, steps

    # ------------------------------------------------------------ outer loop
    def run(self, P, u0, u_exact=None, tol=1e-10, maxit=50, eta=0.1,
            local_tol_min=1e-13, local_maxit=50, verbose=True):
        """P: callable v -> approx (I - K)^{-1} v (identity: lambda v: v).
        Local tolerance: max(local_tol_min, eta * |R(u_{k-1})|_inf)."""
        u = np.asarray(u0, float).copy()
        logs, r_prev = [], 1.0
        for k in range(maxit):
            tloc = max(local_tol_min, eta * r_prev)
            t0 = time.perf_counter()
            Eu, steps = self.E(u, tloc, local_maxit)
            t_loc = time.perf_counter() - t0
            Rk = u - Eu
            r = np.max(np.abs(Rk))
            log = dict(outer=k, residual=r, local_tol=tloc, local_steps_max=max(steps),
                       local_steps_mean=float(np.mean(steps)), t_local=t_loc)
            if u_exact is not None:
                log["err_iface"] = np.max(np.abs(u - u_exact))
            if r < tol:
                logs.append(log)
                if verbose:
                    self._print(log)
                break
            t0 = time.perf_counter()
            u = u - np.asarray(P(Rk)).reshape(-1)
            log["t_precond"] = time.perf_counter() - t0
            logs.append(log)
            if verbose:
                self._print(log)
            r_prev = r
        return u, logs

    @staticmethod
    def _print(l):
        print("  outer %3d  |R| %.2e  %slocal tol %.1e  steps max %d mean %.1f  t_local %.2fs%s"
              % (l["outer"], l["residual"],
                 ("err_iface %.2e  " % l["err_iface"]) if "err_iface" in l else "",
                 l["local_tol"], l["local_steps_max"], l["local_steps_mean"], l["t_local"],
                 ("  t_P %.2fs" % l["t_precond"]) if "t_precond" in l else ""))

    # ------------------------------------------------------------ diagnostics
    def volume_errors(self):
        """max |phi_i - phi*| over the leaf nodes of each slab."""
        return np.array([np.max(np.abs(s.phi - self.mms.phistar(s.W))) for s in self.slabs])

    def check_jacobian(self, Stot, seed=0):
        """|| (I - K) v - Stot v || / ||Stot v|| with K applied by local LINEAR
        solves (zero wall data, zero body load): checks P's operator against
        the local solves used by E (expect ~1e-12)."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.OMS.Ntot)
        Kv = np.empty_like(v)
        for i, s in enumerate(self.slabs):
            ub = np.zeros(len(s.ops.Ib))
            if s.L >= 0:
                ub[s.info.Il] = v[self.gtd[s.L]]
            if s.R >= 0:
                ub[s.info.Ir] = v[self.gtd[s.R]]
            xI = -np.asarray(s.hnd.solver_ii @ (s.ops.Aib @ ub)).reshape(-1)
            Kv[self.gtd[i]] = xI[s.info.Ic]
        Sv = Stot @ v
        return np.linalg.norm(v - Kv - Sv) / np.linalg.norm(Sv)
