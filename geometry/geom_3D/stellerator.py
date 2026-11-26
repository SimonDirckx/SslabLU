"""
Stellarator-type torus (toy W7-X-like boundary), parametrized over the same
periodic straight cylinder as full_twisted_torus.

Reference (OMS) coordinates  p = (t, x, y):
    t in [0, L), periodic  axial (slabs are stacked along p[:,0]),  L = 2 pi R00
    (x, y) in disk(Rc)     cross-section (5-patch HPS cylinder: core + 4 arches)

Forward map z = Phi(p)  (VMEC stellarator-symmetric Fourier form, "The Numerics
of VMEC++" eqs. 3.36-3.37, interior filled with the rho^m regularity factor):
    phi  = 2 pi t / L           toroidal (cylindrical) angle
    zeta = nfp phi              angle per field period
    w    = (x + i y) / Rc = rho e^{i theta}
    R    = sum_mn R_mn Re[w^m e^{-i n zeta}] = sum R_mn rho^m cos(m theta - n zeta)
    Z    = sum_mn Z_mn Im[w^m e^{-i n zeta}] = sum Z_mn rho^m sin(m theta - n zeta)
    z    = (R cos phi, R sin phi, Z)
The wall x^2 + y^2 = Rc^2 (rho = 1) is the boundary surface given by the
coefficients; rho^m = |w|^m makes every term a polynomial in (x, y), so the map
is smooth through the axis, and R, Z are harmonic in (x, y).  Phi is L-periodic
in t for ANY coefficients (no integer-twist restriction).  psi = rho^2 =
(x^2 + y^2)/Rc^2 plays the role of VMEC's flux label s.

PDO transform (transform_helmholtz_pdo is overridden).  hpsmultidomain wants
derivatives of the inverse map y = Phi^{-1}(q) at q = Phi(p); there is no closed
form here, but with J = dz/dp and G = J^{-1} (rows = grad_q y_k), all of them
follow from the FORWARD map at p:
    grad_q y_k          = G[k, :]
    M = G G^T           (inverse metric)            c_kk = M_kk,  c_kl = 2 M_kl
    Lap_q y_k           = - sum_a G[k, a] sum_bc (d^2 z_a / dp_b dp_c) M_bc
    c_k = -Lap_q y_k    (same convention as ParametrizedGeometry3D)
so the coefficients are exact and need no inversion.  The inverse map itself
(y1, y2, y3, contains) is a Newton solve at fixed toroidal angle ("Numerics of
VMEC++" Sec. 9.6): phi = atan2(q2, q1) gives t exactly, then (x, y) solves
(R, Z)(t, x, y) = (rho_q, q3).

Boundary data: by default the VMEC input DEFAULT_INPUT (the W7-X configuration
in test/direct-solve/Poisson-Boltzmann/input.w7x; override with the environment
variable STELLERATOR_INPUT or configure(path)): NFP, RBC(n,m), ZBS(n,m) and the
magnetic axis RAXIS_CC, ZAXIS_CS.  The axis is the image of rho = 0 (the m = 0
terms are blended axis -> boundary with rho^2), so the reference centre line is
VMEC's magnetic axis.  configure('toy') selects a small W7-X-like toy boundary
(aspect ~5).  MHD inputs (PHIEDGE, AM, ...) are not used: only the domain.

All functions accept numpy, torch or jax arrays of shape (..., 3) and return the
same type, except the Newton inverse (numpy internally, converted back).
"""
import numpy as np

from solver.hpsmultidomain.hpsmultidomain.geom import ParametrizedGeometry3D as ParametrizedGeometry3Dalt
from solver.hpsmultidomain.hpsmultidomain.pdo import PDO_3d


def _xp(p):
    """Array namespace of p (numpy, torch or jax.numpy)."""
    mod = type(p).__module__
    if mod.startswith("torch"):
        import torch
        return torch
    if mod.startswith("jax"):
        import jax.numpy as jnp
        return jnp
    return np


def _cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


_DER1 = ("t", "x", "y")
_PAIR = {("t", "t"): "tt", ("t", "x"): "tx", ("x", "t"): "tx", ("t", "y"): "ty", ("y", "t"): "ty",
         ("x", "x"): "xx", ("x", "y"): "xy", ("y", "x"): "xy", ("y", "y"): "yy"}
_KEYS = ("", "t", "x", "y", "tt", "tx", "ty", "xx", "xy", "yy")


class FourierTorus:
    """Fourier torus R = sum R_mn rho^m cos(m theta - n zeta), Z = sum Z_mn rho^m
    sin(m theta - n zeta) over the reference cylinder [0, L) x disk(Rc).
    modes: list of (m, n, R_mn, Z_mn), m >= 0 (m = 0 terms: the axis; Z_0n
    enters as -Z_0n sin(n zeta)).  (0, 0) gives R00 and L = 2 pi R00.

    axis (optional): list of (n, Raxis_n, Zaxis_n), the magnetic axis
    R = sum Raxis_n cos(n zeta), Z = -sum Zaxis_n sin(n zeta) (same sign
    convention as the m = 0 boundary terms).  The m = 0 terms are then blended
    radially, R_0n(rho) = Raxis_n + rho^2 (R_0n - Raxis_n) (likewise Z): rho = 0
    is the axis, rho = 1 the boundary, and rho^2 = (x^2 + y^2)/Rc^2 keeps the map
    polynomial in (x, y)."""

    def __init__(self, modes, nfp, Rc, s=0.5, axis=None):
        self.modes = [(int(m), int(n), float(r), float(z)) for m, n, r, z in modes]
        if any(m < 0 for m, *_ in self.modes):
            raise ValueError("poloidal mode numbers must be >= 0")
        self.nfp, self.Rc, self.s = int(nfp), float(Rc), float(s)
        self.R00 = sum(r for m, n, r, z in self.modes if m == 0 and n == 0)
        if self.R00 <= 0:
            raise ValueError("need a (0, 0) mode with R00 > 0")
        self.L = 2 * np.pi * self.R00
        self.mmax = max(m for m, *_ in self.modes)
        self.bnds = [[0., -self.Rc, -self.Rc], [self.L, self.Rc, self.Rc]]
        self.axis_terms = []                       # (n, Raxis, Rbdry, Zaxis, Zbdry)
        if axis:
            bd = {}
            for m, n, r, z in self.modes:
                if m == 0:
                    rb, zb = bd.get(n, (0., 0.))
                    bd[n] = (rb + r, zb + z)
            ax = {int(n): (float(r), float(z)) for n, r, z in axis}
            for n in sorted(set(bd) | set(ax)):
                ra, za = ax.get(n, (0., 0.))
                rb, zb = bd.get(n, (0., 0.))
                self.axis_terms.append((n, ra, rb, za, zb))
            self.modes = [md for md in self.modes if md[0] != 0]

    # ------------------------------------------------------------ R, Z and derivatives
    def _series(self, p, modes, order=2, mmax=None):
        """sum over modes (m, n, R_mn, Z_mn) of R_mn Re / Z_mn Im [w^m e^{-i n zeta}]
        and its (t, x, y) derivatives (dict keys as in fields)."""
        xp = _xp(p)
        t, x, y = p[..., 0], p[..., 1], p[..., 2]
        zero = 0. * t
        u, v = x / self.Rc, y / self.Rc
        P, Q = [zero + 1.], [zero]                                  # w^m = P_m + i Q_m
        for _ in range(self.mmax if mmax is None else mmax):
            P.append(P[-1] * u - Q[-1] * v)
            Q.append(P[-2] * v + Q[-1] * u)
        k = self.nfp * 2 * np.pi / self.L                           # d zeta / dt
        zeta = k * t
        keys = _KEYS[:1] if order == 0 else (_KEYS[:4] if order == 1 else _KEYS)
        R = {key: zero for key in keys}
        Z = {key: zero for key in keys}
        Rc = self.Rc
        for m, n, rmn, zmn in modes:
            cn, sn = xp.cos(n * zeta), xp.sin(n * zeta)
            E0 = (cn, -sn)                                          # e^{-i n zeta}
            nk = n * k
            W0 = (P[m], Q[m])
            terms = {"": _cmul(W0, E0)}
            if order >= 1:
                Et = (-nk * sn, -nk * cn)                           # -i n k e^{-i n zeta}
                if m >= 1:
                    Wx = (m * P[m - 1] / Rc, m * Q[m - 1] / Rc)
                    Wy = (-Wx[1], Wx[0])
                else:
                    Wx = Wy = (zero, zero)
                terms["t"] = _cmul(W0, Et)
                terms["x"] = _cmul(Wx, E0)
                terms["y"] = _cmul(Wy, E0)
                if order >= 2:
                    Ett = (-nk * nk * E0[0], -nk * nk * E0[1])
                    if m >= 2:
                        Wxx = (m * (m - 1) * P[m - 2] / Rc ** 2, m * (m - 1) * Q[m - 2] / Rc ** 2)
                    else:
                        Wxx = (zero, zero)
                    Wxy = (-Wxx[1], Wxx[0])
                    Wyy = (-Wxx[0], -Wxx[1])
                    terms.update(tt=_cmul(W0, Ett), tx=_cmul(Wx, Et), ty=_cmul(Wy, Et),
                                 xx=_cmul(Wxx, E0), xy=_cmul(Wxy, E0), yy=_cmul(Wyy, E0))
            for key in keys:
                re, im = terms[key]
                if rmn:
                    R[key] = R[key] + rmn * re
                if zmn:
                    Z[key] = Z[key] + zmn * im
        return R, Z

    def fields(self, p, order=2):
        """R, Z and their derivatives w.r.t. the reference coords (t, x, y):
        dicts with keys '', 't', 'x', 'y' (order >= 1), 'tt', 'tx', 'ty', 'xx',
        'xy', 'yy' (order 2)."""
        R, Z = self._series(p, self.modes, order)
        xp = _xp(p)
        t, x, y = p[..., 0], p[..., 1], p[..., 2]
        zero = 0. * t
        u, v = x / self.Rc, y / self.Rc
        k = self.nfp * 2 * np.pi / self.L                           # d zeta / dt
        zeta = k * t
        keys = _KEYS[:1] if order == 0 else (_KEYS[:4] if order == 1 else _KEYS)
        Rc = self.Rc
        if self.axis_terms:                        # m = 0, blended axis -> boundary
            g = u * u + v * v                      # rho^2
            gd = {"x": 2 * u / Rc, "y": 2 * v / Rc, "xx": zero + 2 / Rc ** 2,
                  "xy": zero, "yy": zero + 2 / Rc ** 2}
            for n, ra, rb, za, zb in self.axis_terms:
                cn, sn = xp.cos(n * zeta), xp.sin(n * zeta)
                nk = n * k
                ph = {"": (cn, -sn), "t": (-nk * sn, -nk * cn), "tt": (-nk * nk * cn, nk * nk * sn)}
                for comp, (A, B, out, part) in (("R", (ra, rb, R, 0)), ("Z", (za, zb, Z, 1))):
                    if A == 0 and B == 0:
                        continue
                    f, df = A + (B - A) * g, B - A
                    for key in keys:
                        tkey = "".join(ch for ch in key if ch == "t")
                        skey = "".join(ch for ch in key if ch != "t")
                        e = ph[tkey][part]
                        out[key] = out[key] + (f * e if skey == "" else df * gd[skey] * e)
        return R, Z

    # ------------------------------------------------------------ Cartesian map
    def forward(self, p):
        xp = _xp(p)
        R, Z = self.fields(p, order=0)
        phi = 2 * np.pi * p[..., 0] / self.L
        return R[""] * xp.cos(phi), R[""] * xp.sin(phi), Z[""]

    def jet(self, p):
        """z and its first/second derivatives: J[a][b] = dz_a/dp_b, H[a][b][c]
        = d^2 z_a/dp_b dp_c (p = (t, x, y))."""
        xp = _xp(p)
        R, Z = self.fields(p, order=2)
        g = 2 * np.pi / self.L                                      # d phi / dt
        phi = g * p[..., 0]
        c, s = xp.cos(phi), xp.sin(phi)
        d1, d2 = {}, {}                                             # z1, z2 derivatives
        d1[""], d2[""] = R[""] * c, R[""] * s
        d1["t"], d2["t"] = R["t"] * c - g * R[""] * s, R["t"] * s + g * R[""] * c
        for a in ("x", "y"):
            d1[a], d2[a] = R[a] * c, R[a] * s
        A = R["tt"] - g * g * R[""]
        d1["tt"], d2["tt"] = A * c - 2 * g * R["t"] * s, A * s + 2 * g * R["t"] * c
        for a in ("x", "y"):
            d1["t" + a] = R["t" + a] * c - g * R[a] * s
            d2["t" + a] = R["t" + a] * s + g * R[a] * c
        for key in ("xx", "xy", "yy"):
            d1[key], d2[key] = R[key] * c, R[key] * s
        comp = (d1, d2, Z)
        J = [[comp[a][b] for b in _DER1] for a in range(3)]
        H = [[[comp[a][_PAIR[(b, cc)]] for cc in _DER1] for b in _DER1] for a in range(3)]
        return (d1[""], d2[""], Z[""]), J, H

    # ------------------------------------------------------------ PDO coefficients
    def pdo_coeffs(self, p):
        """Coefficients of -Lap in reference coordinates at reference points p
        (hpsmultidomain PDO_3d convention), from the forward map only."""
        _, J, H = self.jet(p)
        C = [[J[(a + 1) % 3][(b + 1) % 3] * J[(a + 2) % 3][(b + 2) % 3]
              - J[(a + 1) % 3][(b + 2) % 3] * J[(a + 2) % 3][(b + 1) % 3]
              for b in range(3)] for a in range(3)]
        det = J[0][0] * C[0][0] + J[0][1] * C[0][1] + J[0][2] * C[0][2]
        G = [[C[i][k] / det for i in range(3)] for k in range(3)]   # G[k][i] = dy_k/dq_i
        M = [[sum(G[k][i] * G[l][i] for i in range(3)) for l in range(3)] for k in range(3)]
        lt = [sum(H[a][b][c] * M[b][c] for b in range(3) for c in range(3)) for a in range(3)]
        ck = [sum(G[k][a] * lt[a] for a in range(3)) for k in range(3)]   # = -Lap y_k
        return dict(c11=M[0][0], c22=M[1][1], c33=M[2][2],
                    c12=2 * M[0][1], c13=2 * M[0][2], c23=2 * M[1][2],
                    c1=ck[0], c2=ck[1], c3=ck[2], det=det)

    # ------------------------------------------------------------ Newton inverse
    def inverse(self, q, tol=1e-13, maxit=60, rmax=1.05):
        """Reference points p = Phi^{-1}(q) for physical points q (numpy (..., 3)).
        Newton at fixed toroidal angle; iterates are kept in rho <= rmax (outside
        the disk the map can fold back onto the domain, e.g. the high-degree
        VmecTorus, giving spurious roots).  Points that still end at rho > 1 or
        unconverged are restarted from the nearest point of a (rho, theta) table."""
        q = np.asarray(q, dtype=np.float64)
        shape = q.shape
        q = q.reshape(-1, 3)
        phi = np.mod(np.arctan2(q[:, 1], q[:, 0]), 2 * np.pi)
        t = self.L * phi / (2 * np.pi)
        Rq, Zq = np.hypot(q[:, 0], q[:, 1]), q[:, 2]
        p = np.c_[t, 0 * t, 0 * t]
        R, Z = self.fields(p, order=1)                              # axis + linear part
        det = R["x"] * Z["y"] - R["y"] * Z["x"]
        dR, dZ = Rq - R[""], Zq - Z[""]
        p[:, 1] = (Z["y"] * dR - R["y"] * dZ) / det
        p[:, 2] = (-Z["x"] * dR + R["x"] * dZ) / det
        scale = max(1.0, np.max(np.abs(q)))
        p, res = self._newton(p, Rq, Zq, tol * scale, maxit, rmax)
        bad = (np.hypot(p[:, 1], p[:, 2]) > self.Rc * (1 + 1e-9)) | (res > tol * scale)
        if bad.any():                                              # restart from a lookup table
            ib = np.flatnonzero(bad)
            rr, tt = np.meshgrid(np.linspace(0.05, 1, 16), np.linspace(0, 2 * np.pi, 48, endpoint=False),
                                 indexing="ij")
            gx, gy = self.Rc * (rr * np.cos(tt)).ravel(), self.Rc * (rr * np.sin(tt)).ravel()
            G = len(gx)
            for c0 in range(0, len(ib), max(1, 20000 // G)):
                ic = ib[c0:c0 + max(1, 20000 // G)]
                P = np.c_[np.repeat(t[ic], G), np.tile(gx, len(ic)), np.tile(gy, len(ic))]
                Rg, Zg = self.fields(P, order=0)
                d = (Rg[""].reshape(len(ic), G) - Rq[ic, None]) ** 2 + (Zg[""].reshape(len(ic), G) - Zq[ic, None]) ** 2
                j = np.argmin(d, axis=1)
                p[ic, 1], p[ic, 2] = gx[j], gy[j]
            p[ib], _ = self._newton(p[ib], Rq[ib], Zq[ib], tol * scale, maxit, rmax)
        return p.reshape(shape)

    def _newton(self, p, Rq, Zq, tol, maxit, rmax):
        def clamp(pp):
            r = np.hypot(pp[:, 1], pp[:, 2])
            f = np.minimum(1.0, rmax * self.Rc / np.maximum(r, 1e-300))
            pp[:, 1] *= f
            pp[:, 2] *= f
            return pp
        p = clamp(p.copy())
        res = np.full(len(p), np.inf)
        for _ in range(maxit):
            R, Z = self.fields(p, order=1)
            rR, rZ = R[""] - Rq, Z[""] - Zq
            res = np.hypot(rR, rZ)
            if np.max(res, initial=0.) < tol:
                break
            det = R["x"] * Z["y"] - R["y"] * Z["x"]
            sx = -(Z["y"] * rR - R["y"] * rZ) / det
            sy = -(-Z["x"] * rR + R["x"] * rZ) / det
            beta = np.ones_like(res)
            for _h in range(12):                                    # step halving
                trial = p.copy()
                trial[:, 1] += beta * sx
                trial[:, 2] += beta * sy
                trial = clamp(trial)
                Rt, Zt = self.fields(trial, order=0)
                worse = np.hypot(Rt[""] - Rq, Zt[""] - Zq) > res
                if not worse.any():
                    break
                beta = np.where(worse, beta / 2, beta)
            p = trial
        else:
            R, Z = self.fields(p, order=0)
            res = np.hypot(R[""] - Rq, Z[""] - Zq)
        return p, res


####################################
#   boundary data: VMEC input (default: W7-X) or the toy model
####################################

import os
import re

DEFAULT_INPUT = os.environ.get(
    "STELLERATOR_INPUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "test", "direct-solve", "Poisson-Boltzmann", "input.w7x"))


def read_vmec_input(path):
    """Boundary and axis of a VMEC fixed-boundary input file (&INDATA).
    Returns dict(nfp, modes=[(m, n, RBC, ZBS)], axis=[(n, RAXIS_CC, ZAXIS_CS)]).
    VMEC writes RBC(n, m): toroidal index first.  Stellarator symmetry only
    (RBS/ZBC/RAXIS_CS/ZAXIS_CC must be absent or zero)."""
    txt = open(path).read()
    txt = re.sub(r"!.*", "", txt)                                   # Fortran comments
    num = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?)"
    tofl = lambda v: float(v.replace("d", "e").replace("D", "e"))
    m_nfp = re.search(r"\bNFP\s*=\s*(\d+)", txt, re.I)
    if not m_nfp:
        raise ValueError("%s: no NFP" % path)
    coef = {}
    for name in ("RBC", "ZBS", "RBS", "ZBC"):
        for n, m, v in re.findall(r"\b%s\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*=\s*%s" % (name, num), txt, re.I):
            coef.setdefault(name, {})[(int(m), int(n))] = tofl(v)
    for name in ("RBS", "ZBC"):
        if any(abs(v) > 0 for v in coef.get(name, {}).values()):
            raise ValueError("%s: non-stellarator-symmetric %s terms are not supported" % (path, name))

    def array(name):
        mt = re.search(r"\b%s\s*=\s*((?:\s*%s)+)" % (name, num), txt, re.I)
        return [tofl(v) for v in re.findall(num, mt.group(1))] if mt else []

    for name in ("RAXIS_CS", "ZAXIS_CC"):
        if any(abs(v) > 0 for v in array(name)[1:]):
            raise ValueError("%s: non-stellarator-symmetric %s is not supported" % (path, name))
    rbc, zbs = coef.get("RBC", {}), coef.get("ZBS", {})
    if (0, 0) not in rbc:
        raise ValueError("%s: no RBC(0,0)" % path)
    keys = sorted(set(rbc) | set(zbs))
    modes = [(m, n, rbc.get((m, n), 0.), zbs.get((m, n), 0.)) for m, n in keys]
    if any(m == 0 and n < 0 for m, n in keys):
        raise ValueError("%s: m = 0 terms with n < 0 (redundant) are not supported" % path)
    rax, zax = array("RAXIS_CC") or array("RAXIS"), array("ZAXIS_CS") or array("ZAXIS")
    nax = max(len(rax), len(zax))
    axis = [(n, rax[n] if n < len(rax) else 0., zax[n] if n < len(zax) else 0.) for n in range(nax)]
    return dict(nfp=int(m_nfp.group(1)), modes=modes, axis=axis)


def read_vmec_wout(path):
    """Boundary (last flux surface) and axis of a VMEC output file wout_*.nc.
    Same dict as read_vmec_input.  wout stores rmnc/zmns(radius, mn) with the
    physical toroidal number xn = n nfp, cos(xm theta - xn phi); here n = xn / nfp.
    The axis raxis_cc / zaxis_cs uses the same sign convention as the m = 0
    terms (checked: they equal rmnc / zmns of the m = 0 modes at js = 0)."""
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as f:
        v = f.variables
        if "lasym__logical__" in v and int(v["lasym__logical__"].data):
            raise ValueError("%s: non-stellarator-symmetric (lasym) equilibria are not supported" % path)
        nfp = int(v["nfp"].data)
        xm = np.asarray(v["xm"].data).astype(int)
        xn = np.asarray(v["xn"].data)
        rb, zb = np.asarray(v["rmnc"].data)[-1], np.asarray(v["zmns"].data)[-1]
        rax = np.asarray(v["raxis_cc"].data).ravel()
        zax = np.asarray(v["zaxis_cs"].data).ravel()
    n = np.rint(xn / nfp).astype(int)
    if np.any(np.abs(n * nfp - xn) > 1e-8):
        raise ValueError("%s: xn not a multiple of nfp" % path)
    modes = [(int(m), int(k), float(r), float(z)) for m, k, r, z in zip(xm, n, rb, zb)
             if (m, k) == (0, 0) or abs(r) > 0 or abs(z) > 0]
    axis = [(k, float(rax[k]), float(zax[k])) for k in range(len(rax))]
    return dict(nfp=nfp, modes=modes, axis=axis)


class VmecTorus(FourierTorus):
    """Map through VMEC's own flux coordinates (interior surfaces from wout):
        R = sum_mn R_mn(rho) cos(m theta - n zeta),  Z = sum_mn Z_mn(rho) sin(...)
        R_mn(rho) = rho^m sum_{k=0}^{deg} a_mnk rho^(2k)       (likewise Z_mn)
    i.e. R = sum_k g^k F_k with g = rho^2 = (x^2 + y^2)/Rc^2 and F_k an ordinary
    Fourier series (FourierTorus._series): a polynomial in (x, y) at each zeta,
    smooth through the axis, with closed-form derivatives.  rho^2 = s (VMEC's
    flux label), so rho = const are VMEC's flux surfaces; rho = 0 is the axis
    (the k = 0, m = 0 terms), rho = 1 the last closed surface.

    layers: list over k of modes (m, n, a_Rmnk, a_Zmnk).  Use fit_wout / build."""

    def __init__(self, layers, nfp, Rc, s=0.5):
        bd = {}
        for lay in layers:                                     # boundary = sum_k a_k
            for m, n, r, z in lay:
                rb, zb = bd.get((m, n), (0., 0.))
                bd[(m, n)] = (rb + r, zb + z)
        super().__init__([(m, n, r, z) for (m, n), (r, z) in sorted(bd.items())], nfp, Rc, s)
        self.layers = [[(int(m), int(n), float(r), float(z)) for m, n, r, z in lay if r or z]
                       for lay in layers]
        self.deg = len(layers) - 1
        self.boundary_modes = self.modes
        self.axis_terms = [(n, sum(r for m, nn, r, z in layers[0] if m == 0 and nn == n),
                            r, sum(zz for m, nn, rr, zz in layers[0] if m == 0 and nn == n), z)
                           for m, n, r, z in self.modes if m == 0]   # (info only)
        self.fit_info = None
        # dense coefficients for the vectorized evaluation: A[j, k, m] for n = ns[j]
        self._ns = np.array(sorted({n for lay in self.layers for m, n, r, z in lay}), dtype=float)
        K, M = self.deg + 1, self.mmax + 1
        jn = {int(n): j for j, n in enumerate(self._ns)}
        AR, AZ = np.zeros((len(self._ns), K, M)), np.zeros((len(self._ns), K, M))
        for k, lay in enumerate(self.layers):
            for m, n, r, z in lay:
                AR[jn[n], k, m] += r
                AZ[jn[n], k, m] += z
        self._AR, self._AZ = AR.reshape(len(self._ns), K * M), AZ.reshape(len(self._ns), K * M)

    chunk = 8192                                       # points per block in fields

    def fields(self, p, order=2):
        """Vectorized: per point block, R = sum_k g^k sum_m [Re W_m (E A_k)_m]
        with E = e^{-i n zeta} (matrix products over n), W_m = w^m (einsum over
        m), and the product rule in g^k for the x, y derivatives.  Same result
        as _fields_loop (mode-by-mode reference)."""
        xp = _xp(p)
        keys = _KEYS[:1] if order == 0 else (_KEYS[:4] if order == 1 else _KEYS)
        shape = p.shape[:-1]
        q = p.reshape(-1, 3)
        n = q.shape[0]
        if n <= self.chunk:
            R, Z = self._fields_block(xp, q, keys, order)
        else:
            parts = [self._fields_block(xp, q[i:i + self.chunk], keys, order) for i in range(0, n, self.chunk)]
            cat = (lambda L: xp.cat(L)) if xp.__name__ == "torch" else xp.concatenate
            R = {key: cat([pr[0][key] for pr in parts]) for key in keys}
            Z = {key: cat([pr[1][key] for pr in parts]) for key in keys}
        return ({key: R[key].reshape(shape) for key in keys}, {key: Z[key].reshape(shape) for key in keys})

    def _fields_block(self, xp, q, keys, order):
        if xp.__name__ == "torch":
            import torch
            const = lambda a: torch.as_tensor(a, dtype=q.dtype, device=q.device)
        else:
            const = lambda a: xp.asarray(a, dtype=q.dtype)
        K, M, Rc = self.deg + 1, self.mmax + 1, self.Rc
        npt = q.shape[0]
        t, x, y = q[:, 0], q[:, 1], q[:, 2]
        zero = 0. * t
        u, v = x / Rc, y / Rc
        g = u * u + v * v
        # W_m = w^m = P_m + i Q_m and its x/y-derivative variants (npt, M)
        P, Q = [zero + 1.], [zero]
        for _ in range(M - 1):
            P.append(P[-1] * u - Q[-1] * v)
            Q.append(P[-2] * v + Q[-1] * u)
        P, Q = xp.stack(P, 1), xp.stack(Q, 1)
        mv = const(np.arange(M, dtype=float))
        Z0 = 0. * P[:, :1]
        W = {"": (P, Q)}
        if order >= 1:
            sh = lambda A, j: xp.concatenate([Z0] * j + [A[:, :M - j]], 1) if xp.__name__ != "torch" \
                else xp.cat([Z0] * j + [A[:, :M - j]], 1)
            Wx = (mv * sh(P, 1) / Rc, mv * sh(Q, 1) / Rc)
            W["x"], W["y"] = Wx, (-Wx[1], Wx[0])
            if order >= 2:
                c2 = mv * (mv - 1) / Rc ** 2
                Wxx = (c2 * sh(P, 2), c2 * sh(Q, 2))
                W["xx"], W["xy"], W["yy"] = Wxx, (-Wxx[1], Wxx[0]), (-Wxx[0], -Wxx[1])
        # E_n = e^{-i n zeta} and t-derivative variants (npt, N), then (E A) -> (npt, K, M)
        kk = self.nfp * 2 * np.pi / self.L
        nk = const(self._ns * kk)
        ang = (kk * t)[:, None] * const(self._ns)[None, :]
        c, s = xp.cos(ang), xp.sin(ang)
        Ev = {"": (c, -s)}
        if order >= 1:
            Ev["t"] = (-nk * s, -nk * c)
        if order >= 2:
            Ev["tt"] = (-nk * nk * c, nk * nk * s)
        AR, AZ = const(self._AR), const(self._AZ)
        EA = {}
        for tk, (er, ei) in Ev.items():
            EA[tk] = [(e @ A).reshape(npt, K, M) for e in (er, ei) for A in (AR, AZ)]   # erR, erZ, eiR, eiZ
        # F_k for every key: Re / Im of sum_m W_m (E A_k)_m
        FR, FZ = {}, {}
        for key in keys:
            tk = "".join(ch for ch in key if ch == "t")
            sk = key.replace("t", "")
            Wr, Wi = W[sk]
            erR, erZ, eiR, eiZ = EA[tk]
            FR[key] = xp.einsum("pm,pkm->pk", Wr, erR) - xp.einsum("pm,pkm->pk", Wi, eiR)
            FZ[key] = xp.einsum("pm,pkm->pk", Wr, eiZ) + xp.einsum("pm,pkm->pk", Wi, erZ)
        # h_k = g^k and its x/y derivatives (npt, K)
        G = [zero + 1.]
        for _ in range(K - 1):
            G.append(G[-1] * g)
        G = xp.stack(G, 1)
        kv = const(np.arange(K, dtype=float))
        h = {"": G}
        if order >= 1:
            Zk = 0. * G[:, :1]
            cat = xp.cat if xp.__name__ == "torch" else xp.concatenate
            G1 = cat([Zk, G[:, :K - 1]], 1)
            G2 = cat([Zk, Zk, G[:, :K - 2]], 1) if K >= 2 else 0. * G
            gd = {"x": (2 * u / Rc)[:, None], "y": (2 * v / Rc)[:, None]}
            for a_ in ("x", "y"):
                h[a_] = kv * G1 * gd[a_]
            if order >= 2:
                for a_, b_ in (("x", "x"), ("x", "y"), ("y", "y")):
                    gab = (2 / Rc ** 2) if a_ == b_ else 0.
                    h[a_ + b_] = kv * (kv - 1) * G2 * gd[a_] * gd[b_] + kv * G1 * gab
        R, Z = {}, {}
        for F, out in ((FR, R), (FZ, Z)):
            for key in keys:
                tk = "t" if "t" in key else ""
                sk = key.replace("t", "")
                val = F[key] * h[""]
                if len(sk) == 1:
                    val = val + F[tk] * h[sk]
                elif len(sk) == 2:
                    a_, b_ = sk
                    val = val + F[tk + a_ if tk else a_] * h[b_] + F[tk + b_ if tk else b_] * h[a_] + F[tk] * h[sk]
                out[key] = val.sum(1)
        return R, Z

    def _fields_loop(self, p, order=2):
        """Mode-by-mode reference implementation of fields (slow)."""
        keys = _KEYS[:1] if order == 0 else (_KEYS[:4] if order == 1 else _KEYS)
        x, y = p[..., 1], p[..., 2]
        zero = 0. * x
        Rc = self.Rc
        u, v = x / Rc, y / Rc
        g = u * u + v * v
        gd = {"x": 2 * u / Rc, "y": 2 * v / Rc, "xx": zero + 2 / Rc ** 2, "xy": zero, "yy": zero + 2 / Rc ** 2}
        gp = [zero + 1.]                                       # g^j
        for _ in range(self.deg):
            gp.append(gp[-1] * g)
        gpow = lambda j: gp[j] if j >= 0 else zero
        R = {key: zero for key in keys}
        Z = {key: zero for key in keys}
        for k, lay in enumerate(self.layers):
            if not lay:
                continue
            FR, FZ = self._series(p, lay, order)
            h = {"": gp[k]}                                    # derivatives of g^k
            if order >= 1:
                for a in ("x", "y"):
                    h[a] = k * gpow(k - 1) * gd[a]
            if order >= 2:
                for a, b in (("x", "x"), ("x", "y"), ("y", "y")):
                    h[a + b] = k * (k - 1) * gpow(k - 2) * gd[a] * gd[b] + k * gpow(k - 1) * gd[a + b]
            for F, out in ((FR, R), (FZ, Z)):
                for key in keys:                               # product rule F * g^k
                    tk = "t" if "t" in key else ""
                    sk = key.replace("t", "")
                    val = F[key] * h[""]
                    if len(sk) == 1:
                        val = val + F[tk] * h[sk]
                    elif len(sk) == 2:
                        a, b = sk
                        val = val + F[a] * h[b] + F[b] * h[a] + F[""] * h[sk]
                    out[key] = out[key] + val
        return R, Z


def fit_wout(path, deg=8, mmax=None, nmax=None, s=0.5, w_edge=1e4, w_axis=1e2):
    """VmecTorus from wout_*.nc: per (m, n) least-squares fit of rmnc / zmns over
    all ns surfaces, R_mn(rho) = rho^m P(rho^2), deg(P) = deg, rho = sqrt(s_j).
    The last surface (and the axis) are weighted by w_edge (w_axis) so the
    boundary is matched to ~ fit precision.  Rc as in build: sqrt(|R_10 Z_10|)
    of the boundary.  fit_info: max fit error per surface (R and Z together)."""
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as f:
        v = f.variables
        if "lasym__logical__" in v and int(v["lasym__logical__"].data):
            raise ValueError("%s: non-stellarator-symmetric (lasym) equilibria are not supported" % path)
        nfp = int(v["nfp"].data)
        xm = np.asarray(v["xm"].data).astype(int)
        xn = np.asarray(v["xn"].data)
        rmnc, zmns = np.asarray(v["rmnc"].data, float), np.asarray(v["zmns"].data, float)
    ns = rmnc.shape[0]
    rho = np.sqrt(np.linspace(0, 1, ns))
    wts = np.ones(ns)
    wts[-1], wts[0] = w_edge, w_axis
    n_all = np.rint(xn / nfp).astype(int)
    layers = [[] for _ in range(deg + 1)]
    err = np.zeros(ns)
    for i, (m, n) in enumerate(zip(xm, n_all)):
        if (mmax is not None and m > mmax) or (nmax is not None and abs(n) > nmax):
            continue
        A = np.stack([rho ** (m + 2 * k) for k in range(deg + 1)], 1)
        sc = np.linalg.norm(A, axis=0)
        sc[sc == 0] = 1
        Aw = A * wts[:, None] / sc
        aR = np.linalg.lstsq(Aw, wts * rmnc[:, i], rcond=None)[0] / sc
        aZ = np.linalg.lstsq(Aw, wts * zmns[:, i], rcond=None)[0] / sc
        err += np.abs(A @ aR - rmnc[:, i]) + np.abs(A @ aZ - zmns[:, i])
        for k in range(deg + 1):
            layers[k].append((int(m), int(n), aR[k], aZ[k]))
    r10 = sum(r for lay in layers for m, n, r, z in lay if (m, n) == (1, 0))
    z10 = sum(z for lay in layers for m, n, r, z in lay if (m, n) == (1, 0))
    tor = VmecTorus(layers, nfp, float(np.sqrt(abs(r10 * z10))), s)
    tor.fit_info = dict(rho=rho, err=err)
    return tor


def toy_modes():
    """Toy W7-X-like boundary (aspect ~5): nfp = 5, an ellipse rotating half a
    turn per field period, rotating triangularity, axis excursion."""
    abar, ell = 0.6, 0.2
    return dict(nfp=5, modes=[(0, 0, 3.0, 0.0), (0, 1, 0.15, 0.12), (1, 0, abar, abar),
                              (1, 1, ell, -ell), (2, 1, 0.05, 0.05)],
                axis=None, Rc=float(np.sqrt(abar ** 2 - ell ** 2)))


def build(source=None, mmax=None, nmax=None, s=0.5, fill="vmec", deg=10):
    """FourierTorus from a VMEC input path (default DEFAULT_INPUT) or 'toy'.
    mmax / nmax truncate the boundary spectrum.  Rc = sqrt(|R_10 Z_10|) (radius
    of the m = 1, n = 0 cross-section, so the reference disk has the physical
    size of the plasma).
    For a wout_*.nc file, fill = 'vmec' (default) maps through VMEC's interior
    flux surfaces (VmecTorus, fit degree deg in rho^2); fill = 'boundary' uses
    only the last surface + axis with the rho^m fill (FourierTorus)."""
    if str(source).endswith(".nc") and fill == "vmec":
        return fit_wout(source, deg=deg, mmax=mmax, nmax=nmax, s=s)
    if fill not in ("vmec", "boundary"):
        raise ValueError("fill must be 'vmec' or 'boundary'")
    if source == "toy":
        data = toy_modes()
    elif str(source).endswith(".nc"):
        data = read_vmec_wout(source)                # VMEC output (wout_*.nc)
    else:
        data = read_vmec_input(source or DEFAULT_INPUT)
    modes = [md for md in data["modes"]
             if (mmax is None or md[0] <= mmax) and (nmax is None or abs(md[1]) <= nmax)]
    Rc = data.get("Rc")
    if Rc is None:
        r10 = sum(r for m, n, r, z in modes if (m, n) == (1, 0))
        z10 = sum(z for m, n, r, z in modes if (m, n) == (1, 0))
        Rc = float(np.sqrt(abs(r10 * z10)))
    return FourierTorus(modes, data["nfp"], Rc, s, axis=data.get("axis"))


def configure(source=None, mmax=None, nmax=None, s=0.5, fill="vmec", deg=10):
    """Select the module geometry (rebinds _geom, nfp, modes, Rc, s, R0, L,
    bnds).  Call before param_geom / dSlabs; default is DEFAULT_INPUT (W7-X)."""
    global _geom, nfp, modes, Rc, R0, L, bnds
    _geom = build(source, mmax, nmax, s, fill=fill, deg=deg)
    globals()["s"] = _geom.s
    nfp, modes, Rc = _geom.nfp, _geom.modes, _geom.Rc
    R0, L, bnds = _geom.R00, _geom.L, _geom.bnds
    return _geom


s = 0.5                                    # core half-width / Rc (5-patch cross-section)
configure()


def twisted_torus_modes(R0=3.0, a=1.2, b=0.75, twist=2):
    """Coefficients reproducing full_twisted_torus (nfp = 1, Rc = sqrt(a b))."""
    return [(0, 0, R0, 0.0), (1, -twist, (a + b) / 2, (a + b) / 2),
            (1, twist, (a - b) / 2, (b - a) / 2)]


####################################
#      FORWARD MAP  z = Phi(p)
####################################

def z1(p, jax_avail=True, torch_avail=False):
    return _geom.forward(p)[0]


def z2(p, jax_avail=True, torch_avail=False):
    return _geom.forward(p)[1]


def z3(p, jax_avail=True, torch_avail=False):
    return _geom.forward(p)[2]


####################################
#   INVERSE MAP (Newton) y = Phi^{-1}(q)
####################################

def _inverse_any(q):
    if hasattr(q, "detach"):
        import torch
        out = _geom.inverse(q.detach().cpu().numpy())
        return torch.as_tensor(out, dtype=q.dtype, device=q.device)
    return _geom.inverse(np.asarray(q))


def y1(q, jax_avail=True, torch_avail=False):
    return _inverse_any(q)[..., 0]


def y2(q, jax_avail=True, torch_avail=False):
    return _inverse_any(q)[..., 1]


def y3(q, jax_avail=True, torch_avail=False):
    return _inverse_any(q)[..., 2]


def contains(q):
    """True for physical points q (n, 3) strictly inside the torus."""
    p = _geom.inverse(np.asarray(q, dtype=np.float64))
    return p[:, 1] ** 2 + p[:, 2] ** 2 < Rc ** 2


####################################
#   GEOMETRY OBJECT with the forward-map PDO transform
####################################

class StellaratorGeometry(ParametrizedGeometry3Dalt):
    """ParametrizedGeometry3D whose transform_helmholtz_pdo evaluates the
    coefficients from the forward map (FourierTorus.pdo_coeffs) instead of
    from derivatives of the inverse map (not available in closed form)."""

    def __init__(self, torus, box_geom, z, y):
        super().__init__(box_geom, z1=z[0], z2=z[1], z3=z[2], y1=y[0], y2=y[1], y3=y[2])
        self.torus = torus
        self._cache = {}

    def _coeffs(self, xx):
        """All coefficients at reference points xx, cached (hpsmultidomain
        evaluates every coefficient on the same points in turn)."""
        cx = self._cache.get("xx")
        same = cx is not None and cx.shape == xx.shape and type(cx) is type(xx)
        if same:
            same = bool((cx == xx).all())
        if not same:
            xc = xx.detach().clone() if hasattr(xx, "detach") else np.array(xx, copy=True)
            self._cache = {"xx": xc, "val": self.torus.pdo_coeffs(xx)}
        return self._cache["val"]

    def transform_helmholtz_pdo(self, bfield, kh):
        def coef(name):
            return lambda xx: self._coeffs(xx)[name]

        def c(xx):
            return bfield(self.parameter_map(xx), kh)

        return PDO_3d(c11=coef("c11"), c22=coef("c22"), c33=coef("c33"),
                      c12=coef("c12"), c13=coef("c13"), c23=coef("c23"),
                      c1=coef("c1"), c2=coef("c2"), c3=coef("c3"), c=c)


####################################
#   BOUNDARY (reference coordinates)
####################################

def gb_np(p):
    # periodic in t: no end caps, the tube wall is the only physical boundary
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    return p[:, 1] ** 2 + p[:, 2] ** 2 >= Rc ** 2 * (1 - 1e-8)


def gb(p, jax_avail=True, torch_avail=False):
    return gb_np(p)


def box_geom(jax_avail=True, torch_avail=False):
    if jax_avail:
        import jax.numpy as jnp
        return jnp.array(bnds)
    return np.array(bnds)


def bounds():
    return bnds


def param_geom(jax_avail=True, torch_avail=False, hpsalt=False):
    if not hpsalt:
        raise NotImplementedError("only the hpsmultidomain (hpsalt) geometry is provided")
    f = lambda g: (lambda p: g(p, jax_avail, torch_avail))
    return StellaratorGeometry(_geom, box_geom(jax_avail, torch_avail),
                               (f(z1), f(z2), f(z3)), (f(y1), f(y2), f(y3)))


####################################
#       slab functions
####################################

def dSlabs(N):
    """Periodic decomposition of [0, L) into N single slabs of width H = L/N
    (same layout as full_twisted_torus.dSlabs)."""
    if N < 3:
        raise ValueError("a periodic decomposition needs N >= 3 slabs, got %d" % N)
    H = (bnds[1][0] - bnds[0][0]) / N
    slabs = [[[bnds[0][0] + n * H, bnds[0][1], bnds[0][2]],
              [bnds[0][0] + (n + 2) * H, bnds[1][1], bnds[1][2]]] for n in range(N)]
    connectivity = [[(n - 1) % N, (n + 1) % N] for n in range(N)]
    return slabs, connectivity, H


####################################
#       validate
####################################

def _wrap(d, period):
    return d - period * np.floor(d / period + 0.5)


def _sample(torus, n, rng):
    r = torus.Rc * np.sqrt(rng.uniform(size=n)); th = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(0, torus.L, n)
    t[: n // 4] = rng.uniform(-1e-3, 1e-3, n // 4) % torus.L      # at the seam
    return np.c_[t, 0.98 * r * np.cos(th), 0.98 * r * np.sin(th)]


def check_param(n=200, seed=0, torus=None, ngrid=(400, 24, 64)):
    """(1) y(z(p)) = p (Newton), (2) L-periodicity, (3) finite-difference checks
    of J and H, (4) PDO coefficients vs. finite differences of the Newton
    inverse, (5) min/max det J over a reference grid (sign change = folding)."""
    torus = _geom if torus is None else torus
    rng = np.random.default_rng(seed)
    p = _sample(torus, n, rng)
    q = np.stack(torus.forward(p), -1)
    pi = torus.inverse(q)
    err_y = max(np.max(np.abs(_wrap(pi[:, 0] - p[:, 0], torus.L))), np.max(np.abs(pi[:, 1:] - p[:, 1:])))
    err_per = np.max(np.abs(np.stack(torus.forward(p + [torus.L, 0., 0.]), -1) - q))
    # J and H by central differences of the forward map
    _, J, H = torus.jet(p)
    h = 1e-5
    err_J = err_H = 0.
    for b in range(3):
        e = np.zeros(3); e[b] = h
        _, Jp, _ = torus.jet(p + e)
        _, Jm, _ = torus.jet(p - e)
        zp, zm = torus.forward(p + e), torus.forward(p - e)
        for a in range(3):
            err_J = max(err_J, np.max(np.abs((zp[a] - zm[a]) / (2 * h) - J[a][b])))
            for c in range(3):
                err_H = max(err_H, np.max(np.abs((Jp[a][c] - Jm[a][c]) / (2 * h) - H[a][b][c])))
    # PDO coefficients vs. derivatives of the Newton inverse (independent route)
    co = torus.pdo_coeffs(p)
    hq = 1e-4
    y0 = torus.inverse(q)
    grads = np.zeros((len(p), 3, 3)); lap = np.zeros((len(p), 3))
    for i in range(3):
        e = np.zeros(3); e[i] = hq
        yp, ym = torus.inverse(q + e), torus.inverse(q - e)
        dp, dm = yp - y0, ym - y0
        dp[:, 0], dm[:, 0] = _wrap(dp[:, 0], torus.L), _wrap(dm[:, 0], torus.L)
        grads[:, :, i] = (dp - dm) / (2 * hq)
        lap += (dp + dm) / hq ** 2
    Mfd = np.einsum("nki,nli->nkl", grads, grads)
    names = [("c11", 0, 0, 1), ("c22", 1, 1, 1), ("c33", 2, 2, 1), ("c12", 0, 1, 2), ("c13", 0, 2, 2), ("c23", 1, 2, 2)]
    err_M = max(np.max(np.abs(co[nm] - f * Mfd[:, k, l])) for nm, k, l, f in names)
    err_c = max(np.max(np.abs(co["c%d" % (k + 1)] + lap[:, k])) for k in range(3))
    # det J over a reference grid, wall included
    nt, nr, nth = ngrid
    T, Rr, Th = np.meshgrid(np.linspace(0, torus.L, nt, endpoint=False), np.linspace(0, 1, nr),
                            np.linspace(0, 2 * np.pi, nth, endpoint=False), indexing="ij")
    P = np.c_[T.ravel(), torus.Rc * (Rr * np.cos(Th)).ravel(), torus.Rc * (Rr * np.sin(Th)).ravel()]
    det = torus.pdo_coeffs(P)["det"]
    print("err y(z(p)) - p = %.2e,  periodicity = %.2e,  dz = %.2e,  d2z = %.2e"
          % (err_y, err_per, err_J, err_H))
    print("PDO coeffs vs Newton-inverse FD: metric %.2e, first-order %.2e (FD-limited, expect ~1e-6)"
          % (err_M, err_c))
    print("det J over the reference cylinder: min %.3e, max %.3e (%s)"
          % (det.min(), det.max(), "no folding" if det.min() * det.max() > 0 else "FOLDED"))
    return dict(y=err_y, per=err_per, J=err_J, H=err_H, M=err_M, c=err_c,
                det_min=float(det.min()), det_max=float(det.max()))


def check_twisted_torus(n=200, seed=1):
    """Regression: the Fourier torus with twisted_torus_modes() reproduces
    full_twisted_torus (map and PDO coefficients from its closed-form inverse)."""
    import torch
    import geometry.geom_3D.full_twisted_torus as ftt
    tor = FourierTorus(twisted_torus_modes(ftt.R0, ftt.a, ftt.b, int(ftt.twist)), 1, ftt.Rc)
    rng = np.random.default_rng(seed)
    p = _sample(tor, n, rng)
    err_z = np.max(np.abs(np.stack(tor.forward(p), -1) - np.stack(ftt._forward(p), -1)))
    ref = ftt.param_geom(jax_avail=False, torch_avail=True, hpsalt=True)
    pdo = ref.transform_helmholtz_pdo(lambda qq, kh: 0. * qq[..., 0], 0.)
    pt = torch.from_numpy(p)
    co = tor.pdo_coeffs(p)
    err = 0.
    for nm in ("c11", "c22", "c33", "c12", "c13", "c23", "c1", "c2", "c3"):
        f = getattr(pdo, nm)
        v = np.zeros(n) if f is None else np.asarray(f(pt).detach().cpu().numpy() if torch.is_tensor(f(pt)) else f(pt))
        err = max(err, np.max(np.abs(co[nm] - v)))
    print("twisted-torus regression: map %.2e, PDO coefficients %.2e (expect ~1e-13)" % (err_z, err))
    return err_z, err
