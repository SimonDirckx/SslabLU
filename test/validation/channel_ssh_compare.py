#!/usr/bin/env python3
# =============================================================================
# channel_ssh_compare.py
#
# Direct SSH comparison: SslabLU (channel_barotropic_timestep.py) vs
# Oceananigans (reentrant_channel_sslablu.jl), on the Oceananigans grid.
#
# SslabLU is spectrally accurate on its leaf boxes, so IT is the side that is
# moved: channel_barotropic_timestep.py evaluates its leaf interpolant at the
# Oceananigans cell centers of every n x n grid in SSLABLU_SSH_NS and writes
# channel_timestep_ssh.npz. The Oceananigans side is used as-is (2nd-order FV,
# cell-center eta), so no interpolation error is added to the difference.
#
# Fairness adjustments (see the conversation notes / header of each model):
#   * both fields are DE-MEANED before differencing; the means are reported
#     separately (Oceananigans conserves volume exactly, SslabLU's emulated
#     no-flux walls leak slightly -- its mass_resid diagnostic)
#   * norms are reported both over the full domain and over the INTERIOR band
#     outside SslabLU's wall sponge/taper (dist to wall >= SPONGE_W), where the
#     two wall treatments differ by construction; the band is hatched on maps
#   * H_Oceananigans - H_analytic at the u-faces is plotted: the face depth
#     min(H[i-1], H[i]) that the momentum/continuity terms use, plus
#     cell-center sampling of the narrow crest, is the prime suspect for
#     differences that line up with the ridge/gap
#   * several Oceananigans resolutions -> a convergence plot. If the difference
#     is Oceananigans' O(dx^2) FV error it falls ~4x per doubling; where it
#     plateaus is the genuine model difference (walls, lateral viscosity,
#     1st-order IMEX vs AB2, ...)
#
# Usage (from the repo root, after running both models with matching NSTEPS):
#   julia --project test/validation/reentrant_channel_sslablu.jl 400        # 80^2
#   julia --project test/validation/reentrant_channel_sslablu.jl 400 160
#   julia --project test/validation/reentrant_channel_sslablu.jl 400 320
#   python test/validation/channel_barotropic_timestep.py
#   python test/validation/channel_ssh_compare.py [ssh.npz] [a.jld2 b.jld2 ...]
# Defaults: channel_timestep_ssh.npz and run_channel_sslablu_spinup<NSTEPS>*/.
#
# Outputs:
#   channel_ssh_compare_n<N>.png        maps + zonal means + H difference
#   channel_ssh_compare_convergence.png difference vs Oceananigans resolution
#   channel_ssh_compare.csv             all metrics
# =============================================================================

import sys
import glob

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- ridge params: MUST match both model scripts ----------------------------
RIDGE_HR, RIDGE_KB = 0.8, 40.0
GAP_DEPTH, GAP_Y0, GAP_Y1, GAP_W = 1.0, 1.0 / 6.0, 1.0 / 2.0, 0.05


def depth_analytic(x, y, H0, xc_ridge):
    """SslabLU H(x,y) in metres, x and y nondimensional (x/L, y/L); crest at
    x = xc_ridge (0.5 centered, 0.5 + 1/32 with SSLABLU_RIDGE_MIDPANEL=1)."""
    bump = np.exp(RIDGE_KB * (np.cos(2.0 * np.pi * (x - xc_ridge)) - 1.0))
    gap = 1.0 - 0.5 * GAP_DEPTH * (np.tanh((y - GAP_Y0) / GAP_W) -
                                   np.tanh((y - GAP_Y1) / GAP_W))
    return H0 * (1.0 - RIDGE_HR * gap * bump)


def load(f, name):
    """JLD2/HDF5 dataset with the column-major axis reversal undone."""
    a = np.asarray(f[name][()])
    if a.ndim > 1:
        a = a.transpose()
    return np.squeeze(a)


def centered_uv(u, v, n):
    """Oceananigans C-grid (u on x-faces, periodic; v on y-faces, n+1 of
    them) -> cell centers, to match SslabLU's collocated samples."""
    u = u[:n, :n]
    uc = 0.5 * (u + np.roll(u, -1, axis=0))
    vc = 0.5 * (v[:n, :n] + v[:n, 1:n + 1])
    return uc, vc


def tilt(eta, yc):
    """Mean(north third) - mean(south third), as in both model scripts."""
    return eta[:, yc >= 2.0 / 3.0].mean() - eta[:, yc < 1.0 / 3.0].mean()


def rel_l2(d, ref, mask):
    den = np.linalg.norm(ref[mask])
    return np.linalg.norm(d[mask]) / den if den > 0 else np.nan


# ---- inputs -----------------------------------------------------------------
args = sys.argv[1:]
npz_path = args[0] if args and args[0].endswith(".npz") else "channel_timestep_ssh.npz"
jl_paths = [a for a in args if a.endswith(".jld2")]

S = np.load(npz_path)
nsteps = int(S["nsteps"])
if not jl_paths:
    jl_paths = sorted(glob.glob("run_channel_sslablu_spinup%d*/data_final.jld2" % nsteps))
if not jl_paths:
    sys.exit("no Oceananigans output for NSTEPS = %d; pass .jld2 paths" % nsteps)

L, H0, dt = float(S["L"]), float(S["H0"]), float(S["dt"])
sponge_w = float(S["sponge_w"]) if bool(S["wall_noflux"]) else 0.0
ridge_xc = float(S["ridge_xc"]) if "ridge_xc" in S else 0.5   # pre-shift npz files
print("  ridge crest x/L = %.5f" % ridge_xc)
print("SslabLU  : %s  (NSTEPS = %d, dt = %.0f s, p = %d, N = %d, samples at n = %s)"
      % (npz_path, nsteps, dt, int(S["p"]), int(S["N"]), list(S["ns"])))
print("  forcing: tau0 = %.3f, rdrag = %.1e, steric = %.2f m, gamma_s = %.1e"
      % (float(S["tau0"]), float(S["rdrag"]), float(S["steric_amp"]), float(S["gamma_s"])))
print("  interior band (outside wall sponge): %.3f <= y/L <= %.3f"
      % (sponge_w, 1.0 - sponge_w))

rows = []          # CSV
conv = {}          # (tag) -> list of (n, relL2_full, relL2_int)
for jl in jl_paths:
    with h5py.File(jl, 'r') as f:
        if "xc" not in f:
            print("skip %s: pre-comparison output (no xc/Hc/ssh_mid) -- rerun the Julia script" % jl)
            continue
        n = int(f["Nx"][()])
        O = {"xc": load(f, "xc") / L, "yc": load(f, "yc") / L, "Hc": load(f, "Hc"),
             "xf": load(f, "xf") / L, "Hu": load(f, "Hu"),
             "ridge_xc": float(f["ridge_xc"][()]) if "ridge_xc" in f else 0.5,
             "nsteps": int(f["nsteps"][()]), "dt": float(f["dt"][()]),
             "final": (load(f, "ssh"), load(f, "u"), load(f, "v"), float(f["t"][()])),
             "mid": (load(f, "ssh_mid"), load(f, "u_mid"), load(f, "v_mid"), float(f["t_mid"][()]))}
    if O["nsteps"] != nsteps or O["dt"] != dt:
        print("skip %s: NSTEPS/dt = %d/%.0f s, SslabLU has %d/%.0f s"
              % (jl, O["nsteps"], O["dt"], nsteps, dt))
        continue
    if abs(O["ridge_xc"] - ridge_xc) > 1e-12:
        print("skip %s: ridge crest x/L = %.5f, SslabLU has %.5f (SSLABLU_RIDGE_MIDPANEL mismatch)"
              % (jl, O["ridge_xc"], ridge_xc))
        continue
    if "eta_final_%d" % n not in S:
        print("skip %s: no SslabLU samples at n = %d (set SSLABLU_SSH_NS)" % (jl, n))
        continue

    xc, yc = O["xc"], O["yc"]
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    dist = np.minimum(Y, 1.0 - Y)
    full = np.ones_like(X, dtype=bool)
    inner = dist >= sponge_w
    # u-face depth (min of the two neighbouring columns) vs the analytic H at
    # the same face: the depth Oceananigans' dynamics actually sees. (The
    # cell-center column depth equals the analytic H there to round-off.)
    Xf, _ = np.meshgrid(O["xf"][:n], yc, indexing='ij')
    Hdiff = O["Hu"][:n, :n] - depth_analytic(Xf, Y, H0, ridge_xc)

    res = {}
    for tag in ("mid", "final"):
        etaO, uO, vO, tO = O[tag]
        etaO = etaO[:n, :n]
        uO, vO = centered_uv(uO, vO, n)
        etaS = S["eta_%s_%d" % (tag, n)]
        uS, vS = S["u_%s_%d" % (tag, n)], S["v_%s_%d" % (tag, n)]
        tS = float(S["t_" + tag])
        if abs(tS - tO) > 1e-6 * max(tS, 1.0):
            print("  WARNING n=%d %s: t_SslabLU = %.0f s, t_Oceananigans = %.0f s" % (n, tag, tS, tO))

        mS, mO = float(S["eta_mean_" + tag]), etaO.mean()
        etaSp, etaOp = etaS - mS, etaO - mO
        d = etaSp - etaOp
        r = {"etaS": etaSp, "etaO": etaOp, "d": d, "t": tS,
             "l2_full": rel_l2(d, etaSp, full), "l2_int": rel_l2(d, etaSp, inner),
             "max_full": np.abs(d).max(), "max_int": np.abs(d[inner]).max(),
             "tiltS": tilt(etaS, yc), "tiltO": tilt(etaO, yc), "meanS": mS, "meanO": mO,
             "u_int": rel_l2(uS - uO, uS, inner), "v_int": rel_l2(vS - vO, vS, inner)}
        res[tag] = r
        conv.setdefault(tag, []).append((n, r["l2_full"], r["l2_int"]))
        rows.append([n, tS / 3600.0, 0 if tag == "mid" else 1,
                     r["l2_full"], r["l2_int"], r["max_full"], r["max_int"],
                     r["tiltS"], r["tiltO"], mS, mO, r["u_int"], r["v_int"],
                     np.abs(Hdiff).max()])

    print("")
    print("=============SUMMARY  n = %d  (%s)=============" % (n, jl))
    print("max |H^u_Ocean - H_analytic| = %7.2f m   (min H_Ocean center/u-face = %.1f / %.1f m)"
          % (np.abs(Hdiff).max(), O["Hc"].min(), O["Hu"].min()))
    for tag in ("mid", "final"):
        r = res[tag]
        print("t = %6.1f h (%s)" % (r["t"] / 3600.0, tag))
        print("  mean eta   S / O          = %10.3E / %10.3E m" % (r["meanS"], r["meanO"]))
        print("  tilt N-S   S / O          = %10.4E / %10.4E m" % (r["tiltS"], r["tiltO"]))
        print("  rel L2 eta' full / int.   = %10.3E / %10.3E" % (r["l2_full"], r["l2_int"]))
        print("  max|d eta'| full / int.   = %10.3E / %10.3E m" % (r["max_full"], r["max_int"]))
        print("  rel L2 u / v (interior)   = %10.3E / %10.3E" % (r["u_int"], r["v_int"]))
    print("=================================================")

    # ---- per-resolution figure (final time) --------------------------------
    r = res["final"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    vm = max(np.abs(r["etaS"]).max(), np.abs(r["etaO"]).max(), 1e-30)
    for k, (key, ttl) in enumerate((("etaS", "SslabLU"), ("etaO", "Oceananigans"))):
        pc = ax[0, k].pcolormesh(xc, yc, r[key].T, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='auto')
        fig.colorbar(pc, ax=ax[0, k], shrink=0.85, label=r"$\eta'$ [m]")
        ax[0, k].set_title(r"%s $\eta - \langle\eta\rangle$" % ttl)
    dm = max(np.abs(r["d"]).max(), 1e-30)
    pc = ax[0, 2].pcolormesh(xc, yc, r["d"].T, cmap='PuOr', vmin=-dm, vmax=dm, shading='auto')
    fig.colorbar(pc, ax=ax[0, 2], shrink=0.85, label=r"$\Delta\eta'$ [m]")
    ax[0, 2].set_title("SslabLU - Oceananigans\nrel L2 %.2e (int. %.2e)" % (r["l2_full"], r["l2_int"]))
    pc = ax[1, 2].pcolormesh(xc, yc, Hdiff.T, cmap='BrBG', shading='auto',
                             vmin=-np.abs(Hdiff).max(), vmax=np.abs(Hdiff).max())
    fig.colorbar(pc, ax=ax[1, 2], shrink=0.85, label='[m]')
    ax[1, 2].set_title(r"$H^u_{Ocean} - H_{analytic}$ at u-faces")
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 2]):
        if sponge_w > 0:
            for y0, y1 in ((0.0, sponge_w), (1.0 - sponge_w, 1.0)):
                a.axhspan(y0, y1, facecolor='none', edgecolor='0.4', hatch='//', lw=0)
        a.set_xlabel('x / L'); a.set_ylabel('y / L'); a.set_aspect('equal')

    zS, zO = r["etaS"].mean(axis=0), r["etaO"].mean(axis=0)
    ax[1, 0].plot(zS, yc, 'C0', label='SslabLU')
    ax[1, 0].plot(zO, yc, 'C3--', label='Oceananigans')
    ax[1, 0].set_xlabel(r"$\langle\eta'\rangle_x$ [m]"); ax[1, 0].set_ylabel('y / L')
    ax[1, 0].set_title('zonal-mean SSH')
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, alpha=0.3)
    ax[1, 1].plot(zS - zO, yc, 'k')
    ax[1, 1].axvline(0, color='0.5', lw=0.6)
    ax[1, 1].set_xlabel(r"$\Delta\langle\eta'\rangle_x$ [m]"); ax[1, 1].set_ylabel('y / L')
    ax[1, 1].set_title('zonal-mean difference (S - O)')
    ax[1, 1].grid(True, alpha=0.3)
    for a in (ax[1, 0], ax[1, 1]):
        if sponge_w > 0:
            for y0, y1 in ((0.0, sponge_w), (1.0 - sponge_w, 1.0)):
                a.axhspan(y0, y1, color='0.85', zorder=0)

    fig.suptitle("SSH comparison at t = %.1f h, Oceananigans %d x %d (hatched: SslabLU wall sponge)"
                 % (r["t"] / 3600.0, n, n), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "channel_ssh_compare_n%d.png" % n
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote " + out)

if not rows:
    sys.exit("nothing compared")

# ---- CSV ----------------------------------------------------------------------
rows = np.array(rows)
with open("channel_ssh_compare.csv", 'w') as f:
    f.write("n,t_hours,is_final,relL2_full,relL2_interior,maxabs_full,maxabs_interior,"
            "tilt_S,tilt_O,mean_S,mean_O,relL2_u_interior,relL2_v_interior,maxabs_Hdiff\n")
    np.savetxt(f, rows, fmt='%.16e', delimiter=',')
print("wrote channel_ssh_compare.csv (%d rows)" % rows.shape[0])

# ---- convergence vs Oceananigans resolution -----------------------------------
if len({c[0] for c in conv["final"]}) >= 2:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for tag, mk in (("mid", 's'), ("final", 'o')):
        c = np.array(sorted(conv[tag]))
        dx = L / c[:, 0] / 1e3
        ax.loglog(dx, c[:, 1], mk + '-', label='%s, full domain' % tag)
        ax.loglog(dx, c[:, 2], mk + '--', label='%s, interior' % tag)
    c = np.array(sorted(conv["final"]))
    dx = L / c[:, 0] / 1e3
    ax.loglog(dx, c[-1, 2] * (dx / dx[-1]) ** 2, 'k:', label=r'$O(\Delta x^2)$')
    ax.set_xlabel(r'Oceananigans $\Delta x$ [km]')
    ax.set_ylabel(r"$\|\eta'_S - \eta'_O\| / \|\eta'_S\|$")
    ax.set_title('SSH difference vs Oceananigans resolution\n(plateau = model difference)')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("channel_ssh_compare_convergence.png", dpi=200)
    print("wrote channel_ssh_compare_convergence.png")
