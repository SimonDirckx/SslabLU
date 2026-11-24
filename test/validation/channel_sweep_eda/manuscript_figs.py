"""Manuscript figures for the centered-ridge barotropic channel (solver + rankiso
sweeps, see manuscript_load.py). Writes SSLABLU_EDA_OUT/manuscript_figs/*.pdf
and *.png and prints the numbers quoted with them.

  fig1 problem (bathymetry, slabs, eta and velocity of the reference run) and,
       if the Oceananigans runs are found (SSLABLU_MS_OCEAN, below), the
       difference from Oceananigans with its grid convergence as an inset
  fig2 spin-up of the reference run
  fig3 self-convergence, slab-count invariance, volume drift / wall flux
  fig4 cost vs DoF, setup in time steps, composition of one step (N = 8, dense rb)
  fig5 slab count at a fixed discretization (the finest npan_y of the solver
       sweep with every p); panel (d) memory held, if the memory runs are found
  fig9 memory held after setup vs DoF (N = 8) and its composition
  tab_memory.tex / tab_memory.csv  memory by component, dense and HBS
  fig6 HBS compression error vs rank (rankiso sweep, npan_y = 32)
  fig7 error vs run time (solver sweep)
  fig8 compression across interface sizes: error vs interface size and vs N,
       run time compressed / dense

Reference: the dense N = 8 run with the most DoF (manuscript_common.REF); the
next finest bounds its error (REF_ERR). A layout's discretization error is its
dense N = 8 run vs REF. Run times: setup (assemble/factor/keep) + 800 steps of
the conservative form; the solver preset steps both forms, rankiso one, so
total wall times of the two presets are not comparable.

Oceananigans comparison (fig1 d): oceananigans_n<n>.npz files, converted from
the data_final.jld2 of reentrant_channel_sslablu.jl (SSLABLU_RIDGE_MIDPANEL=0,
run to the reference run's final time), in SSLABLU_MS_OCEAN (default
central_ridge_runs/oceananigans_ridgectr). Both fields are de-meaned and
compared at the Oceananigans cell centers, as in channel_ssh_compare.py. Grids
the reference run did not sample (640^2) come from a rerun of it,
SSLABLU_MS_REF_RERUN (default central_ridge_runs/reference_rerun_ssh640/<run>),
used only if it reproduces the reference's own samples to 1e-10.

Memory (fig5 d, fig9, tab_memory): channel_timestep_mem.json files of the
SSLABLU_MEMSTATS=1 runs (central_ridge_runs/run_memstats.sh: setup + 5 steps,
same settings as the timing runs) under SSLABLU_MS_MEM (default
central_ridge_runs/memstats_novtx). Memory held after setup, from the buffers
reachable from each component (shared buffers once); MUMPS slab factors are
INFOG(9)/(10) entries, working storage excluded. Components partition the total:
leaf cache (GPU) | MUMPS slab factors | rest of the slab solvers (hpsalt slab
discretization: sparse slab systems, grids, leaf operators) | S-maps |
interface factorization | other (model state)."""
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.transforms
from matplotlib.lines import Line2D
from manuscript_common import *
from eda_common import REPO

FIG = os.path.join(OUT, "manuscript_figs", "")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 4.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
PC = {8: "#0072B2", 12: "#E69F00", 16: "#009E73"}          # Okabe-Ito, one color per p
NM = {8: "o", 16: "s", 32: "^", 64: "D", 128: "v"}          # one marker per slab count N
GREY = "#777777"
PS, RANKS = (8, 12, 16), (4, 6, 8, 12)
FLOOR = (1e-9, 1e-8)       # spread of repeated compressed runs (printed below), shaded


def save(fig, name):
    fig.savefig(FIG + name + ".pdf"); fig.savefig(FIG + name + ".png", dpi=220); plt.close(fig)


def panel(a, s):
    a.text(0.0, 1.02, s, transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")


def tidy(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def leaves(n):
    return "%d$\\times$%d" % (2 * n, n)


# dense N = 8 layouts and their discretization error vs REF
LAY = sorted({(r["p"], r["npy"]) for r in rows if r["kind"] == "rb" and r["N"] == 8})
NPYS = sorted({n for _, n in LAY})
REFLAY = (REF["p"], REF["npy"])
ERR = {(p, n): rel(fld(rbany(n, p)), fld(REF)) for p, n in LAY if (p, n) != REFLAY}
disc = lambda p, n: ERR.get((p, n), REF_ERR)
NSOF = lambda n: [N for N in (8, 16, 32, 64, 128) if N <= n]

# ---------------------------------------------------------------- 1. the problem
RHR, RKB, GY0, GY1, GW = 0.8, 40.0, 1 / 6, 0.5, 0.05
xs = (np.arange(320) + 0.5) / 320
X, Y = np.meshgrid(xs, xs, indexing="ij")
gap = 1 - 0.5 * (np.tanh((Y - GY0) / GW) - np.tanh((Y - GY1) / GW))
Hf = 1 - RHR * gap * np.exp(RKB * (np.cos(2 * np.pi * (X - 0.5)) - 1))
eta, u, v = fld(REF, "eta"), fld(REF, "u"), fld(REF, "v")

# Oceananigans runs at the reference run's final time, one per grid n
OCE_DIR = os.environ.get("SSLABLU_MS_OCEAN", os.path.join(REPO, "central_ridge_runs", "oceananigans_ridgectr"))
_REFNPZ = np.load(os.path.join(REPO, ROOTS[REF["sweep"]], REF["dir"], "channel_timestep_ssh.npz"))
# a rerun of the reference that also samples finer grids (SSLABLU_SSH_NS), used
# only if it reproduces the reference on the grids both have
_RERUN = os.environ.get("SSLABLU_MS_REF_RERUN", os.path.join(REPO, "central_ridge_runs", "reference_rerun_ssh640", REF["dir"],
                                                             "channel_timestep_ssh.npz"))
if os.path.exists(_RERUN):
    _X = np.load(_RERUN)
    _dev = max(rel(_X["eta_final_%d" % n], _REFNPZ["eta_final_%d" % n]) for n in _REFNPZ["ns"] if "eta_final_%d" % n in _X)
    if _dev < 1e-10:
        print("reference samples from the rerun %s (reproduces the reference to %.1e)" % (_RERUN, _dev))
        _REFNPZ = _X
    else:
        print("ignoring the rerun %s: differs from the reference by %.1e" % (_RERUN, _dev))
OCE = {}
for _f in sorted(glob.glob(os.path.join(OCE_DIR, "oceananigans_n*.npz"))):
    _O = dict(np.load(_f))
    if abs(float(_O["t_final"]) - float(_REFNPZ["t_final"])) > 1.0 or abs(float(_O["ridge_xc"]) - 0.5) > 1e-12:
        print("skip %s: t = %.0f s, ridge x/L = %.4f (reference: t = %.0f s, centered ridge)"
              % (_f, _O["t_final"], _O["ridge_xc"], _REFNPZ["t_final"]))
        continue
    n = int(_O["n"])
    if "eta_final_%d" % n not in _REFNPZ:
        print("skip %s: the reference run has no samples on the %d^2 grid" % (_f, n))
        continue
    eS = _REFNPZ["eta_final_%d" % n] - float(_REFNPZ["eta_mean_final"])
    eO = _O["eta_final"] - _O["eta_final"].mean()
    OCE[n] = dict(d=eS - eO, rel=rel(eO, eS), dt=float(_O["dt"]), nsteps=int(_O["nsteps"]))

if OCE:
    fig, ax = plt.subplots(2, 2, figsize=(5.9, 5.0), gridspec_kw=dict(wspace=0.55, hspace=0.42))
    ax = ax.ravel()
else:
    print("no Oceananigans runs in %s: fig1 without panel (d)" % OCE_DIR)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.45), gridspec_kw=dict(wspace=0.62))
ext = (0, 1, 0, 1)
im = ax[0].imshow(Hf.T, origin="lower", extent=ext, cmap="cividis", vmin=0.2, vmax=1)
N_, npy_ = 8, 16
for n in range(N_):
    ax[0].axvline(n / N_, color="w", lw=0.7, ls=(0, (3, 2)))
c, hw = 0.5, 1 / N_
ax[0].add_patch(plt.Rectangle((c - hw, 0), 2 * hw, 1, fill=False, ec="#d55e00", lw=1.3))
for k in range(1, 4 * npy_ // N_):
    ax[0].plot([c - hw + k * 2 * hw / (4 * npy_ // N_)] * 2, [0, 1], color="#d55e00", lw=0.35)
for k in range(1, npy_):
    ax[0].plot([c - hw, c + hw], [k / npy_] * 2, color="#d55e00", lw=0.35)
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03).set_label("$H/H_0$")
ax[0].set_title("(a) bathymetry and slabs", fontsize=8.5, loc="left")
m = np.abs(eta).max()
im = ax[1].imshow(eta.T * 100, origin="lower", extent=ext, cmap="RdBu_r", vmin=-m * 100, vmax=m * 100)
ax[1].contour(xs, xs, Hf.T, levels=[0.5], colors="k", linewidths=0.5, linestyles="--")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).set_label(r"$\eta - \bar\eta$  [cm]")
ax[1].set_title(r"(b) sea-surface height, $t$ = 100 h", fontsize=8.5, loc="left")
sp = np.hypot(u, v)
im = ax[2].imshow(sp.T * 100, origin="lower", extent=ext, cmap="magma_r", vmin=0)
ax[2].streamplot(xs, xs, u.T, v.T, color="w", linewidth=0.45, density=1.1, arrowsize=0.5)
ax[2].contour(xs, xs, Hf.T, levels=[0.5], colors="#56b4e9", linewidths=0.6, linestyles="--")
fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03).set_label(r"$|\mathbf{u}|$  [cm/s]")
ax[2].set_title("(c) velocity, $t$ = 100 h", fontsize=8.5, loc="left")
if OCE:
    nO = max(OCE)
    dmax = np.abs(OCE[nO]["d"]).max()
    im = ax[3].imshow(OCE[nO]["d"].T * 1e6, origin="lower", extent=ext, cmap="PuOr_r", vmin=-dmax * 1e6, vmax=dmax * 1e6)
    ax[3].contour(xs, xs, Hf.T, levels=[0.5], colors="k", linewidths=0.5, linestyles="--")
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.03).set_label(r"SslabLU $-$ Oceananigans  [$\mu$m]")
    ax[3].set_title(r"(d) difference from Oceananigans, $%d^2$" % nO, fontsize=8.5, loc="left")
    ns_ = sorted(OCE)
    hx = np.array([1000.0 / n for n in ns_])                    # grid spacing [km], L = 1000 km
    ri = np.array([OCE[n]["rel"] for n in ns_])
    ins = ax[3].inset_axes([0.14, 0.08, 0.29, 0.25])           # quiet lower-left quadrant, clear of ridge and gap
    ins.loglog(hx, ri, "o-", color="k", ms=2.5, lw=0.9)
    ins.loglog(hx, 0.6 * ri[-1] * (hx / hx[-1]), color=GREY, lw=0.6, ls=":")    # first order (observed)
    ins.text(hx[1], 0.6 * ri[-1] * (hx[1] / hx[-1]) * 0.75, "1", color=GREY, fontsize=6, ha="center", va="top")
    ins.set_xticks(hx); ins.set_xticklabels(["%.2g" % h for h in hx]); ins.minorticks_off()
    ins.tick_params(labelsize=5.5, length=1.5, pad=1)
    ins.set_xlabel(r"$\Delta x_{\rm O}$ [km]", fontsize=5.5, labelpad=1)
    ins.set_title(r"rel. $L^2$ difference", fontsize=5.5, pad=2)
    ins.patch.set_alpha(0.85)
for i, a in enumerate(ax):
    a.set_xlabel("$x/L$"); a.set_ylabel("$y/L$") if i % (2 if OCE else 3) == 0 else None
    a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1]); a.set_xlim(0, 1); a.set_ylim(0, 1)
save(fig, "fig1_problem")

# ---------------------------------------------------------------- 2. spin-up (reference run)
T = REF["ts"]
th = T["t_hours"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.38))
ax[0].plot(th, T["energy"] * 1e3, color="k")
ax[0].set_ylabel(r"energy  [$10^{-3}$ J/$\rho_0$]")
ax[1].plot(th, T["max_u"] * 100, color="k", label=r"max $|u|$")
ax[1].plot(th, T["max_v"] * 100, color=GREY, ls="--", label=r"max $|v|$")
ax[1].set_ylabel("[cm/s]"); ax[1].legend(loc="upper left")
ax[2].plot(th, T["tilt_NmS"] * 100, color="k")
ax[2].set_ylabel("N$-$S SSH tilt  [cm]")
for i, a in enumerate(ax):
    a.set_xlabel("$t$  [h]"); a.set_xlim(0, 100); tidy(a); panel(a, "(%s)" % "abc"[i])
save(fig, "fig2_spinup")

# ---------------------------------------------------------------- 3. accuracy
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.38))
a = ax[0]
for p in PS:
    ns = [n for n in NPYS if (p, n) in ERR]
    d = [rbany(n, p)["dof"] for n in ns]
    a.loglog(d, [ERR[(p, n)] for n in ns], "-o", color=PC[p])
    a.loglog(d, [rel(fld(rbany(n, p), "u"), fld(REF, "u")) for n in ns], ":", marker="o", mfc="none", color=PC[p])
a.set_xlabel("degrees of freedom"); a.set_ylabel("rel. $L^2$ error vs. reference")
a.add_artist(a.legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS], loc="lower left"))
a.legend(handles=[Line2D([], [], color=GREY, ls="-", marker="o", label=r"$\eta$"), Line2D([], [], color=GREY, ls=":", marker="o", mfc="none", label="$u$")],
         loc="upper right", handlelength=1.6, fontsize=7)
a.set_ylim(1e-12, 1)
a = ax[1]
for p, n in LAY:
    b = rbany(n, p)
    for N in NSOF(n)[1:]:
        x = rbany(n, p, N)
        a.semilogy(b["dof"], max(rel(fld(x, f), fld(b, f)) for f in ("eta", "u", "v")), NM[N], color=PC[p], mfc="none", ms=6, mew=0.8)
    if (p, n) in ERR:                                         # the reference has no error of its own to show
        a.semilogy(b["dof"], disc(p, n), "o", color=PC[p], ms=4.5)
a.set_xscale("log")
a.axhspan(1e-16, 1e-11, color="#eeeeee", lw=0, zorder=0)
a.text(0.97, 0.04, "round-off band", transform=a.transAxes, ha="right", fontsize=7, color=GREY)
a.set_ylim(1e-14, 1e2); a.set_xlabel("degrees of freedom")
a.set_ylabel("rel. difference, $N$ slabs vs. $N$ = 8")
a.legend(handles=[Line2D([], [], color=GREY, ls="", marker="o", label=r"disc. error, $\eta$")]
         + [Line2D([], [], color=GREY, ls="", marker=NM[N], mfc="none", label="$N$ = %d" % N) for N in (16, 32, 64, 128)],
         loc="upper right", bbox_to_anchor=(1.02, 1.0), ncol=2, columnspacing=0.6, handletextpad=0.2, handlelength=1.2, fontsize=6.5)
a = ax[2]
for p in PS:
    rr = [rbany(n, p) for n in NPYS if (p, n) in LAY]
    a.loglog([r["dof"] for r in rr], [abs(r["mass_rel"]) for r in rr], "-o", color=PC[p])
    a.loglog([r["dof"] for r in rr], [r["wall_v"] / r["ts"]["max_u"][-1] for r in rr], ":", marker="v", mfc="none", color=PC[p])
a.set_xlabel("degrees of freedom"); a.set_ylabel("normalized defect")
a.legend(handles=[Line2D([], [], color=GREY, marker="o", label=r"volume drift $|\Delta V| / \int|\eta|$"),
                  Line2D([], [], color=GREY, ls=":", marker="v", mfc="none", label=r"wall $\max|v| \,/\, \max|u|$")],
         loc="center left", bbox_to_anchor=(0, 0.42))
a.set_ylim(1e-17, 3)
for i, (a, t) in enumerate(zip(ax, ("self-convergence", "slab-count invariance", "conservation and wall closure"))):
    tidy(a); a.set_title("(%s) %s" % ("abc"[i], t), fontsize=8.5, loc="left")
save(fig, "fig3_accuracy")

# ---------------------------------------------------------------- memory runs (optional)
MEM_DIR = os.environ.get("SSLABLU_MS_MEM", os.path.join(REPO, "central_ridge_runs", "memstats_novtx"))
_tb = lambda h: h["cpu"] + h["gpu"] + h["mumps"]


def _memrow(d):
    h, ph = d["held"], d["phases"]
    m = dict(p=d["p"], npy=d["npan_y"], N=d["N"], solver=d["solver"], fastleaf=bool(d["fastleaf"]),
             cache=_tb(h["  leaf cache"]), mumps=h["slab solvers (all)"]["mumps"], smap=_tb(h["S-maps"]),
             ifac=_tb(h["interface factor"]), total=_tb(h["model total"]),
             gpu_peak=max(v.get("gpu_peak", 0) for v in ph.values()), host_peak=max(v["cpu_peak"] for v in ph.values()))
    m["slab"] = _tb(h["slab solvers (all)"]) - m["cache"] - m["mumps"]
    m["iface"] = m["smap"] + m["ifac"]
    m["other"] = m["total"] - m["cache"] - m["mumps"] - m["slab"] - m["iface"]
    nx = 2 * m["npy"]                                   # as manuscript_common
    m["dof"] = nx * m["npy"] * m["p"] ** 2 + m["p"] * nx * (2 * m["npy"] + 1)
    return m


MEM = {}
for _f in glob.glob(os.path.join(MEM_DIR, "**", "channel_timestep_mem.json"), recursive=True):
    _m = _memrow(json.load(open(_f)))
    MEM[(_m["p"], _m["npy"], _m["N"], _m["solver"], _m["fastleaf"])] = _m
mem = lambda p, n, N=8, s="rb", fl=True: MEM.get((p, n, N, s, fl))
GB = 1e-9

# ---------------------------------------------------------------- 4. cost scaling, N = 8, dense rb
# (a) absolute times; (b) setup in units of steps; (c) composition of one step.
# Nothing here depends on the 800-step run length. Factorization (~0.1% of
# setup) is too small to plot; its share is printed below.
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(wspace=0.42))
a = ax[0]
for p in PS:
    rr = [rbany(n, p) for n in NPYS if (p, n) in LAY]
    d = [r["dof"] for r in rr]
    a.loglog(d, [r["asm"] for r in rr], "--s", color=PC[p], mfc="none")
    a.loglog(d, [r["keep"] for r in rr], "-.D", color=PC[p], mfc="none", ms=3.5)
    a.loglog(d, [r["leaf"] for r in rr], "-o", color=PC[p])
    a.loglog(d, [r["t_solve"] for r in rr], ":^", color=PC[p], mfc="none")
dd = np.array([2e4, 6e6])
a.loglog(dd, 0.0035 * dd / 2e4, color=GREY, lw=0.7)          # below rhs + recon.
a.text(2.5e5, 0.012, r"$\propto$ DoF", color=GREY, fontsize=7)
a.loglog(dd, 3.0 * dd / 2e4, color=GREY, lw=0.7)             # above assembly / storage
a.text(6e6, 1.25e3, r"$\propto$ DoF", color=GREY, fontsize=7, ha="right")
a.set_xlabel("degrees of freedom"); a.set_ylabel("wall time  [s]")
a.legend(handles=[Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="S-map assembly"),
                  Line2D([], [], color=GREY, ls="-.", marker="D", mfc="none", ms=3.5, label="slab storage"),
                  Line2D([], [], color=GREY, ls="-", marker="o", label="rhs + recon., per step"),
                  Line2D([], [], color=GREY, ls=":", marker="^", mfc="none", label="interface solve, per step")],
         loc="upper left", handlelength=2.2, fontsize=6.5)
a.set_ylim(3e-4, 1e5)
a = ax[1]
for p in PS:
    rr = [rbany(n, p) for n in NPYS if (p, n) in LAY]
    a.loglog([r["dof"] for r in rr], [r["once"] / r["step"] for r in rr], "-o", color=PC[p])
a.axhline(800, color=GREY, lw=0.6, ls="--")
a.text(1.1e4, 800 * 1.12, "800 steps (runs here)", color=GREY, fontsize=6.5, va="bottom")
a.set_xlabel("degrees of freedom"); a.set_ylabel("setup cost  [time steps]")
a.legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS],
         loc="upper right", bbox_to_anchor=(1, 0.86))             # between the 800 line and the curves
a.set_ylim(50, 1100)
a.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
a = ax[2]
for p in PS:
    rr = [rbany(n, p) for n in NPYS if (p, n) in LAY]
    d = [r["dof"] for r in rr]
    for f, sty in (("t_rhs", "-o"), ("t_recon", "--d"), ("t_solve", ":^")):
        a.loglog(d, [100 * r[f] / r["step"] for r in rr], sty, color=PC[p], mfc=PC[p] if f == "t_rhs" else "none", ms=4)
a.set_xlabel("degrees of freedom"); a.set_ylabel("share of one time step  [%]")
a.legend(handles=[Line2D([], [], color=GREY, ls="-", marker="o", label="predictor + slab rhs solve"),
                  Line2D([], [], color=GREY, ls="--", marker="d", mfc="none", label="slab recon. + velocity update"),
                  Line2D([], [], color=GREY, ls=":", marker="^", mfc="none", label="interface solve")],
         loc="center left", bbox_to_anchor=(0.0, 0.6), handlelength=2.2, fontsize=6.5)   # in the gap between the curves
a.set_ylim(0.5, 100)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
for i, (a, t) in enumerate(zip(ax, ("cost vs. problem size", "setup in time steps", "one time step"))):
    tidy(a); a.set_title("(%s) %s" % ("abc"[i], t), fontsize=8.5, loc="left")
save(fig, "fig4_cost")

# ---------------------------------------------------------------- 5. slab count at fixed discretization
NPY5 = max(n for n in NPYS if all(one(sweep="solver", npy=n, p=p, N=8, kind="rb") for p in PS))
NS = NSOF(NPY5)
MEM5 = all(mem(p, NPY5, N, s) for p in PS for N in NS for s in ("rb", "rbhbs"))
if MEM5:
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.7), gridspec_kw=dict(wspace=0.36, hspace=0.55))
    ax = ax.ravel()
else:
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.4))
for p in PS:
    rb_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="rb") for N in NS]
    hb_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="both", rk=p) for N in NS]
    ax[0].plot(NS, [r["leaf"] for r in rb_], "-o", color=PC[p])
    ax[1].semilogy(NS, [r["t_solve"] for r in rb_], "-o", color=PC[p])
    ax[1].semilogy(NS, [r["t_solve"] for r in hb_], "--s", color=PC[p], mfc="none")
    ax[2].semilogy(NS, [r["asm"] for r in rb_], "-o", color=PC[p])
    ax[2].semilogy(NS, [r["asm"] for r in hb_], "--s", color=PC[p], mfc="none")
    ax[2].semilogy(NS, [r["fac"] for r in rb_], "-.o", color=PC[p], ms=3)
    ax[2].semilogy(NS, [r["fac"] for r in hb_], ":s", color=PC[p], mfc="none", ms=3)
ax[0].set_ylabel("wall time per step  [s]"); ax[0].set_ylim(0)
ax[0].legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS], loc="lower right")
ax[1].set_ylabel("wall time per step  [s]")
ax[1].legend(handles=[Line2D([], [], color=GREY, marker="o", label="dense red-black"),
                      Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="HBS red-black, rank $p$")], loc="upper left")
ax[1].set_ylim(2e-3, 3)
ax[2].set_ylabel("wall time  [s]")
ax[2].legend(handles=[Line2D([], [], color=GREY, marker="o", label="assembly, dense"),
                      Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="assembly, HBS"),
                      Line2D([], [], color=GREY, ls="-.", marker="o", ms=3, label="factor, dense"),
                      Line2D([], [], color=GREY, ls=":", marker="s", mfc="none", ms=3, label="factor, HBS")],
             **(dict(loc="upper left") if MEM5 else dict(loc="lower left", bbox_to_anchor=(0, -0.02))),
             ncol=2, columnspacing=0.6, handlelength=1.8)
ax[2].set_ylim(3e-2 if MEM5 else 1e-2, 2e4 if MEM5 else 2e3)
if MEM5:                       # (d) memory held after setup: total and interface (S-maps + factorization)
    a = ax[3]
    for p in PS:
        rb_ = [mem(p, NPY5, N, "rb") for N in NS]
        hb_ = [mem(p, NPY5, N, "rbhbs") for N in NS]
        a.semilogy(NS, [GB * m["total"] for m in rb_], "-o", color=PC[p])
        a.semilogy(NS, [GB * m["total"] for m in hb_], "--s", color=PC[p], mfc="none")
        a.semilogy(NS, [GB * m["iface"] for m in rb_], "-.o", color=PC[p], ms=3)
        a.semilogy(NS, [GB * m["iface"] for m in hb_], ":s", color=PC[p], mfc="none", ms=3)
    a.set_ylabel("memory held  [GB]")
    a.legend(handles=[Line2D([], [], color=GREY, marker="o", label="total, dense"),
                      Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="total, HBS"),
                      Line2D([], [], color=GREY, ls="-.", marker="o", ms=3, label="interface, dense"),
                      Line2D([], [], color=GREY, ls=":", marker="s", mfc="none", ms=3, label="interface, HBS")],
             loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.8)
    a.set_ylim(5e-3, 2e3)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
T5 = ("leaf work per step", "interface solve per step", "one-time setup", "memory held after setup")
for i, a in enumerate(ax):
    a.set_xscale("log", base=2); a.set_xticks(NS); a.set_xticklabels([str(N) for N in NS]); a.minorticks_off()
    a.set_xlabel("double slabs $N$ (= interfaces)"); tidy(a)   # all at leaves(NPY5) leaves: in the caption
    a.set_title("(%s) %s" % ("abcd"[i], T5[i]), fontsize=8.5, loc="left")
save(fig, "fig5_slabcount")


# ---------------------------------------------------------------- 9. memory vs problem size, N = 8
# (a) held memory by component vs DoF, dense rb (HBS interface for contrast); (b) composition of the
# largest layout per p: dense | HBS | dense without the leaf cache (where run).
MEMCOMP = (("cache", "leaf cache (GPU)", "#56B4E9"), ("mumps", "slab factors (MUMPS)", "#0072B2"),
           ("slab", "slab discretization", "#999999"), ("smap", "S-maps", "#E69F00"),
           ("ifac", "interface factorization", "#D55E00"), ("other", "other", "#dddddd"))
LAY9 = [(p, n) for p, n in LAY if mem(p, n) and mem(p, n, s="rbhbs")]
if LAY9:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(wspace=0.3, width_ratios=(1, 1.15)))
    a = ax[0]
    for p in PS:
        mm = [mem(p, n) for n in NPYS if (p, n) in LAY9]
        hh = [mem(p, n, s="rbhbs") for n in NPYS if (p, n) in LAY9]
        d = [m["dof"] for m in mm]
        a.loglog(d, [GB * m["total"] for m in mm], "-o", color=PC[p])
        a.loglog(d, [GB * m["cache"] for m in mm], "--d", color=PC[p], mfc="none", ms=4)
        a.loglog(d, [GB * m["mumps"] for m in mm], "-.v", color=PC[p], mfc="none", ms=4)
        a.loglog(d, [GB * m["iface"] for m in mm], ":^", color=PC[p], mfc="none")
        a.loglog(d, [GB * m["iface"] for m in hh], ":", color=PC[p], lw=0.8, marker="x", ms=3.5)
        for n in NPYS:
            m = mem(p, n, fl=False)
            if m:
                a.loglog(m["dof"], GB * m["total"], "*", color=PC[p], ms=7, mec="k", mew=0.4, zorder=5)
    dd = np.array([1e4, 6e6])
    a.loglog(dd, 2e-5 * dd, color=GREY, lw=0.7)                # above the totals
    a.text(6e6, 2e-5 * 6e6 * 1.4, r"$\propto$ DoF", color=GREY, fontsize=7, ha="right", va="bottom")
    a.set_xlabel("degrees of freedom"); a.set_ylabel("memory held  [GB]")
    a.legend(handles=[Line2D([], [], color=GREY, marker="o", label="total"),
                      Line2D([], [], color=GREY, ls="--", marker="d", mfc="none", ms=4, label="leaf cache"),
                      Line2D([], [], color=GREY, ls="-.", marker="v", mfc="none", ms=4, label="slab factors"),
                      Line2D([], [], color=GREY, ls=":", marker="^", mfc="none", label="interface, dense"),
                      Line2D([], [], color=GREY, ls=":", lw=0.8, marker="x", ms=3.5, label="interface, HBS"),
                      Line2D([], [], color=GREY, ls="none", marker="*", ms=7, mec="k", mew=0.4, label="total, no leaf cache")]
             + [Line2D([], [], color=PC[p], lw=3, label="$p$ = %d" % p) for p in PS],
             loc="upper left", fontsize=6.2, handlelength=2.0, ncol=2, columnspacing=0.8)
    a.set_ylim(3e-4, 3e3)
    a = ax[1]
    big = [(p, max(n for q, n in LAY9 if q == p)) for p in PS if any(q == p for q, _ in LAY9)]
    bars, labs, x, xt = [], [], 0.0, []
    for p, n in big:
        grp = [("dense", mem(p, n)), ("HBS", mem(p, n, s="rbhbs"))] + ([("no cache", mem(p, n, fl=False))] if mem(p, n, fl=False) else [])
        for nm, m in grp:
            bars.append((x, m)); labs.append(nm.replace(" ", "\n")); x += 1
        xt.append((x - (len(grp) + 1) / 2, "$p$ = %d, %s\n%.1fM DoF" % (p, leaves(n), grp[0][1]["dof"] / 1e6)))
        x += 0.6
    for xb, m in bars:
        y0 = 0.0
        for k, _, c in MEMCOMP:
            a.bar(xb, GB * m[k], bottom=y0, width=0.8, color=c, edgecolor="white", lw=0.3)
            y0 += GB * m[k]
        a.text(xb, y0 + 0.3, "%.1f" % y0, ha="center", va="bottom", fontsize=6)
    a.set_xticks([b[0] for b in bars]); a.set_xticklabels(labs, fontsize=6.5, rotation=0)
    for xc, s in xt:
        a.text(xc, -0.2, s, transform=matplotlib.transforms.blended_transform_factory(a.transData, a.transAxes),
               ha="center", va="top", fontsize=7)
    a.set_ylabel("memory held  [GB]")
    a.legend(handles=[matplotlib.patches.Patch(color=c, label=l) for _, l, c in MEMCOMP][::-1],
             loc="upper right", fontsize=6.3, handlelength=1.2)
    a.set_ylim(0, 1.18 * max(GB * m["total"] for _, m in bars))
    for i, (a, t) in enumerate(zip(ax, ("memory vs. problem size ($N$ = 8)", "composition, largest layouts"))):
        tidy(a); a.set_title("(%s) %s" % ("ab"[i], t), fontsize=8.5, loc="left")
    save(fig, "fig9_memory")

# ---------------------------------------------------------------- memory table (LaTeX + CSV)
# fig5 layout at every N, plus the largest N = 8 layout per p, dense and HBS rank p.
TROWS = sorted({(p, NPY5, N) for p in PS for N in NS} | {(p, max(n for q, n in LAY9 if q == p), 8) for p in {q for q, _ in LAY9}},
               key=lambda t: (t[1] != NPY5, t[0], t[1], t[2]))
TROWS = [t for t in TROWS if mem(*t) and mem(*t, s="rbhbs")]
if TROWS:
    hdr = ["p", "leaves", "N", "DoF", "leaf_cache", "slab_factors", "slab_disc", "other",
           "smap_dense", "smap_hbs", "ifac_dense", "ifac_hbs", "total_dense", "total_hbs", "saving_pct",
           "gpu_peak_dense", "host_peak_dense"]
    csv = [",".join(hdr)]
    tex = [r"\begin{tabular}{rrrr|rrr|rr|rr|rrr}", r"\toprule",
           r" & & & & leaf & slab & slab & \multicolumn{2}{c|}{S-maps} & \multicolumn{2}{c|}{interface factor.} & \multicolumn{3}{c}{total} \\",
           r"$p$ & leaves & $N$ & DoF & cache & factors & discr. & dense & HBS & dense & HBS & dense & HBS & saving \\", r"\midrule"]
    prev = None
    for p, n, N in TROWS:
        d, h = mem(p, n, N), mem(p, n, N, "rbhbs")
        sav = 100 * (1 - h["total"] / d["total"])
        csv.append(",".join(str(v) for v in [p, "%dx%d" % (2 * n, n), N, d["dof"]] +
                   ["%.4f" % (GB * v) for v in (d["cache"], d["mumps"], d["slab"], d["other"], d["smap"], h["smap"],
                                                d["ifac"], h["ifac"], d["total"], h["total"])] +
                   ["%.1f" % sav, "%.3f" % (GB * d["gpu_peak"]), "%.3f" % (GB * d["host_peak"])]))
        grp = n == NPY5
        if prev is not None and prev != grp:
            tex.append(r"\midrule")
        prev = grp
        f2 = lambda v: "%.2f" % (GB * v)
        tex.append(" & ".join([str(p), leaves(n), str(N), "%.2fM" % (d["dof"] / 1e6), f2(d["cache"]), f2(d["mumps"]),
                               f2(d["slab"] + d["other"]), f2(d["smap"]), f2(h["smap"]), f2(d["ifac"]), f2(h["ifac"]),
                               f2(d["total"]), f2(h["total"]), "%.0f\\%%" % sav]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(FIG + "tab_memory.csv", "w").write("\n".join(csv) + "\n")
    open(FIG + "tab_memory.tex", "w").write("%% memory held after setup [GB]; generated by manuscript_figs.py from %s\n" % MEM_DIR
                                           + "% 'slab discr.' includes model state (other); leaf cache is GPU, the rest host\n"
                                           + "\n".join(tex) + "\n")


# ---------------------------------------------------------------- 6. HBS rank (rankiso, npan_y = 32)
def herr(k, n, p, N, r):
    x = (one(sweep="rankiso", npy=n, p=p, N=N, kind="smap", rk=r) if k == "smap" else
         one(sweep="rankiso", npy=n, p=p, N=N, solver="rbhbs", rk=int(1.5 * p), rbrk=r))
    return rel(fld(x), fld(rbof(x))) if x else np.nan


PMK = {8: dict(marker="o", ms=4.5), 12: dict(marker="s", ms=4.5), 16: dict(marker="^", ms=4, mfc="none", mew=0.9)}  # p = 12 and 16 overlap
fig = plt.figure(figsize=(7.2, 5.0))
sfs = fig.subfigures(2, 1, hspace=0.04)
for i, (k, nm) in enumerate((("smap", "S-map compression (dense solver)"), ("solver", "solver compression (S-maps at rank 1.5$p$)"))):
    sfs[i].suptitle(nm, fontsize=9, fontweight="bold", y=0.99)
    ax = sfs[i].subplots(1, 3, sharey=True, gridspec_kw=dict(wspace=0.08))
    sfs[i].subplots_adjust(top=0.82, bottom=0.18 if i else 0.08)
    for j, N in enumerate((8, 16, 32)):
        a = ax[j]
        a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
        for p in PS:
            a.semilogy(RANKS, [herr(k, 32, p, N, r) for r in RANKS], "-", color=PC[p], zorder=3, **PMK[p])
            a.axhline(disc(p, 32), color=PC[p], ls=":", lw=0.9, zorder=2)
        a.set_xticks(RANKS); a.set_ylim(1e-10, 3e-2); tidy(a)
        a.set_title("(%s) $N$ = %d" % ("abcdef"[3 * i + j], N), fontsize=8.5, loc="left")
        if j == 0:
            a.set_ylabel(r"rel. error in $\eta$ vs. dense")
        a.set_xlabel("rank") if i == 1 else a.set_xticklabels([])
fig.legend(handles=[Line2D([], [], color=PC[p], label="$p$ = %d" % p, **PMK[p]) for p in PS]
           + [Line2D([], [], color=GREY, ls=":", label="discretization error of that $p$, %s leaves" % leaves(32)),
              Line2D([], [], color="#eeeeee", lw=6, label="repeat-run spread")],
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), columnspacing=1.2)
save(fig, "fig6_rank")

# ---------------------------------------------------------------- 7. accuracy vs cost (solver sweep)
fig, a = plt.subplots(figsize=(4.8, 3.0))
for x in find(sweep="solver"):
    a.loglog(x["cost"], rel(fld(x), fld(REF)), "o" if x["kind"] == "rb" else "x", color=PC[x["p"]],
             ms=4 if x["kind"] == "rb" else 4.5, mew=0.9, alpha=1 if x["kind"] == "rb" else 0.7)
for p in PS:
    ns = [n for n in NPYS if one(sweep="solver", npy=n, p=p, N=8, kind="rb")]
    a.plot([one(sweep="solver", npy=n, p=p, N=8, kind="rb")["cost"] for n in ns], [disc(p, n) for n in ns], "-", color=PC[p], lw=0.8, zorder=0)
a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
a.axhline(REF_ERR, color=GREY, lw=0.6, ls="--")
a.text(min(x["cost"] for x in find(sweep="solver")), REF_ERR * 1.5, "reference error bound", fontsize=6.5, color=GREY, ha="left")
TK = [t for t in (10, 30, 100, 300, 1000, 3000, 10000) if min(x["cost"] for x in rows) / 1.5 < t < max(x["cost"] for x in rows) * 1.5]
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xticks(TK); a.set_xticklabels([str(t) for t in TK])
a.set_xlabel("run time, setup + 800 steps  [s]"); a.set_ylabel(r"relative error in $\eta$ vs. reference")
a.legend(handles=[Line2D([], [], color=PC[p], marker="o", ls="-", lw=0.8, label="$p$ = %d" % p) for p in PS]
         + [Line2D([], [], color=GREY, marker="o", ls="", label="dense"),
            Line2D([], [], color=GREY, marker="x", ls="", label="HBS, rank\n$p/2, p, 3p/2$"),
            Line2D([], [], color="#eeeeee", lw=6, label="repeat-run spread")],
         loc="center left", bbox_to_anchor=(1.0, 0.5))
tidy(a); a.set_title("accuracy vs. run time", fontsize=8.5, loc="left")
save(fig, "fig7_error_cost")

# ---------------------------------------------------------------- 8. compression across interface sizes (rankiso)
ISO = sorted({(r["p"], r["npy"]) for r in find(sweep="rankiso", kind="rb", N=8)}, key=lambda t: t[0] * t[1])
NPYB = max(n for _, n in ISO)
PB = max(p for p, n in ISO if n == NPYB)
NC = {8: "#0072B2", 32: "#D55E00"}
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw=dict(wspace=0.42))
a = ax[0]
for N in (8, 32):
    for k, mk in (("smap", "o"), ("solver", "s")):
        pts = sorted((p * n, herr(k, n, p, N, 8)) for p, n in ISO)
        a.loglog(*zip(*pts), mk, color=NC[N], mfc=NC[N] if k == "smap" else "none", mew=0.9, ls="-" if k == "smap" else "--", lw=0.7)
a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
a.set_xlabel("interface size $n_c$ = npan$_y\\,p$"); a.set_ylabel(r"rel. error in $\eta$, compressed vs. dense")
NCT = sorted({p * n for p, n in ISO})
a.legend(handles=[Line2D([], [], color=NC[N], marker="o", ls="", label="$N$ = %d" % N) for N in (8, 32)]
         + [Line2D([], [], color=GREY, marker="o", ls="-", lw=0.7, label="S-maps"),
            Line2D([], [], color=GREY, marker="s", mfc="none", ls="--", lw=0.7, label="solver")], loc="upper left", ncol=2, fontsize=7)
a.text(0.97, 3e-9, "repeat-run spread", transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color=GREY)
a.set_ylim(1e-9, 1e-2)
a = ax[1]
NSB = NSOF(NPYB)
for k, ls in (("smap", "-"), ("solver", "--")):
    for r, mk in ((8, "o"), (12, "s")):
        a.loglog(NSB, [herr(k, NPYB, PB, N, r) for N in NSB], ls, marker=mk, color=PC[PB], mfc=PC[PB] if r == 8 else "none", mew=0.9)
a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
a.axhline(disc(PB, NPYB), color=PC[PB], ls=":", lw=0.9)
a.text(NSB[-1], disc(PB, NPYB) * 2, "discretization error", fontsize=6.5, color=PC[PB], ha="right")
a.set_xscale("log", base=2); a.set_xticks(NSB); a.set_xticklabels([str(N) for N in NSB]); a.minorticks_off()
a.set_xlabel("double slabs $N$ (= interfaces)"); a.set_ylabel(r"rel. error in $\eta$, compressed vs. dense")
a.legend(handles=[Line2D([], [], color=GREY, ls="-", label="S-maps"), Line2D([], [], color=GREY, ls="--", label="solver"),
                  Line2D([], [], color=GREY, marker="o", ls="", label="rank 8"), Line2D([], [], color=GREY, marker="s", mfc="none", ls="", label="rank 12")],
         loc="upper left", ncol=2, fontsize=7)
a.text(0.97, 3e-9, "repeat-run spread", transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color=GREY)
a = ax[2]
for k, mk in (("smap", "o"), ("solver", "s")):
    for p in PS:
        pts = []
        for pp, n in ISO:
            if pp != p:
                continue
            x = (one(sweep="rankiso", npy=n, p=p, N=8, kind="smap", rk=8) if k == "smap" else
                 one(sweep="rankiso", npy=n, p=p, N=8, solver="rbhbs", rk=int(1.5 * p), rbrk=8))
            pts.append((p * n, x["wall"] / rbof(x)["wall"]))
        if pts:
            a.semilogx(*zip(*sorted(pts)), "-" if k == "smap" else "--", marker=mk, color=PC[p], mfc=PC[p] if k == "smap" else "none", mew=0.9)
a.axhline(1.0, color=GREY, lw=0.6)
a.set_xlabel("interface size $n_c$"); a.set_ylabel("run time ratio  ($N$ = 8)")
a.legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS]
         + [Line2D([], [], color=GREY, ls="-", marker="o", label="S-maps, rank 8"), Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="solver, rank 8")],
         loc="lower left", fontsize=7)
for a in (ax[0], ax[2]):
    a.set_xticks(NCT); a.set_xticklabels([str(t) for t in NCT], rotation=45, fontsize=6.5)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# (b): %s leaves (NPYB), stated in the caption
F8T = ("rank 8 vs. interface size", "vs. slab count, $p$ = %d" % PB, "compressed / dense run time")
for i, a in enumerate(ax):
    tidy(a); a.set_title("(%s) %s" % ("abc"[i], F8T[i]), fontsize=8.5, loc="left")
save(fig, "fig8_rank_scaling")

# ---------------------------------------------------------------- numbers quoted with the figures
g = lambda v: "%.1e" % v
print("data:", ROOTS)
print("reference: dense p = %d, %s leaves, N = 8 (%s sweep); closest other dense run p = %d, %s differs by %s (reference error bound)"
      % (REF["p"], leaves(REF["npy"]).replace("$\\times$", "x"), REF["sweep"], REF2["p"], leaves(REF2["npy"]).replace("$\\times$", "x"), g(REF_ERR)))
if OCE:
    print("vs Oceananigans (%s), eta at t = 100 h, de-meaned; SSH range %.2f cm:" % (OCE_DIR, 100 * np.ptp(eta)))
    for n in sorted(OCE):
        print("  n = %3d (dx %5.2f km, dt %g s x %d): rel L2 %s, max |diff| %.1f um = %.2f%% of the SSH range"
              % (n, 1000.0 / n, OCE[n]["dt"], OCE[n]["nsteps"], g(OCE[n]["rel"]), 1e6 * np.abs(OCE[n]["d"]).max(),
                 100 * np.abs(OCE[n]["d"]).max() / np.ptp(eta)))
    _ns = sorted(OCE)
    print("  observed order in dx:", ", ".join("%.2f" % np.log2(OCE[a]["rel"] / OCE[b]["rel"]) for a, b in zip(_ns, _ns[1:])))
print("discretization error in eta, t = 100 h:")
for p in PS:
    print("  p=%2d: %s" % (p, "  ".join("%s %s" % (leaves(n).replace("$\\times$", "x"), g(ERR[(p, n)])) for n in NPYS if (p, n) in ERR)))
print("observed orders, finest two grids per p:", {p: "%.1f" % np.log2(ERR[(p, ns[-2])] / ERR[(p, ns[-1])])
      for p in PS for ns in [[n for n in NPYS if (p, n) in ERR]] if len(ns) > 1})
print("slab-count invariance, max over layouts/fields: %s" % g(max(max(rel(fld(rbany(n, p, N), f), fld(rbany(n, p), f)) for f in ("eta", "u", "v"))
      for p, n in LAY for N in NSOF(n)[1:])))
print("GATE max %s | rb vs Thomas max %s | cached leaf path vs hpsalt max %s | wall max|v| %s m/s" % (
    g(max(max(r["gate"].values()) for r in rows)), g(max(r["check"] for r in rows if r["solver"] == "rb")),
    g(np.nanmax([r["leaf_check"] for r in rows])), g(max(r["wall_v"] for r in rows))))
print("per step and run composition, dense rb, N = 8:")
for p, n in LAY:
    r = rbany(n, p)
    print("  p=%2d %8s DoF %7d: step %.4f s (rhs %.4f, interface %.5f, recon %.4f) | one-time %.1f s = %.0f steps | run %.0f s: setup %.0f%%, interface solve %.1f%%" % (
        p, leaves(n).replace("$\\times$", "x"), r["dof"], r["step"], r["t_rhs"], r["t_solve"], r["t_recon"], r["once"], r["once"] / r["step"], r["cost"],
        100 * r["once"] / r["cost"], 100 * 800 * r["t_solve"] / r["cost"]))
_rr = [rbany(n, p) for p, n in LAY]
print("fitted DoF exponents, all dense N = 8 layouts (DoF > 3e5 in parentheses):",
      ", ".join("%s %.2f (%.2f)" % (f, *[np.polyfit(np.log([r["dof"] for r in s]), np.log([r[f] for r in s]), 1)[0]
                                     for s in (_rr, [r for r in _rr if r["dof"] > 3e5])])
                for f in ("asm", "keep", "fac", "leaf", "t_solve", "once")))
print("factorization: <= %.2f%% of setup, <= %.2f s" % (max(100 * r["fac"] / r["once"] for r in _rr), max(r["fac"] for r in _rr)))
print("%s leaves, step [s] for N = %s (dense | HBS rank p) and run time HBS / dense:" % (leaves(NPY5).replace("$\\times$", "x"), "/".join(map(str, NS))))
for p in PS:
    a_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="rb") for N in NS]
    b_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="both", rk=p) for N in NS]
    print("  p=%2d: %s | %s | %s" % (p, " / ".join("%.3f" % x["step"] for x in a_), " / ".join("%.3f" % x["step"] for x in b_),
                                   " / ".join("%.2f" % (y["wall"] / x["wall"]) for x, y in zip(a_, b_))))
for k in ("smap", "solver"):
    sl = [-np.polyfit(RANKS, np.log10([herr(k, 32, p, N, r) for r in RANKS]), 1)[0] for p in PS for N in (8, 16, 32)]
    print("%-6s compression error decay (npan_y = 32): %.2f..%.2f decades per unit rank" % (k, min(sl), max(sl)))
    e8 = [herr(k, n, p, 8, 8) for p, n in ISO]
    print("       rank 8, N = 8, interface sizes %d..%d: error %s..%s" % (min(p * n for p, n in ISO), max(p * n for p, n in ISO), g(min(e8)), g(max(e8))))
    gr = [herr(k, NPYB, PB, 2 * N, r) / herr(k, NPYB, PB, N, r) for r in (8, 12) for N in NSB[:-1]]
    print("       growth per doubling of N (%s, p = %d, ranks 8/12): median x%.1f" % (leaves(NPYB).replace("$\\times$", "x"), PB, np.median(gr)))
print("repeated compressed runs (solver sweep RK = RB_RK = 12 vs rankiso solver:12, p = 8, same settings):")
for n in NPYS:
    for N in NSOF(n):
        x = one(sweep="solver", npy=n, p=8, N=N, solver="rbhbs", rk=12)
        y = one(sweep="rankiso", npy=n, p=8, N=N, solver="rbhbs", rk=12, rbrk=12)
        if x and y:
            print("  %8s N=%3d: error %s / %s, difference %s" % (leaves(n).replace("$\\times$", "x"), N, g(rel(fld(x), fld(rbof(x)))), g(rel(fld(y), fld(rbof(y)))), g(rel(fld(x), fld(y)))))
if MEM:
    print("memory held after setup [GB] (%s), dense | HBS rank p; interface = S-maps + factorization:" % MEM_DIR)
    for p, n, N in TROWS:
        d, h = mem(p, n, N), mem(p, n, N, "rbhbs")
        print("  p=%2d %8s N=%2d DoF %8d: total %6.2f | %6.2f (saving %4.1f%%) | interface %.3f | %.3f (x%.1f, %4.1f%% of dense total)"
              " | leaf cache %5.2f (%2.0f%%) slab factors %5.2f slab discr. %5.2f | peak GPU %5.2f host %5.2f" % (
              p, leaves(n).replace("$\\times$", "x"), N, d["dof"], GB * d["total"], GB * h["total"], 100 * (1 - h["total"] / d["total"]),
              GB * d["iface"], GB * h["iface"], d["iface"] / h["iface"], 100 * d["iface"] / d["total"],
              GB * d["cache"], 100 * d["cache"] / d["total"], GB * d["mumps"], GB * d["slab"], GB * d["gpu_peak"], GB * d["host_peak"]))
    for (p, n, N, s, fl), m in sorted(MEM.items()):
        if not fl:
            c = mem(p, n, N, s)
            print("  no leaf cache p=%d %s N=%d: total %.2f GB vs %.2f with the cache (x%.1f)" % (
                p, leaves(n).replace("$\\times$", "x"), N, GB * m["total"], GB * c["total"], c["total"] / m["total"]))
print("wrote", FIG)
