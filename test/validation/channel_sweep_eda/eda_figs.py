"""fig1_rank_decay, fig2_replicates, fig3_accuracy_cost (PNG, Agg) into
SSLABLU_EDA_OUT. Run eda_load.py and eda_extra.py first."""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eda_common import *
X = pickle.load(open(os.path.join(OUT, "eda_extra.pkl"), "rb"))

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 11, "axes.titlecolor": INK, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def style(a):
    a.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


RANKS = (4, 6, 8, 12)
DISC = {8: 5.0e-05, 12: 8.2e-07, 16: 1.0e-08}


def herr(k, p, N, rank):
    es = []
    for rp in ("1013", "1018"):
        x = (one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, kind="smap", rk=rank) if k == "smap" else
             one(sweep="rankiso", rep=rp, npy=32, p=p, N=N, solver="rbhbs", rk=int(1.5 * p), rbrk=rank))
        es.append(rel(fld(x), fld(rbof(x))))
    return np.array(es)


# ---- 1: compression error vs absolute rank --------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
for i, (k, name) in enumerate((("smap", "S-map compression only (dense rb solver)"),
                               ("solver", "solver compression only (S-maps at RK = 1.5p)"))):
    for j, N in enumerate((8, 16, 32)):
        a = ax[i, j]
        a.axhspan(1e-10, 1e-8, color="#d9d8d2", alpha=0.5, lw=0, zorder=0)
        for c, p in enumerate((8, 12, 16)):
            e = np.array([herr(k, p, N, r) for r in RANKS])            # (ranks, reps)
            gm = np.exp(np.log(e).mean(1))
            a.plot(RANKS, gm, "-", color=COL[c], marker=MK[c], ms=7, lw=2, mec="#fcfcfb", mew=1.2, zorder=3)
            a.axhline(DISC[p], color=COL[c], ls=":", lw=1.2, zorder=2)
        a.set_yscale("log"); a.set_ylim(1e-10, 5e-2); a.set_xticks(RANKS)
        style(a)
        if i == 0:
            a.set_title("N = %d slabs (npan_x = %d)" % (N, 128 // N))
        if j == 0:
            a.set_ylabel(name.split(" (")[0] + "\nrel. L2 error in $\\eta$ vs rb")
        if i == 1:
            a.set_xlabel("absolute rank")
ax[0, 2].text(12, 2.5e-9, "replicate noise floor", ha="right", va="center", fontsize=8.5, color=MUTED)
h = [Line2D([], [], color=COL[c], marker=MK[c], ms=7, lw=2, mec="#fcfcfb", label="p = %d" % p) for c, p in enumerate((8, 12, 16))]
h += [Line2D([], [], color=MUTED, ls=":", lw=1.2, label="discretization error of that p (p = 16 extrapolated)")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False)
fig.suptitle("Compression error decays ~0.5-0.7 decades per unit rank, nearly independent of p; npan_y = 32, t = 100 h",
             fontsize=12, color=INK)
fig.tight_layout(rect=[0, 0.05, 1, 0.96]); fig.savefig(os.path.join(OUT, "fig1_rank_decay.png"), dpi=170); plt.close(fig)

# ---- 2: replicate disagreement vs compression error -------------------------------
E, D = X["E"], X["D"]
fig, a = plt.subplots(figsize=(6.4, 5.2))
a.loglog([1e-10, 1], [1e-10, 1], color=MUTED, lw=1, ls="--", zorder=1)
a.scatter(E, D, s=26, color=COL[0], edgecolor="#fcfcfb", lw=0.8, zorder=3)
a.text(2e-3, 2e-2, "disagreement = error", fontsize=8.5, color=MUTED, ha="right")
a.set_xlabel("compression error  $\\|X - X_{rb}\\| / \\|X_{rb}\\|$  (1013 run)")
a.set_ylabel("replicate disagreement  $\\|X_{1018} - X_{1013}\\| / \\|X_{rb}\\|$")
a.set_title("Below ~1e-8 the two replicates disagree as much as\nthey disagree with rb: compression error is unresolvable there")
a.set_xlim(1e-10, 1); a.set_ylim(1e-14, 1)
style(a); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_replicates.png"), dpi=170); plt.close(fig)

# ---- 3: accuracy vs cost --------------------------------------------------------------
pts = X["pts"]
fig, a = plt.subplots(figsize=(8.2, 5.6))
kinds = (("rb", "dense rb"), ("smap", "S-map compressed, dense solver"), ("solver", "HBS solver, S-maps at 1.5p"),
         ("both", "both at RK = RB_RK"))
for c, (k, lab) in enumerate(kinds):
    sel = [t for t in pts if t[4]["kind"] == k and not (t[4]["p"] == 16 and t[4]["npy"] == 32 and k == "rb")]
    err = [max(t[2] if not (t[4]["p"] == 16 and t[4]["npy"] == 32) else DISC[16], t[3]) for t in sel]
    a.scatter([t[0] for t in sel], err, s=34, marker=MK[c], color=COL[c], edgecolor="#fcfcfb", lw=0.8, zorder=3, label=lab)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("cost: setup + 800 steps, conservative form only [s]  (chunk-clean step cost)")
a.set_ylabel("max(discretization error, compression error) in $\\eta$")
a.set_title("Error is set by (p, npan_y); compression only adds error, never saves time here")
a.legend(frameon=False, loc="lower left")
style(a); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_accuracy_cost.png"), dpi=170); plt.close(fig)
print("wrote fig1_rank_decay.png fig2_replicates.png fig3_accuracy_cost.png")
