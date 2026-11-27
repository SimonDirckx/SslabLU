"""Outer convergence of the three variants, parsed from a test_periodic_vanishing_edge.py
log (default comparison_iter.txt): left |R(u_k)|_inf, right the interface error
|u_k - u*|_inf against the manufactured solution.  Output name carries p from the
log's command line (csx_outer_variants_p<p>.png/.pdf) unless --out is given.

    python plot_outer_log.py [--log comparison_iter.txt] [--out NAME]
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"coarse": ("red-black preconditioner (P = RB)", "#2a78d6", "o", 6, 1),
         "schwarz": ("nonlinear overlapping Schwarz (P = I)", "#eb6834", "s", 4, 5),
         "dh": ("Debye-Hückel linearization (P = RB, start 0)", "#1baf7a", "^", 5, 5)}
INK, MUTED = "#1f1f1e", "#8a8984"


def parse(path):
    runs, cur, p = {}, None, None
    for line in open(path):
        m = re.search(r"--p (\d+)", line)
        if m and p is None:
            p = int(m.group(1))
        m = re.match(r"\s*\[(\w+)\] outer iteration", line)
        if m:
            cur = runs.setdefault(m.group(1), dict(R=[], err=[]))
            continue
        m = re.match(r"\s*outer\s+\d+\s+\|R\| (\S+)(?:\s+err_iface (\S+))?", line)
        if m and cur is not None:
            cur["R"].append(float(m.group(1)))
            if m.group(2):
                cur["err"].append(float(m.group(2)))
    return runs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="comparison_iter.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()
    runs, p = parse(args.log)
    out = args.out or "csx_outer_variants_p%s" % p
    kmax = max(len(r["R"]) for r in runs.values())

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for key, (lab, col, mk, ms, every) in STYLE.items():
        if key not in runs:
            continue
        r = runs[key]
        axs[0].semilogy(r["R"], "-" + mk, color=col, lw=2, ms=ms, markevery=every, label=lab)
        k = len(r["R"]) - 1
        off = (8, 0) if key == "coarse" else (0, -14)
        axs[0].annotate("%d its" % len(r["R"]), (k, r["R"][-1]), xytext=off, textcoords="offset points",
                        color=INK, va="center" if key == "coarse" else "baseline",
                        ha="left" if key == "coarse" else "center", fontsize=9)
        if r["err"]:
            axs[1].semilogy(r["err"], "-" + mk, color=col, lw=2, ms=ms, markevery=every, label=lab)
    axs[0].axhline(args.tol, color=MUTED, lw=1, ls="--")
    axs[0].text(0.68 * kmax, args.tol * 1.4, "tolerance $10^{%d}$" % round(__import__("math").log10(args.tol)),
                color=MUTED, ha="left", va="bottom", fontsize=9)
    errs = [r["err"][-1] for r in runs.values() if r["err"]]
    if errs:
        axs[1].axhline(errs[0], color=MUTED, lw=1, ls="--")
        axs[1].text(0.6 * kmax, errs[0] * 1.5, "discretization error %.2e" % errs[0],
                    color=MUTED, ha="left", va="bottom", fontsize=9)
    axs[0].set_ylabel(r"$\|R(u_k)\|_\infty$")
    axs[1].set_ylabel(r"$\|u_k - u^*\|_\infty$ (interface)")
    for a in axs:
        a.set_xlabel("outer iteration $k$")
        a.set_xlim(-1, kmax + 1)
        a.grid(True, which="major", color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(MUTED)
        a.tick_params(colors=INK)
    axs[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (out, ext), dpi=200)
    print("saved %s.png/.pdf" % out)
    for key, r in runs.items():
        R = r["R"]
        rate = (R[-1] / R[0]) ** (1 / (len(R) - 1))
        tail = R[-11:] if len(R) > 11 else R
        print("%-8s %3d its  mean rate %.3e  last-10 rate %.3e" % (key, len(R), rate,
              (tail[-1] / tail[0]) ** (1 / (len(tail) - 1))))


if __name__ == "__main__":
    main()
