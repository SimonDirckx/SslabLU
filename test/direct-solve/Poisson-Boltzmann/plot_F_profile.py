"""
plot_F_profile: the flux function F(psi) of the manufactured solution
(pb.manufactured.PBPedestal, tanh profile) and its two parts,

    F(psi) = A [ -(1 - psi) + k H(psi) ],   H = (1 + tanh((psi - psi0) / delta)) / 2.

    python plot_F_profile.py [--A 3 --k 0.5 --psi0 0.7 --delta 0.5] [--out F_profile.png]
"""
import argparse

import numpy as np

from plot_fields import _tex_rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=float, default=3.0)
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--psi0", type=float, default=0.7)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--out", default="F_profile.png")
    args = ap.parse_args()
    A, k, p0, dl = args.A, args.k, args.psi0, args.delta

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psi = np.linspace(0.0, 1.0, 401)
    H = 0.5 * (1.0 + np.tanh((psi - p0) / dl))
    well, step = -A * (1.0 - psi), A * k * H
    F = well + step

    with plt.rc_context(_tex_rc()):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.axvline(p0, color="0.6", lw=0.8, ls=":")
        ax.plot(psi, well, color="0.45", lw=1.4, ls="--", label=r"$-A(1-\psi)$")
        ax.plot(psi, step, color="0.45", lw=1.4, ls="-.", label=r"$A k\, H(\psi)$")
        ax.plot(psi, F, color="C0", lw=2.2, label=r"$F(\psi)$")
        ax.annotate(r"$\psi_0$", (p0, ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\psi$")
        ax.legend(frameon=False, loc="upper left")
        fig.savefig(args.out, dpi=200)
        plt.close(fig)
    print("saved", args.out, "| F(0) = %.3f, F(psi0) = %.3f, F(1) = %.3f" % (F[0], np.interp(p0, psi, F), F[-1]))


if __name__ == "__main__":
    main()
