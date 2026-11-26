"""Outer residuals |R|_inf of the three variants, CSX manufactured run (kink 0),
Ns = 64, nm 8,2, p 8: red-black preconditioned with the coarse c_ref (baseline),
P = I (nonlinear overlapping Schwarz, same c_ref and start), and the Debye-Hueckel
linearization (c_ref = Lam^2 N, its own red-black, start phi = 0).
Values copied from the run log."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rb = [9.64e-05, 4.56e-06, 3.00e-08, 2.10e-10, 1.66e-12]
schwarz = [9.64e-05, 3.51e-05, 2.08e-05, 1.47e-05, 1.01e-05, 6.87e-06, 5.49e-06, 4.48e-06, 3.66e-06, 3.01e-06,
           2.47e-06, 2.03e-06, 1.67e-06, 1.37e-06, 1.13e-06, 9.33e-07, 7.73e-07, 6.44e-07, 5.37e-07, 4.48e-07,
           3.74e-07, 3.12e-07, 2.63e-07, 2.22e-07, 1.88e-07, 1.59e-07, 1.35e-07, 1.15e-07, 9.85e-08, 8.45e-08,
           7.28e-08, 6.30e-08, 5.46e-08, 4.73e-08, 4.10e-08, 3.55e-08, 3.07e-08, 2.65e-08, 2.29e-08, 1.98e-08,
           1.70e-08, 1.47e-08, 1.27e-08, 1.09e-08, 9.40e-09, 8.09e-09, 6.96e-09, 5.99e-09, 5.15e-09, 4.43e-09,
           3.81e-09, 3.27e-09, 2.81e-09, 2.41e-09, 2.07e-09, 1.78e-09, 1.53e-09, 1.31e-09, 1.12e-09, 9.65e-10,
           8.28e-10, 7.10e-10, 6.09e-10, 5.22e-10, 4.47e-10, 3.84e-10, 3.29e-10, 2.82e-10, 2.41e-10, 2.07e-10,
           1.77e-10, 1.52e-10, 1.30e-10, 1.11e-10, 9.55e-11]
dh = [1.14e+00, 4.37e-01, 2.65e-01, 1.45e-01, 9.80e-02, 6.78e-02, 4.74e-02, 3.33e-02, 2.34e-02, 1.64e-02,
      1.15e-02, 8.09e-03, 5.67e-03, 3.97e-03, 2.78e-03, 1.94e-03, 1.35e-03, 9.43e-04, 6.57e-04, 4.58e-04,
      3.19e-04, 2.22e-04, 1.54e-04, 1.07e-04, 7.45e-05, 5.18e-05, 3.60e-05, 2.50e-05, 1.74e-05, 1.21e-05,
      8.38e-06, 5.82e-06, 4.04e-06, 2.81e-06, 1.95e-06, 1.36e-06, 9.43e-07, 6.55e-07, 4.55e-07, 3.16e-07,
      2.19e-07, 1.52e-07, 1.06e-07, 7.36e-08, 5.11e-08, 3.55e-08, 2.47e-08, 1.71e-08, 1.19e-08, 8.27e-09,
      5.75e-09, 3.99e-09, 2.78e-09, 1.93e-09, 1.34e-09, 9.30e-10, 6.46e-10, 4.49e-10, 3.12e-10, 2.17e-10,
      1.51e-10, 1.05e-10, 7.27e-11]
TOL = 1e-10
C_RB, C_SW, C_DH, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#8a8984"

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogy(range(len(rb)), rb, "-o", color=C_RB, lw=2, ms=6, label="red-black preconditioner (P = RB)")
ax.semilogy(range(len(schwarz)), schwarz, "-s", color=C_SW, lw=2, ms=4, markevery=5,
            label="nonlinear overlapping Schwarz (P = I)")
ax.semilogy(range(len(dh)), dh, "-^", color=C_DH, lw=2, ms=5, markevery=5,
            label="Debye-Hückel linearization (P = RB, start 0)")
ax.axhline(TOL, color=MUTED, lw=1, ls="--")
ax.text(12, TOL * 1.4, "tolerance $10^{-10}$", color=MUTED, ha="left", va="bottom", fontsize=9)
ax.annotate("%d its" % len(rb), (len(rb) - 1, rb[-1]), xytext=(8, 0), textcoords="offset points",
            color=INK, va="center", fontsize=9)
ax.annotate("%d its" % len(schwarz), (len(schwarz) - 1, schwarz[-1]), xytext=(0, -14),
            textcoords="offset points", color=INK, ha="right", fontsize=9)
ax.annotate("%d its" % len(dh), (len(dh) - 1, dh[-1]), xytext=(0, -14),
            textcoords="offset points", color=INK, ha="center", fontsize=9)
ax.set_xlabel("outer iteration $k$")
ax.set_ylabel(r"$\|R(u_k)\|_\infty$")
ax.set_xlim(-1, len(schwarz) + 1)
ax.grid(True, which="major", color="#e4e3df", lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(MUTED)
ax.tick_params(colors=INK)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig("csx_outer_residuals_variants.%s" % ext, dpi=200)
print("saved csx_outer_residuals_variants.png/.pdf")
rate = lambda r: (r[-1] / r[0]) ** (1 / (len(r) - 1))
print("mean rate RB %.2e, Schwarz %.3f, DH %.3f (DH last 10: %.3f)"
      % (rate(rb), rate(schwarz), rate(dh), rate(dh[-11:])))
