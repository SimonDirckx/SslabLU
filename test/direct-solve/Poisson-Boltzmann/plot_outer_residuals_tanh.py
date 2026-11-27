"""Outer residuals |R|_inf of the three variants, CSX manufactured run with the tanh
pedestal (delta = 0.5), Ns = 64, nz 2, nm 8,2, p 8, Lam = sqrt(70) (minor-radius units):
red-black preconditioned with the coarse c_ref (baseline), P = I (nonlinear
overlapping Schwarz, same c_ref and start), and the Debye-Hueckel linearization
(c_ref = Lam^2 N, its own red-black, start phi = 0).  Values copied from the run log
(comparison_iter.txt).  All three end at err iface 1.97e-05, volume 2.59e-05."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rb = [6.78e-03, 3.95e-04, 1.81e-06, 1.51e-08, 1.27e-10, 8.30e-13]
schwarz = [6.78e-03, 2.62e-03, 1.71e-03, 1.35e-03, 1.08e-03, 8.77e-04, 7.12e-04, 5.80e-04, 4.76e-04, 3.93e-04,
           3.28e-04, 2.76e-04, 2.34e-04, 2.01e-04, 1.73e-04, 1.49e-04, 1.28e-04, 1.10e-04, 9.46e-05, 8.11e-05,
           6.95e-05, 5.95e-05, 5.09e-05, 4.35e-05, 3.71e-05, 3.17e-05, 2.70e-05, 2.30e-05, 1.96e-05, 1.67e-05,
           1.42e-05, 1.21e-05, 1.03e-05, 8.72e-06, 7.41e-06, 6.29e-06, 5.34e-06, 4.54e-06, 3.85e-06, 3.27e-06,
           2.77e-06, 2.35e-06, 2.00e-06, 1.69e-06, 1.44e-06, 1.22e-06, 1.03e-06, 8.75e-07, 7.42e-07, 6.29e-07,
           5.33e-07, 4.51e-07, 3.83e-07, 3.24e-07, 2.75e-07, 2.33e-07, 1.97e-07, 1.67e-07, 1.41e-07, 1.20e-07,
           1.01e-07, 8.59e-08, 7.28e-08, 6.16e-08, 5.22e-08, 4.42e-08, 3.74e-08, 3.17e-08, 2.68e-08, 2.27e-08,
           1.92e-08, 1.63e-08, 1.38e-08, 1.17e-08, 9.89e-09, 8.37e-09, 7.09e-09, 6.00e-09, 5.08e-09, 4.30e-09,
           3.64e-09, 3.08e-09, 2.61e-09, 2.21e-09, 1.87e-09, 1.58e-09, 1.34e-09, 1.13e-09, 9.61e-10, 8.13e-10,
           6.88e-10, 5.83e-10, 4.93e-10, 4.18e-10, 3.54e-10, 2.99e-10, 2.53e-10, 2.14e-10, 1.82e-10, 1.54e-10,
           1.30e-10, 1.10e-10, 9.32e-11]
dh = [1.15e+00, 4.29e-01, 2.43e-01, 1.36e-01, 8.95e-02, 6.04e-02, 4.10e-02, 2.79e-02, 1.90e-02, 1.29e-02,
      8.80e-03, 5.99e-03, 4.07e-03, 2.76e-03, 1.87e-03, 1.27e-03, 8.57e-04, 5.79e-04, 3.91e-04, 2.64e-04,
      1.78e-04, 1.20e-04, 8.12e-05, 5.48e-05, 3.70e-05, 2.49e-05, 1.68e-05, 1.13e-05, 7.65e-06, 5.16e-06,
      3.48e-06, 2.35e-06, 1.58e-06, 1.07e-06, 7.19e-07, 4.85e-07, 3.27e-07, 2.20e-07, 1.49e-07, 1.00e-07,
      6.76e-08, 4.55e-08, 3.07e-08, 2.07e-08, 1.40e-08, 9.41e-09, 6.35e-09, 4.28e-09, 2.88e-09, 1.94e-09,
      1.31e-09, 8.84e-10, 5.96e-10, 4.02e-10, 2.71e-10, 1.83e-10, 1.23e-10, 8.30e-11]
assert len(rb) == 6 and len(schwarz) == 103 and len(dh) == 58
TOL = 1e-10
C_RB, C_SW, C_DH, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#8a8984"
OUT = "csx_outer_residuals_variants_tanh"

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogy(range(len(rb)), rb, "-o", color=C_RB, lw=2, ms=6, label="red-black preconditioner (P = RB)")
ax.semilogy(range(len(schwarz)), schwarz, "-s", color=C_SW, lw=2, ms=4, markevery=5,
            label="nonlinear overlapping Schwarz (P = I)")
ax.semilogy(range(len(dh)), dh, "-^", color=C_DH, lw=2, ms=5, markevery=5,
            label="Debye-Hückel linearization (P = RB, start 0)")
ax.axhline(TOL, color=MUTED, lw=1, ls="--")
ax.text(70, TOL * 1.4, "tolerance $10^{-10}$", color=MUTED, ha="left", va="bottom", fontsize=9)
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
    fig.savefig("%s.%s" % (OUT, ext), dpi=200)
print("saved %s.png/.pdf" % OUT)
rate = lambda r: (r[-1] / r[0]) ** (1 / (len(r) - 1))
print("mean rate RB %.2e, Schwarz %.3f (last 10: %.3f), DH %.3f (last 10: %.3f)"
      % (rate(rb), rate(schwarz), rate(schwarz[-11:]), rate(dh), rate(dh[-11:])))
