"""Fiber figure (Theorem 5, SM S6), real algorithm, n=3, data from data_fiber.py (fiber.npz).
One excited level lambda_1 (k=1), one completion W, one delta_s, one i.i.d. frame sequence; the fiber is
E^u = span_C{v_0} = R^2, coordinate z.
(a) Steps until the energy drops below lambda_1 - gap/4, over the fiber: a single peak, the bad point z*.
(b) |delta_u,t| against |delta_s,t| for starts at distance 10^-1..10^-7 (x 0.05) from z*: the stable part
    contracts for all of them, the unstable part comes back unless z = z*.
(c) Escape time against log distance from z*: slope = 1/(top Lyapunov exponent), from lyapunov.csv (k=1)."""
import csv
import numpy as np
from pcommon import *
from matplotlib.colors import LogNorm

d = np.load("fiber.npz"); esc, xs, zs, por, et, dists = d["esc"], d["xs"], complex(d["zs"]), d["por"], d["et"], d["dists"]
lyap = max(float(r["exponent"]) for r in csv.DictReader(open("../lyapunov.csv")) if int(r["k"]) == 1)
r0 = dists*0.05                                   # |z - z*| of the portrait starts
etg = et[1:].reshape(len(dists), -1)

fig = plt.figure(figsize=(7.0, 2.35))
# (a) escape-time map on the fiber
a = fig.add_axes([0.075, 0.17, 0.225, 0.70]); cb = fig.add_axes([0.306, 0.17, 0.008, 0.70])
s = 100                                           # axis unit 10^-2
ext = [xs[0]*s, xs[-1]*s, xs[0]*s, xs[-1]*s]
im = a.imshow(esc, origin="lower", extent=ext, cmap=SEQ, norm=LogNorm(esc.min(), esc.max()), interpolation="bilinear")
a.contour(xs*s, xs*s, esc, levels=[60, 80, 110, 140], colors="white", linewidths=0.4, alpha=0.8)
a.scatter([zs.real*s], [zs.imag*s], s=26, marker="X", color=ORANGE, ec="white", lw=0.6, zorder=5)
a.annotate(r"$z^*$", (zs.real*s, zs.imag*s), (zs.real*s + 2.4, zs.imag*s + 3.2), fontsize=7.5, color=INK,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
a.set_xlabel(r"$\mathrm{Re}\,z$ $(10^{-2})$", labelpad=1); a.set_ylabel(r"$\mathrm{Im}\,z$ $(10^{-2})$", labelpad=0)
a.set_xticks([-8, -4, 0, 4, 8]); a.set_yticks([-8, -4, 0, 4, 8]); a.tick_params(length=2, pad=1)
for sp in a.spines.values(): sp.set_visible(False)
title(a, r"one fiber $\delta_s+E^u$: one bad point")
plt.colorbar(im, cax=cb, ticks=[50, 100, 150]); cb.set_yticklabels(["50", "100", "150"])
cb.tick_params(labelsize=6, length=1.5, pad=1); cb.minorticks_off()
cb.set_ylabel("steps to leave", fontsize=6.5, labelpad=1)
for sp in cb.spines.values(): sp.set_visible(False)
tag(a, "(a)", -0.30, 1.13)

# (b) phase portrait in tube coordinates
b = fig.add_axes([0.43, 0.17, 0.235, 0.70])
cols = SEQ(np.linspace(0.35, 1.0, len(dists)))
for i in range(len(dists)):
    for j in range(etg.shape[1]):
        P = por[1 + i*etg.shape[1] + j]
        b.plot(P[:, 0], P[:, 1], color=cols[i], lw=0.7, alpha=0.9)
b.plot(por[0, :470, 0], por[0, :470, 1], color=ORANGE, lw=LW, zorder=5)
b.scatter([por[0, 0, 0]], [por[0, 0, 1]], s=14, color=INK, ec="white", lw=0.5, zorder=6)
b.set_xscale("log"); b.set_yscale("log"); b.minorticks_off()
b.set_xlim(1e-10, 0.12); b.set_ylim(1e-10, 3)
b.set_xticks([1e-9, 1e-6, 1e-3]); b.set_yticks([1e-9, 1e-6, 1e-3, 1])
b.set_xlabel(r"stable part $|\delta_{s,t}|$", labelpad=1); b.set_ylabel(r"unstable part $|\delta_{u,t}|$", labelpad=1)
b.grid(color=GRID, lw=0.4); b.set_axisbelow(True)
b.text(2e-6, 2e-9, r"$z=z^*$: into" + "\n" + "the saddle", color=ORANGE_TXT, fontsize=ANN_FS, ha="left", va="center")
b.text(2e-9, 0.25, r"$z\neq z^*$: pushed out along $E^u$", color=DEEP, fontsize=ANN_FS, ha="left", va="center")
b.text(0.035, 0.012, "start", fontsize=ANN_FS, color=INK, ha="right", va="center")
title(b, r"trajectories in tube coordinates")
tag(b, "(b)", -0.27, 1.13)

# (c) escape time vs distance to z*
c = fig.add_axes([0.765, 0.17, 0.225, 0.70])
c.grid(color=GRID, lw=0.4); c.set_axisbelow(True)
rr = np.logspace(-9.3, -2, 50)
c.plot(rr, etg[0].mean() + np.log(r0[0]/rr)/lyap, color=INK, lw=0.8, ls=(0, (1, 1.2)))
for i in range(len(dists)):
    c.scatter(np.full(etg.shape[1], r0[i]), etg[i], s=14, color=cols[i], ec="white", lw=0.5, zorder=3)
c.axhline(et[0], color=ORANGE, lw=LW)
c.set_xscale("log"); c.minorticks_off(); c.set_xlim(3e-10, 1e-2); c.set_ylim(0, 1100)
c.set_xticks([1e-9, 1e-6, 1e-3]); c.set_yticks([0, 250, 500, 750, 1000])
c.set_xlabel(r"distance $|z-z^*|$ at $t=0$", labelpad=1); c.set_ylabel("steps to leave", labelpad=1)
c.text(4e-10, et[0] - 30, r"$z^*$ itself (floating point)", color=ORANGE_TXT, fontsize=ANN_FS, ha="left", va="top")
c.text(1.3e-6, 520, r"slope $1/\lambda_{\rm Lyap}$" + "\n" + rf"$\lambda_{{\rm Lyap}}={lyap:.3f}$", color=INK,
       fontsize=ANN_FS, ha="left", va="center")
title(c, r"escape time $\propto\log(1/|z-z^*|)$")
tag(c, "(c)", -0.25, 1.13)
save(fig, "fig_fiber")
slope = np.polyfit(np.log(r0), etg.mean(1), 1)[0]
print("fit steps per e-fold", -slope, "=> rate", -1/slope, "lyap", lyap)
