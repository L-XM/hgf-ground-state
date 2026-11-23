"""Draft figure: local Pauli projection (Wiersema–Killoran, output side) vs. this work.
Data: localpauli.csv from data_localpauli.py. Same style as plot_figs.py
(ours = blue, no frame = grey; local Pauli is a new entity and gets red)."""
import sys, csv, collections
sys.path.insert(0, "concept")
import numpy as np
from pcommon import *

D = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in csv.DictReader(open("localpauli.csv")):
    D[(r["H"], r["variant"])][r["start"]][int(r["t"])] = (float(r["delta"]), float(r["g2"]), float(r["localproj2"]))

def arr(h, v, k):
    runs = D[(h, v)]
    ts = sorted(next(iter(runs.values())))
    return np.array(ts), np.array([[runs[s][t][k] for t in ts] for s in sorted(runs)])

LP = "lp_full@0.5/ell"
series = [("ours", BLUE, "-"), (LP, RED, "-"), ("noframe", MUTED, (0, (2, 1.2)))]
titles = {"tfim": "transverse-field Ising", "heisenberg": "Heisenberg ring", "random2local": "random 2-local"}

fig = plt.figure(figsize=(7.0, 2.25))
axes = [fig.add_axes([0.065 + i * 0.238, 0.19, 0.19, 0.68]) for i in range(4)]
for ax, h, lab in zip(axes, titles, "abc"):
    for v, col, ls in series:
        t, a = arr(h, v, 0); band(ax, t, a, col, ls)
    logy(ax); ax.set_xlim(0, 3000); ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xlabel("step $t$", labelpad=1); ax.set_title(titles[h], fontsize=TITLE_FS, color=INK, pad=3)
    tag(ax, f"({lab})", -0.32, 1.14)
axes[0].set_ylabel(r"$F(\psi_t)-E_0$", labelpad=1)
a0 = axes[0]
a0.text(1250, 1e-5, "this work", color=BLUE, fontsize=ANN_FS, ha="left", va="center")
a0.text(1500, 1.2e-2, "local Pauli", color=RED, fontsize=ANN_FS, ha="center", va="top")
a0.text(1500, 2.5, "no frame", color=INK2, fontsize=ANN_FS, ha="center", va="bottom")

# (d) why the local scheme stops: full gradient vs. its local part, transverse-field Ising
d = axes[3]; d.set_position([0.81, 0.19, 0.165, 0.68])
t, g2 = arr("tfim", LP, 1); t, c2 = arr("tfim", LP, 2)
band(d, t, g2, INK2, "-")
for run in c2: d.plot(t, np.clip(run, 1e-14, None), color=RED, lw=0.5, alpha=0.35)
d.plot(t, np.exp(np.median(np.log(np.clip(c2, 1e-14, None)), 0)), color=RED, lw=LW, ls=(0, (4, 1.2)))
logy(d, top=1e3, bottom=1e-14); d.set_xlim(0, 3000)
d.set_xticks([0, 1000, 2000, 3000])
d.set_xlabel("step $t$", labelpad=1); d.set_title("local Pauli, Ising", fontsize=TITLE_FS, color=INK, pad=3)
d.set_ylabel("squared gradient", labelpad=1)
d.text(2950, 3, r"full $|g|^2$", color=INK2, fontsize=ANN_FS, ha="right", va="bottom")
d.text(1100, 1e-11, "local part", color=RED, fontsize=ANN_FS, ha="left", va="center")
tag(d, "(d)", -0.36, 1.14)
save(fig, "fig_localpauli")
for h in titles:
    for v, _, _ in series:
        t, a = arr(h, v, 0); print(h, v, "final median", np.median(a[:, -1]))
