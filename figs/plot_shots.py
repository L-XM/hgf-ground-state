"""Finite-shot experiments (shots.csv from data_shots.py).
(a) n=4 transverse-field Ising ring, energy error vs step for S = 1e2..1e6 shots per Pauli term and exact estimates.
(b) plateau (geometric mean over the last quarter of the run, median over starts) vs S, both Hamiltonians.
(c) plateau vs n at S = 1e4, transverse-field Ising ring.
(d) trap circuit of Proposition 1 with exact estimates and with S = 1e5, 1e3."""
import sys, csv, collections
sys.path.insert(0, "concept")
import numpy as np
from pcommon import *

runs = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "shots.csv")):
    runs[(r["expt"], r["H"], int(r["n"]), int(r["S"]), int(r["seed"]))].append((int(r["t"]), float(r["delta"])))
runs = {k: (np.array([a for a, _ in v]), np.array([b for _, b in v])) for k, v in runs.items()}


def median_curve(expt, h, n, S):
    trs = [v for k, v in runs.items() if k[:4] == (expt, h, n, S)]
    L = min(len(t) for t, _ in trs)
    return trs[0][0][:L], np.exp(np.median([np.log(np.clip(d[:L], 1e-300, None)) for _, d in trs], axis=0))


def plateau(expt, h, n, S):
    vals = []
    for k, (t, d) in runs.items():
        if k[:4] != (expt, h, n, S): continue
        m = t >= 0.75 * t[-1]; vals.append(np.exp(np.mean(np.log(d[m]))))
    return np.median(vals) if vals else np.nan


Ss = [100, 1000, 10000, 100000, 1000000]
ramp = [SEQ(x) for x in np.linspace(0.35, 1.0, len(Ss))]
fig = plt.figure(figsize=(7.0, 2.1))
w, y0, hgt = 0.185, 0.2, 0.63
axs = [fig.add_axes([0.065 + i * 0.245, y0, w, hgt]) for i in range(4)]
ax, bx, cx, dx = axs

t, d = median_curve("A", "tfim", 4, 0)
ax.plot(*upto_floor(t, d), color=INK2, lw=LW, ls=(0, (4, 1.2)))
for S, col in zip(Ss, ramp):
    t, d = median_curve("A", "tfim", 4, S)
    ax.plot(t, d, color=col, lw=LW)
    ax.text(t[-1] * 1.02, d[-1], rf"$10^{{{int(np.log10(S))}}}$", color=INK2, fontsize=ANN_FS, va="center")
ax.text(0.97 * t[-1], 3e-12, "exact", color=INK2, fontsize=ANN_FS, ha="right")
logy(ax, 10); ax.set_xlim(0, 4000 * 1.17); ax.set_xticks([0, 2000, 4000])
ax.set_xlabel("step $t$", labelpad=1); ax.set_ylabel(r"$F-E_0$", labelpad=1)
title(ax, r"$n=4$ Ising, shots $S$")

for h, mk in (("heisenberg", "o"), ("tfim", "s")):
    P = np.array([plateau("A", h, 4, S) for S in Ss])
    bx.plot(Ss, P, color=BLUE, lw=LW, marker=mk, ms=3.5, mec="white", mew=0.6)
bx.plot(Ss, 0.8 / np.array(Ss), color=MUTED, lw=LW_REF, ls=(0, (1, 1.5)))
bx.text(2e4, 0.8 / 2e4 * 0.45, r"$\propto 1/S$", color=MUTED, fontsize=ANN_FS, ha="right", va="top")
bx.set_xscale("log"); bx.set_yscale("log"); bx.minorticks_off()
bx.set_xticks(Ss[::2]); bx.set_ylim(1e-6, 1e-1)
bx.grid(axis="y", color=GRID, lw=0.4); bx.set_axisbelow(True)
bx.set_xlabel("shots $S$", labelpad=1); bx.set_ylabel("plateau of $F-E_0$", labelpad=1)
title(bx, r"$n=4$, both models")
from matplotlib.lines import Line2D
bx.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=0, ms=3.5, label="Heisenberg"),
                   Line2D([], [], color=BLUE, marker="s", lw=0, ms=3.5, label="Ising")],
          loc="upper right", fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)

ns = sorted({k[2] for k in runs if k[0] == "B"})
P = np.array([plateau("B", "tfim", n, 10000) for n in ns])
cx.plot(ns, P, color=BLUE, lw=LW, marker="s", ms=3.5, mec="white", mew=0.6)
cx.plot(ns, P[0] * 2.0 ** (np.array(ns) - ns[0]), color=MUTED, lw=LW_REF, ls=(0, (1, 1.5)))
cx.text(ns[-1] - 0.2, P[0] * 2.0 ** (ns[-1] - ns[0]) * 1.5, r"$\propto N$", color=MUTED, fontsize=ANN_FS, ha="right", va="bottom")
cx.set_yscale("log"); cx.minorticks_off(); cx.set_ylim(2e-4, 2e-2)
cx.set_xticks(ns); cx.grid(axis="y", color=GRID, lw=0.4); cx.set_axisbelow(True)
cx.set_xlabel("qubits $n$", labelpad=1); cx.set_ylabel("plateau of $F-E_0$", labelpad=1)
title(cx, r"Ising, $S=10^4$")

for S, col, ls, lab in ((0, ORANGE, "-", "exact"), (100000, DEEP, "-", r"$S=10^5$"), (1000, BLUE, (0, (4, 1.2)), r"$S=10^3$")):
    t, d = median_curve("C", "zz", 4, S)
    dx.plot(*upto_floor(t, d), color=col, lw=LW, ls=ls, label=lab)
logy(dx, 10, 1e-8); dx.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
dx.set_xlim(0, 3000); dx.set_xticks([0, 1500, 3000])
dx.set_xlabel("step $t$", labelpad=1); dx.set_ylabel(r"$F-E_0$", labelpad=1)
dx.legend(loc="lower right", fontsize=6.3, handlelength=2.0, borderaxespad=0.1)
title(dx, "trap circuit, Prop. 1")

for a, s in zip(axs, "abcd"): tag(a, f"({s})", -0.36, 1.17)
save(fig, "fig_shots")
print("plateaus A:", {h: [f"{plateau('A', h, 4, S):.2e}" for S in Ss] for h in ("heisenberg", "tfim")})
print("plateaus B:", dict(zip(ns, [f"{p:.2e}" for p in P])))
