"""Scaling with n: (a) median steps to F-E0<1e-8, (b) observed local rate / Theorem 3 rate.
Reads scaling.csv (+ scaling_n10.csv); only finished runs count, groups with <5 finished runs are hollow."""
import sys, csv, collections
sys.path.insert(0, "concept")
import numpy as np
from pcommon import *
from data_scaling import hamiltonian

files = sys.argv[1:] or ["scaling.csv", "scaling_n10.csv"]
runs = collections.defaultdict(list)
for fn in files:
    for r in csv.DictReader(open(fn)):
        runs[(r["H"], int(r["n"]), r["variant"], int(r["seed"]))].append((int(r["t"]), float(r["delta"])))

res = collections.defaultdict(lambda: ([], []))
for (h, n, v, s), tr in runs.items():
    t = np.array([a for a, _ in tr]); d = np.array([b for _, b in tr])
    if d[-1] >= 1e-8: continue
    res[(h, n, v)][0].append((t, d))

spec = {}
for h, n in {(k[0], k[1]) for k in res}:
    E = np.linalg.eigvalsh(hamiltonian(h, n)); E0 = E[0]; D = E[E > E0 + 1e-9][0] - E0; ell = 2 * (E[-1] - E0)
    eta = 1 / ell; rho = 2 * (eta / 2) * (3 / (2 ** n - 1)) * D
    spec[(h, n)] = (D, -np.log(1 - rho))

pts = collections.defaultdict(list)
for (h, n, v), (trs, _) in sorted(res.items()):
    D, th = spec[(h, n)]
    steps, rates = [], []
    for t, d in trs:
        steps.append(t[-1])
        m = (d < D / 2) & (d > 1e-8)
        if m.sum() > 5: rates.append(-np.polyfit(t[m], np.log(d[m]), 1)[0])
    pts[(h, v)].append((n, np.median(steps), np.median(rates) / th, len(trs)))

fig = plt.figure(figsize=(7.0, 2.4))
ax = fig.add_axes([0.075, 0.17, 0.39, 0.72]); bx = fig.add_axes([0.585, 0.17, 0.39, 0.72])
sty = {"iid": (BLUE, "-"), "acc": (DEEP, (0, (4, 1.2)))}
mk = {"heisenberg": "o", "tfim": "s"}
for a in (ax, bx):
    a.grid(axis="y", color=GRID, lw=0.4); a.set_axisbelow(True)
    a.set_xticks(range(3, 11)); a.set_xlim(2.6, 10.4); a.set_xlabel("qubits $n$", labelpad=1)
for (h, v), P in pts.items():
    col, ls = sty[v]; P = np.array(P)
    for a, y in ((ax, P[:, 1]), (bx, P[:, 2])):
        a.plot(P[:, 0], y, color=col, ls=ls, lw=1.0, zorder=3)
        for x, yy, k in zip(P[:, 0], y, P[:, 3]):
            a.scatter(x, yy, marker=mk[h], s=14, zorder=4, lw=0.8,
                      color=col if k >= 5 else "white", ec=col if k < 5 else "white")
n = np.arange(3, 11)
ax.plot(n, 30 * 2.0 ** n, color=MUTED, lw=0.7, ls=(0, (1, 1.5)))
ax.text(9.6, 30 * 2 ** 9.6 * 1.6, r"$\propto N$", color=MUTED, fontsize=7, ha="right", va="bottom")
ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylabel(r"steps to $F-E_0<10^{-8}$", labelpad=1)
bx.axhline(1, color=MUTED, lw=0.7, ls=(0, (1, 1.5)))
bx.text(10.3, 1.08, "Theorem 3", color=MUTED, fontsize=7, ha="right", va="bottom")
bx.set_ylim(0, 5); bx.set_ylabel("observed rate / guaranteed rate", labelpad=1)
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=BLUE, lw=1.0, label="i.i.d."), Line2D([], [], color=DEEP, lw=1.0, ls=(0, (4, 1.2)), label="single-CNOT"),
      Line2D([], [], color=INK2, marker="o", lw=0, ms=4, label="Heisenberg ring"),
      Line2D([], [], color=INK2, marker="s", lw=0, ms=4, label="transverse-field Ising")]
bx.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=6.5, handlelength=2.2,
          columnspacing=1.0, borderaxespad=0.2)
tag(ax, "(a)", -0.17, 1.12); tag(bx, "(b)", -0.17, 1.12)
save(fig, "fig_scaling")
for (h, v), P in sorted(pts.items()):
    print(h, v, [(int(a), int(b), round(c, 2), int(k)) for a, b, c, k in P])
