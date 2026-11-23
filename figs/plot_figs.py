"""Paper Figs. 2 and 3, same style as concept/ (Latin Modern, validated palette: ours = blue, single-CNOT = deep blue,
Pauli = aqua, trap = orange, no frame = grey; diverging blue <-> red with grey midpoint for signed exponents).
The previous version is plot_figs_v1.py."""
import sys, re, csv, collections
sys.path.insert(0, "concept")
import numpy as np
from pcommon import *
from matplotlib.colors import TwoSlopeNorm

def load(fn):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(l for l in open(fn) if not l.startswith("#")):
        d[r["variant"]][int(r["seed"])].append(float(r["delta"]))
    return {v: np.array([s[k] for k in sorted(s)]) for v, s in d.items()}
hdr = open("convergence.csv").readline(); rho = float(re.search(r"rho_TheoremB=([0-9.]+)", hdr).group(1)); Delta = 1.0
D = load("convergence.csv"); D.update(load("noframe.csv"))

# ---------------- Fig. 2 (one column) ----------------
fig = plt.figure(figsize=(3.4, 3.0))
ax = fig.add_axes([0.20, 0.40, 0.77, 0.56]); bx = fig.add_axes([0.20, 0.10, 0.77, 0.17])
series = [("iid", BLUE, "-"), ("accumulated", DEEP, (0, (4, 1.2))), ("pauli", AQUA, "-"), ("noframe", MUTED, (0, (2, 1.2)))]
t = np.arange(D["iid"].shape[1])
for key, col, ls in series:
    band(ax, t, D[key], col, ls)
t0 = max(int(np.argmax(s <= Delta/2)) for s in D["iid"]); tt = np.arange(t0, 760)
ax.plot(tt, Delta/2*(1-rho)**(tt-t0), color=INK, ls=(0, (1, 1.2)), lw=LW_REF)
logy(ax, top=20); ax.set_xlim(0, 2500)
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
ax.set_xlabel("step $t$", labelpad=1); ax.set_ylabel(r"$F(\psi_t)-E_0$", labelpad=1)
for x, y, txt, col, ha in ((430, 3e-11, "single-CNOT", DEEP, "left"),
                           (1240, 2e-8, "one Pauli", AQUA_TXT, "left"), (2480, 0.03, "no frame", INK2, "right"),
                           (600, 3e-9, "Theorem 3", INK, "left")):
    ax.text(x, y, txt, color=col, fontsize=ANN_FS, ha=ha, va="center")
# "independent" runs along its curve, in the gap to the single-CNOT curve
med = np.exp(np.median(np.log(np.clip(D["iid"], FLOOR, None)), 0)); i0, i1 = np.argmax(med < 1e-4), np.argmax(med < 1e-9)
fig.canvas.draw(); p0, p1 = ax.transData.transform([(i0, med[i0]), (i1, med[i1])])
ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
ax.text((i0 + i1) / 2 - 14, np.sqrt(med[i0] * med[i1]), "independent", color=BLUE, fontsize=ANN_FS, rotation=ang,
        rotation_mode="anchor", ha="center", va="top")
tag(ax, "(a)", -0.25, 1.04)
# (b) fitted decay rate per run
rates = {}
for key in ("iid", "accumulated", "pauli"):
    rates[key] = []
    for a in D[key]:
        i = np.where((a < 1e-1) & (a > 1e-10))[0]; rates[key].append(-np.polyfit(i, np.log(a[i]), 1)[0])
bound = -np.log(1-rho)
bx.axvline(bound, color=INK, lw=LW_REF, ls=(0, (1, 1.2)))
for j, (key, col) in enumerate((("iid", BLUE), ("accumulated", DEEP), ("pauli", AQUA))):
    r = np.array(rates[key]); bx.scatter(r, np.full(len(r), j), s=13, color=col, ec="white", lw=0.5, zorder=3)
    bx.plot([r.min(), r.max()], [j, j], color=col, lw=0.6, zorder=2)
bx.set_yticks(range(3)); bx.set_yticklabels(["independent", "single-CNOT", "one Pauli"], fontsize=ANN_FS)
bx.set_ylim(2.6, -0.6); bx.set_xlim(0, 0.16); bx.tick_params(axis="y", length=0)
bx.spines["left"].set_visible(False); bx.grid(axis="x", color=GRID, lw=0.4); bx.set_axisbelow(True)
bx.set_xlabel("decay rate per step (fit, six runs)", labelpad=1)
bx.text(bound + 0.002, 2.45, "Thm. 3", fontsize=ANN_FS, color=INK, va="center")
tag(bx, "(b)", -0.25, 1.30)
save(fig, "fig2_convergence")
print("rates", {k: (round(min(v), 4), round(max(v), 4)) for k, v in rates.items()}, "bound", bound)

# ---------------- Fig. 3 (full width) ----------------
L = collections.defaultdict(list)
for r in csv.DictReader(open("lyapunov.csv")): L[int(r["k"])].append(float(r["exponent"]))
TR = collections.defaultdict(list)
for r in csv.DictReader(open("trap.csv")): TR[r["variant"]].append(float(r["energy_minus_E0"]))
K = sorted(L); M = np.array([sorted(L[k], reverse=True) for k in K])
DIV = LinearSegmentedColormap.from_list("div", ["#a8302f", "#e34948", "#f3b0ae", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
fig = plt.figure(figsize=(7.0, 2.35))
a1 = fig.add_axes([0.065, 0.17, 0.37, 0.70]); cb = fig.add_axes([0.445, 0.17, 0.009, 0.70])
im = a1.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, M.min(), M.max()), aspect="auto", interpolation="nearest")
a1.set_xticks(np.arange(-0.5, M.shape[1]), minor=True); a1.set_yticks(np.arange(-0.5, len(K)), minor=True)
a1.grid(which="minor", color="white", lw=0.8); a1.tick_params(which="minor", length=0)
for sp in a1.spines.values(): sp.set_visible(False)
a1.set_xticks(range(0, 14, 2)); a1.set_xticklabels([str(i+1) for i in range(0, 14, 2)])
a1.set_yticks(range(len(K))); a1.set_yticklabels([str(k) for k in K]); a1.tick_params(length=0, pad=2)
a1.set_xlabel("exponent index $i$ (sorted)", labelpad=1); a1.set_ylabel(r"saddle index $k$", labelpad=1)
# staircase: boundary after the first 2k columns
xs, ys = [], []
for r, k in enumerate(K): xs += [2*k-0.5, 2*k-0.5]; ys += [r-0.5, r+0.5]
a1.plot(xs, ys, color=INK, lw=1.1, solid_joinstyle="miter", clip_on=False)
for r, k in enumerate(K):
    if 2*k < 14: a1.text(2*k-0.5+0.25, r, f"{2*k}", fontsize=6, color=INK, va="center", ha="left")
title(a1, r"Lyapunov exponents at each excited level: exactly $2k$ positive")
plt.colorbar(im, cax=cb, ticks=[-0.5, -0.25, 0, 0.25]); cb.tick_params(labelsize=6, length=1.5, pad=1)
for sp in cb.spines.values(): sp.set_visible(False)
cb.set_ylabel(r"exponent $\lambda_i$", fontsize=6.5, labelpad=1)
tag(a1, "(a)", -0.13, 1.13)

a2 = fig.add_axes([0.6, 0.17, 0.385, 0.70])
tt = np.arange(len(TR["trap"]))
for key, col, ls in (("haar_completion", BLUE, "-"), ("perturbed_state", VIOLET, (0, (4, 1.2))), ("trap", ORANGE, "-")):
    a2.plot(*upto_floor(tt, np.array(TR[key])), color=col, ls=ls, lw=LW, solid_capstyle="round")
logy(a2, top=60); a2.set_xlim(0, 600)
a2.set_xlabel("step $t$", labelpad=1); a2.set_ylabel(r"$F(\psi_t)-E_0$", labelpad=1)
a2.text(595, 9, "trap circuit (Prop. 1): stays at an excited level", color=ORANGE_TXT, fontsize=ANN_FS, ha="right", va="bottom")
a2.text(118, 1e-9, "same state,\nHaar completion", color=BLUE, fontsize=ANN_FS, ha="right", va="center")
a2.text(258, 1e-4, r"trap $+\,10^{-6}$ ground" + "\ncomponent", color=VIOLET, fontsize=ANN_FS, ha="left", va="center")
title(a2, r"$H=Z_1Z_2+Z_2Z_3+Z_3Z_4$: a trap and two perturbations")
tag(a2, "(b)", -0.13, 1.13)
save(fig, "fig3_saddles")
