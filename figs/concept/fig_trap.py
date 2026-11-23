"""Concept figure 2 (full width): a saddle is a point of the landscape, a trap is a property of the circuit.
(a) Exact gradient flow near the saddle of the toy sphere: only one great circle flows in.
(b) Trap: state and hidden column both in span{saddle, max}; the algorithm never leaves that circle.
(c) Same output state, hidden column tilted by 0.3 rad: the trap is gone.
(d) n=4, H = sum Z_i Z_{i+1} (data of Fig. 3(b)): |<x|U_t|y>|^2 for the trap circuit and for the same state with a
    Haar completion, at t=0 and t=600. Rows = basis states sorted by energy (H is diagonal), columns = strings y."""
import numpy as np
from pcommon import *
import toy, alg

# ---------------- n = 4 trap, same construction as figs/data_extra.py (c) ----------------
n = 4; N, op, perm, rand_GL, step = alg.tools(n)
H = sum(op(alg.Z, i) @ op(alg.Z, i+1) for i in range(n-1)); Ed = np.real(np.diag(H)); E0 = Ed.min()
rng = np.random.default_rng(9)
a = 0.7; Utrap = np.eye(N, dtype=complex); Utrap[0, 0] = Utrap[1, 1] = np.cos(a); Utrap[1, 0] = np.sin(a); Utrap[0, 1] = -np.sin(a)
W = np.linalg.qr(rng.normal(size=(N, N))+1j*rng.normal(size=(N, N)))[0]; W[:, 0] = Utrap[:, 0]; Uhaar = np.linalg.qr(W)[0]
Uhaar[:, 0] = Utrap[:, 0]
T = 600; snaps = {}
for name, U in (("trap", Utrap), ("haar", Uhaar)):
    snaps[name, 0] = np.abs(U)**2
    for t in range(T): U = step(U, H, perm(rand_GL(rng)), rng.integers(n-1), 1/8)
    snaps[name, T] = np.abs(U)**2
    print(name, "F-E0 at T:", np.real(np.vdot(U[:, 0], H @ U[:, 0])) - E0)
order = np.argsort(Ed, kind="stable"); levels = Ed[order]

# ---------------- figure ----------------
fig = plt.figure(figsize=(7.0, 2.6))
R = view((0.62, 0.50, 0.62), up=(0, 0, -1)); LAM = toy.LAM; energy = lambda P: (P**2*LAM).sum(-1)
ex, ey, ez = np.eye(3)
tt = np.linspace(0, 2*np.pi, 400); circle = np.stack([np.cos(tt), np.sin(tt), 0*tt], 1)
psiT = np.array([0.45, 0.89, 0.0]); psiT /= np.linalg.norm(psiT); tang = np.array([-psiT[1], psiT[0], 0.0])
W_ = 0.205
def sphere(i):
    ax = fig.add_axes([i*W_, 0.20, W_, 0.74]); S = Sphere(ax, R, energy, 0, LAM.max(), levels=8)
    S.curve(circle, color=ORANGE, lw=0.7, ls=(0, (3, 1.6)), halo=False, z=3)
    return ax, S
def marks(S):
    S.dot(ex, ORANGE, 24, z=9); S.dot(ez, BLUE, 24, z=9); S.dot(ey, INK2, 16, marker="^", z=9)

# (a) exact flow
ax, S = sphere(0)
for eps, al in ((0.32, 0.45), (0.12, 0.65), (0.03, 0.85), (0.004, 1.0)):
    p = psiT + eps*ez; p /= np.linalg.norm(p)
    P, _ = toy.run(toy.frame_from(p, tang), "full", 900, eta=0.02)
    S.curve(P, color=BLUE, lw=0.9, alpha=al); S.arrow_at(P, 60, color=BLUE)
P, _ = toy.run(toy.frame_from(psiT, tang), "full", 900, eta=0.02)
S.curve(P, color=ORANGE, lw=1.3); S.arrow_at(P, 40, color=ORANGE)
marks(S); S.dot(psiT, INK, 10)
S.label(ex, "saddle", 0.18, 0.22); S.label(ez, "ground", 0.28, -0.10)
ax.text(0, -1.13, "exact flow: one curve\nflows into the saddle", ha="center", va="top", fontsize=ANN_FS); tag(ax, "(a)", 0.0, 1.06)

# (b) trap
ax, S = sphere(1)
U = toy.frame_from(psiT, tang); P, _ = toy.run(U, "column", 120, np.random.default_rng(2))
S.curve(densify(P, 12), color=ORANGE, lw=1.6)
S.tangent(psiT, U[:, 1], color=INK, L=0.30); S.tangent(psiT, -U[:, 1], color=INK, L=0.30)
S.tangent(psiT, U[:, 2], color=BLUE, L=0.34)
marks(S); S.dot(psiT, INK, 10)
S.label(np.cos(0.34)*psiT + np.sin(0.34)*U[:, 2], "column toward\nground: slope 0", -0.10, -0.30, ha="center", fs=6.5)
S.label(ex, f"stops at $F={toy.F(P[-1]):.2f}$", -0.10, 0.27, ha="center", fs=6.5)
ax.text(0, -1.13, "trap circuit: it never\nleaves the circle", ha="center", va="top", fontsize=ANN_FS); tag(ax, "(b)", 0.0, 1.06)

# (c) tilted completion
ax, S = sphere(2)
al = 0.3; U = toy.frame_from(psiT, np.cos(al)*tang + np.sin(al)*ez)
P, _ = toy.run(U, "column", 120, np.random.default_rng(0))
S.curve(densify(P, 12), color=BLUE, lw=1.6)
S.tangent(psiT, U[:, 1], color=INK, L=0.30); S.tangent(psiT, -U[:, 1], color=INK, L=0.30)
S.tangent(psiT, U[:, 2], color=BLUE, L=0.34)
marks(S); S.dot(psiT, INK, 10)
k = next(t for t in range(len(P)) if toy.F(P[t]) < 1e-2)
S.label(ez, f"ground\nin {k} steps", -0.42, -0.06, ha="right", fs=6.5)
ax.text(0, -1.13, "same state, column\ntilted by 0.3 rad", ha="center", va="top", fontsize=ANN_FS); tag(ax, "(c)", 0.0, 1.06)

# (d) matrices
x0, w, h, gap = 0.655, 0.135, 0.31, 0.012
cells = [("trap", 0, 0, 0), ("trap", T, 0, 1), ("haar", 0, 1, 0), ("haar", T, 1, 1)]
for name, t, r, c in cells:
    a2 = fig.add_axes([x0 + c*(w+gap+0.01), 0.555 - r*(h+0.075), w, h])
    M = snaps[name, t][order]
    a2.imshow(M, cmap=SEQ, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    a2.set_xticks([0, 15]); a2.set_xticklabels(["0", "15"] if r == 1 else [], fontsize=5.5)
    a2.add_patch(plt.Rectangle((-0.5, -0.5), 1, N, fill=False, ec=ORANGE, lw=0.8, zorder=5, clip_on=False))
    lv = [0] + [i for i in range(1, N) if levels[i] != levels[i-1]]
    a2.set_yticks([l - 0.5 for l in lv[1:]], minor=True); a2.grid(which="minor", axis="y", color=AXIS, lw=0.5)
    a2.set_yticks([]); a2.tick_params(which="both", length=0, pad=1)
    for sp in a2.spines.values(): sp.set_visible(True); sp.set_color(AXIS); sp.set_linewidth(0.5)
    if r == 0: a2.set_title(f"$t={t}$", fontsize=TITLE_FS, pad=2)
    if c == 0:
        a2.set_ylabel(["trap circuit", "Haar completion"][r], fontsize=6.5, labelpad=8)
        for l, nxt in zip(lv, lv[1:] + [N]):
            a2.text(-0.9, (l + nxt - 1)/2, f"{levels[l]:+.0f}", fontsize=5, ha="right", va="center", color=INK2)
    if r == 1: a2.set_xlabel("column $y$", fontsize=6.5, labelpad=1)
tag(fig.axes[3], "(d)", -0.33, 1.36)
fig.text(x0 + 0.003, 0.025, "orange box: column 0 = output state $\\psi_t$", fontsize=6, color=INK2)
cb = fig.add_axes([x0 + 2*(w+gap+0.01) + 0.004, 0.17, 0.008, 0.695])
plt.colorbar(plt.cm.ScalarMappable(cmap=SEQ), cax=cb, ticks=[0, 1]); cb.tick_params(labelsize=5.5, length=1.5, pad=1)
cb.set_ylabel(r"$|\langle x|U_t|y\rangle|^2$", fontsize=6.5, labelpad=-4)
for sp in cb.spines.values(): sp.set_visible(False)
fig.text(x0 - 0.028, 0.885, "energy of $|x\\rangle$", fontsize=5.5, color=INK2, ha="center")
save(fig, "fig_trap")
