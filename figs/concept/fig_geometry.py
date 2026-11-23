"""Concept figure 1 (full width): what one step can see.
(a) Toy landscape on the sphere (N=3 real states): four descents from one start.
(b) Accessible strings A_t for the first 40 steps, n=4: without a frame vs with i.i.d. CNOT frames,
    and the long-run hit frequency of every string (10^5 steps) against kappa = 3/(N-1).
(c) Fraction of the gradient a step sees, |w_t|^2/|g_t|^2, along real n=4 Heisenberg trajectories."""
import numpy as np, itertools
from functools import reduce
from scipy.linalg import expm
from pcommon import *
import toy

# ---------------- n = 4 machinery (same conventions as figs/data_convergence.py) ----------------
n, N = 4, 16
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex)
op = lambda P, i: reduce(np.kron, [P if j == i else I2 for j in range(n)])
H = sum(op(P, i) @ op(P, (i+1) % n) for i in range(n) for P in (X, Y, Z))/4
E = np.linalg.eigvalsh(H); E0 = E[0]; ell = 2*(E[-1]-E[0]); eta = 1/ell
bits = lambda x: np.array([(x >> (n-1-k)) & 1 for k in range(n)]); idx = lambda b: int(''.join(map(str, b)), 2)
def rand_GL(rng):
    while True:
        A = rng.integers(0, 2, (n, n))
        if round(np.linalg.det(A)) % 2: return A
def Aset(A, e):
    out = []
    for ab in (1, 2, 3):
        b = np.zeros(n, int); b[e], b[e+1] = ab >> 1, ab & 1; out.append(idx((A @ b) % 2))
    return out
def proj_step(U, ys):
    """one step of the algorithm written through its accessible columns (Lemma 1): exact."""
    psi = U[:, 0]; Fv = np.real(np.vdot(psi, H @ psi)); g = 2*(H @ psi - Fv*psi)
    G = np.zeros((N, N), complex)
    for y in ys:
        a = np.vdot(U[:, y], g)                     # complex derivative along U|y>
        G[y, 0] += -eta*a; G[0, y] += eta*np.conj(a)
    return U @ expm(G), g
paulis = [reduce(np.kron, [(I2, X, Y, Z)[k] for k in ks]) for ks in itertools.product(range(4), repeat=n) if any(ks)]

rng = np.random.default_rng(4)
# (b) rasters
T_r = 40; I_mat = np.eye(n, dtype=int)
raster_nf = np.zeros((N-1, T_r)); raster_fr = np.zeros((N-1, T_r))
for t in range(T_r):
    for y in Aset(I_mat, rng.integers(n-1)): raster_nf[y-1, t] = 1
    for y in Aset(rand_GL(rng), rng.integers(n-1)): raster_fr[y-1, t] = 1
T_f = 100000; freq_nf = np.zeros(N-1); freq_fr = np.zeros(N-1)
for t in range(T_f):
    for y in Aset(I_mat, rng.integers(n-1)): freq_nf[y-1] += 1/T_f
    for y in Aset(rand_GL(rng), rng.integers(n-1)): freq_fr[y-1] += 1/T_f
# (c) fraction of the gradient seen per step, along trajectories (6 Haar initial circuits)
T_c = 600; ratios = {k: np.full((6, T_c), np.nan) for k in ("frame", "pauli", "noframe")}
for seed in range(6):
    r0 = np.random.default_rng(100+seed); U0 = np.linalg.qr(r0.normal(size=(N, N))+1j*r0.normal(size=(N, N)))[0]
    for var in ratios:
        U = U0.copy(); rr = np.random.default_rng(50+seed)
        for t in range(T_c):
            psi = U[:, 0]; Fv = np.real(np.vdot(psi, H @ psi)); g = 2*(H @ psi - Fv*psi); g2 = np.vdot(g, g).real
            if Fv - E0 < 1e-11: break
            if var == "pauli":
                Gp = 1j*paulis[rr.integers(len(paulis))]; d = np.real(np.vdot(g, U @ Gp[:, 0]))
                ratios[var][seed, t] = d**2/g2; U = U @ expm(-eta*d*Gp)
            else:
                ys = Aset(rand_GL(rr) if var == "frame" else I_mat, rr.integers(n-1))
                ratios[var][seed, t] = sum(abs(np.vdot(U[:, y], g))**2 for y in ys)/g2; U, _ = proj_step(U, ys)
np.savez("geometry_c.npz", **ratios)
print({k: np.nanmean(v) for k, v in ratios.items()}, "kappa", 3/(N-1), "Pauli theory", N/(2*(N**2-1)))

# ---------------- figure ----------------
from matplotlib.lines import Line2D
fig = plt.figure(figsize=(7.0, 2.45))
# (a) sphere
ax = fig.add_axes([0.0, 0.17, 0.255, 0.76])
R = view((0.62, 0.50, 0.62), up=(0, 0, -1)); LAM = toy.LAM
S = Sphere(ax, R, lambda P: (P**2*LAM).sum(-1), 0, LAM.max())
psi0 = np.array([0.35, 0.92, 0.18]); psi0 /= np.linalg.norm(psi0)
U0 = toy.frame_from(psi0, np.array([0.2, -0.1, 0.97]))
runs = [("full", INK, "-", 0.9, 0, "exact gradient"), ("random", AQUA, "-", 0.9, 1, "random direction"),
        ("column", BLUE, "-", 1.3, 2, "this work"), ("noframe", MUTED, (0, (3, 1.5)), 1.1, 1, "no frame")]
for mode, col, ls, lw, seed, _ in runs:
    P, _ = toy.run(U0, mode, 80, np.random.default_rng(seed))
    S.curve(densify(P, 12), color=col, lw=lw, ls=ls, z=5 if mode == "column" else 4)
    if mode == "noframe": stop = P[-1]
ex, ey, ez = np.eye(3)
S.dot(psi0, INK, 12); S.dot(ez, BLUE, 24, z=9); S.dot(ex, ORANGE, 22, z=9); S.dot(ey, INK2, 18, marker="^")
S.label(ez, "ground", -0.30, -0.16, ha="right")
S.label(ex, "saddle", -0.05, 0.22, ha="center")
S.label(ey, "max", 0.10, 0.22, ha="center")
S.label(psi0, r"$\psi_0$", 0.22, 0.14)
S.dot(stop, INK2, 20, marker="X", z=10); S.label(stop, "stalls", 0.30, 0.02)
fig.legend([Line2D([], [], color=c, lw=l, ls=st) for _, c, st, l, _, _ in runs], [r[5] for r in runs],
           loc="lower left", bbox_to_anchor=(0.0, -0.01), ncol=2, handlelength=1.5, columnspacing=0.9,
           handletextpad=0.5, fontsize=6.5)
tag(ax, "(a)", 0.0, 1.08)

# (b) accessible strings: rasters + long-run frequencies
lab = [format(y, "04b") for y in range(1, N)]
ttl = ["no frame: 7 of 15 directions", "CNOT frames: all 15 directions"]
for j, (M, fq, col) in enumerate(((raster_nf, freq_nf, INK2), (raster_fr, freq_fr, BLUE))):
    x0 = 0.335 + j*0.225
    a = fig.add_axes([x0, 0.20, 0.150, 0.68]); b = fig.add_axes([x0+0.155, 0.20, 0.042, 0.68])
    a.imshow(M, aspect="auto", cmap=LinearSegmentedColormap.from_list("on", ["#efeee9", col]),
             interpolation="nearest", vmin=0, vmax=1)
    a.set_xticks(np.arange(-0.5, T_r, 1), minor=True); a.set_yticks(np.arange(-0.5, N-1, 1), minor=True)
    a.grid(which="minor", color="white", lw=0.6); a.tick_params(which="minor", length=0)
    a.set_yticks(range(N-1)); a.set_yticklabels(lab if j == 0 else [], fontsize=5.3, family="monospace")
    a.tick_params(axis="y", length=0, pad=1.5)
    a.set_xticks([0, 19, 39]); a.set_xticklabels(["1", "20", "40"]); a.set_xlabel("step $t$", labelpad=1)
    for sp in a.spines.values(): sp.set_visible(False)
    title(a, ttl[j])
    b.barh(range(N-1), fq, height=0.7, color=col)
    b.axvline(3/(N-1), color=ORANGE, lw=0.8, zorder=0)
    b.set_ylim(N-1.5, -0.5); b.set_yticks([]); b.set_xlim(0, 0.7)
    b.set_xticks([3/(N-1)]); b.set_xticklabels([r"$\kappa$"]); b.tick_params(axis="x", labelsize=6, pad=1)
    b.spines["left"].set_visible(False)
    b.set_xlabel("freq.", labelpad=1, fontsize=6.5)
    if j == 0: a.set_ylabel(r"direction $U_t|y\rangle$", labelpad=1)
tag(fig.axes[1], "(b)", -0.30, 1.13)

# (c) fraction of gradient seen per step
ac = fig.add_axes([0.835, 0.20, 0.160, 0.68])
kap, pth = 3/(N-1), N/(2*(N**2-1))
ac.axhline(kap, color=BLUE, lw=0.6, ls=(0, (2, 1.5))); ac.axhline(pth, color=AQUA, lw=0.6, ls=(0, (2, 1.5)))
w = 15; kern = np.ones(w)/w
for k, col, lw in (("noframe", MUTED, LW), ("pauli", AQUA, LW), ("frame", BLUE, LW)):
    A = ratios[k]; tt = np.arange(A.shape[1])
    m = np.nanmean(A, 0); ok = np.sum(~np.isnan(A), 0) >= 3
    sm = np.convolve(np.where(ok, m, 0), kern, "same") / np.convolve(ok.astype(float), kern, "same")
    sm[~ok] = np.nan
    ac.plot(tt, sm, color=col, lw=lw)
ac.set_yscale("log"); ac.set_ylim(3e-5, 1); ac.minorticks_off(); ac.set_xlim(0, 600); ac.set_xticks([0, 250, 500])
ac.set_xlabel("step $t$", labelpad=1); ac.set_ylabel(r"$|w_t|^2/|g_t|^2$", labelpad=1)
ac.grid(axis="y", color=GRID, lw=0.4); ac.set_axisbelow(True)
ac.text(590, kap*1.35, r"frames $\to\kappa=\frac{3}{N-1}$", color=BLUE, fontsize=ANN_FS, ha="right", va="bottom")
ac.text(590, pth*0.36, r"one Pauli $\to\frac{N}{2(N^2-1)}$", color=AQUA_TXT, fontsize=ANN_FS, ha="right", va="top")
ac.text(230, kap*0.62, "converged", color=BLUE, fontsize=ANN_FS, ha="left", va="top")
ac.text(590, 1.4e-4, "no frame: gradient\nleaves the reachable set", color=INK2, fontsize=ANN_FS, ha="right", va="bottom")
tag(ac, "(c)", -0.30, 1.13)
save(fig, "fig_geometry")
