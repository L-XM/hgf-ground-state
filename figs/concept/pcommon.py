"""Shared style for the concept figures of the paper (English, true size, REVTeX fonts).
Colours: the same validated categorical slots as figs/plot_figs.py
(ours = blue, Pauli = aqua, trap = orange, no frame = grey), a light warm-grey terrain for
energies, and a one-hue blue ramp for frequencies and matrix magnitudes."""
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm

import os
for f in ("lmroman10-regular", "lmroman10-italic", "lmroman10-bold"):   # Latin Modern from TeX Live, if present
    p = f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf"
    if os.path.exists(p): fm.fontManager.addfont(p)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
DEEP = "#184f95"
# terrain: low energy = darker warm grey (the pit), high energy = paper white
TERRAIN = LinearSegmentedColormap.from_list("terrain", ["#a9a59c", "#c4c0b7", "#dad7cf", "#ebe9e3", "#f7f6f2"])
# sequential blue ramp (palette steps 100 -> 650), zero = surface
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"])

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "axes.unicode_minus": True})

HALO = lambda lw: [pe.Stroke(linewidth=lw + 1.3, foreground="white"), pe.Normal()]

# shared sizes and extra slots for all figures
VIOLET = "#4a3aa7"                              # categorical slot 7: second start of the same algorithm
ORANGE_TXT, AQUA_TXT = "#c4511f", "#12805a"     # darker text versions (orange and aqua are < 3:1 on white)
LW, LW_REF = 1.2, 0.8                           # data lines, reference lines (bounds, fits)
TITLE_FS, ANN_FS = 7.5, 6.8                     # panel titles, direct labels inside a panel
FLOOR = 1e-13                                   # lower limit of all energy-error axes (machine precision)


def tag(ax, s, x=0.0, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color=INK)


def title(ax, s):
    ax.set_title(s, fontsize=TITLE_FS, color=INK, loc="left", pad=3)


def logy(ax, top=100, bottom=FLOOR):
    ax.set_yscale("log"); ax.set_ylim(bottom, top); ax.minorticks_off()
    ax.set_yticks([1e-12, 1e-8, 1e-4, 1])
    ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)


def upto_floor(t, y, floor=FLOOR):
    """Cut a curve at the first point below the floor, so it ends on the axis instead of running along it."""
    i = np.argmax(y < floor) if np.any(y < floor) else len(y) - 1
    return t[:i + 1], np.clip(y[:i + 1], floor, None)


def band(ax, t, a, col, ls="-", lw=LW, floor=FLOOR):
    """Median (geometric) line and min-max shading over runs, both cut at the floor."""
    a = np.clip(a, floor * 1e-3, None); med = np.exp(np.median(np.log(a), 0))
    tb, hi = upto_floor(t, a.max(0), floor); lo = np.clip(a.min(0)[:len(tb)], floor, None)
    ax.fill_between(tb, lo, hi, color=col, alpha=0.14, lw=0)
    ax.plot(*upto_floor(t, med, floor), color=col, ls=ls, lw=lw, solid_capstyle="round")


def view(front, up=(0, 0, 1)):
    z = np.asarray(front, float); z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    return np.array([x, y, z])


class Sphere:
    """Orthographic sphere. Colour = energy, thin lines = iso-energy contours, soft light from upper left."""

    def __init__(self, ax, R, energy, vmin, vmax, res=600, levels=9, light=(-0.5, 0.6, 0.62)):
        self.ax, self.R = ax, R
        s = np.linspace(-1, 1, res); X, Y = np.meshgrid(s, s[::-1])
        inside = X**2 + Y**2 <= 1; Z = np.sqrt(np.clip(1 - X**2 - Y**2, 0, None))
        P = np.stack([X, Y, Z], -1) @ R
        E = energy(P); c = TERRAIN((E - vmin) / (vmax - vmin))[..., :3]
        L = np.asarray(light) / np.linalg.norm(light); lam = np.clip(X*L[0] + Y*L[1] + Z*L[2], 0, 1)
        img = np.clip(c * (0.86 + 0.14*lam)[..., None] + 0.05*lam[..., None]**10, 0, 1)
        ax.imshow(np.dstack([img, inside.astype(float)]), extent=(-1, 1, -1, 1), interpolation="bilinear", zorder=1)
        Em = np.where(X**2 + Y**2 <= 0.995, E, np.nan)
        ax.contour(X, Y, Em, levels=np.linspace(vmin, vmax, levels + 2)[1:-1], colors="white",
                   linewidths=0.45, alpha=0.9, zorder=1.5)
        t = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=0.5, zorder=2)
        ax.set_xlim(-1.04, 1.04); ax.set_ylim(-1.04, 1.04); ax.set_aspect("equal"); ax.axis("off")

    def cam(self, P):
        return np.atleast_2d(P) @ self.R.T

    def curve(self, P, color=INK, lw=1.0, ls="-", halo=True, back=False, z=4, alpha=1.0):
        C = self.cam(P); front = C[:, 2] >= -1e-9
        runs, cur = [], [0]
        for i in range(1, len(C)):
            if front[i] == front[cur[-1]]: cur.append(i)
            else: runs.append(cur); cur = [i]
        runs.append(cur)
        for r in runs:
            if front[r[0]]:
                self.ax.plot(C[r, 0], C[r, 1], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                             solid_capstyle="round", path_effects=HALO(lw) if halo else None)
            elif back:
                self.ax.plot(C[r, 0], C[r, 1], color=color, lw=lw*0.6, ls=(0, (1.5, 1.5)), alpha=0.45, zorder=3)

    def arrow_at(self, P, i, color=INK, lw=0.8, ms=6):
        C = self.cam(P)
        if C[i, 2] > 0 and C[i+1, 2] > 0:
            self.ax.annotate("", xy=C[i+1, :2], xytext=C[i, :2], zorder=6,
                             arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms,
                                             shrinkA=0, shrinkB=0))

    def tangent(self, p, v, L=0.30, color=INK, lw=1.1, z=8, ms=7):
        p = np.asarray(p, float); v = np.asarray(v, float); v = v - (v @ p)*p; v /= np.linalg.norm(v)
        a = self.cam(p)[0]; b = self.cam(np.cos(L)*p + np.sin(L)*v)[0]
        self.ax.annotate("", xy=b[:2], xytext=a[:2], zorder=z,
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms,
                                         shrinkA=0, shrinkB=0, path_effects=HALO(lw)))
        return b

    def dot(self, p, color=INK, s=14, marker="o", z=7, ec="white"):
        C = self.cam(p)[0]
        if C[2] >= 0:
            self.ax.scatter(C[0], C[1], s=s, color=color, ec=ec, lw=0.7, zorder=z, marker=marker)
        return C

    def label(self, p, text, dx, dy, ha="left", color=INK, fs=7, va="center"):
        C = self.cam(p)[0]
        self.ax.annotate(text, xy=C[:2], xytext=(C[0]+dx, C[1]+dy), ha=ha, va=va, fontsize=fs, color=color,
                         zorder=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=1, shrinkB=2))


def densify(path, k=10):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        w = np.arccos(np.clip(a @ b, -1, 1))
        if w < 1e-12: continue
        for s in np.linspace(0, 1, k)[1:]:
            out.append((np.sin((1-s)*w)*a + np.sin(s*w)*b)/np.sin(w))
    return np.array(out)


def save(fig, name):
    fig.savefig(f"{name}.pdf"); fig.savefig(f"{name}.png", dpi=300)
    plt.close(fig)
