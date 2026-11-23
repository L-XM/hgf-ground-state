"""Data for the fiber figure (Theorem 5, SM S6), real algorithm, n=3.
H: random three-qubit 2-local Hamiltonian (seed 5, same as figs/data_lyapunov.py). Saddle: level k=1 (lambda_1),
so E^u = span_C{v_0} is a real plane. Tube coordinates (Lemma S-tube): U_0 = exp(X(delta)) W, delta = delta_s + z v_0.
One fixed i.i.d. frame sequence omega, one fixed W and delta_s; z runs over a grid of the fiber.
Saved: escape time on the grid (steps until F < lambda_1 - gap/4), the refined bad point z*, and
|delta_s,t|, |delta_u,t| for trajectories started at distances 10^-1..10^-7 from z* (phase portrait)."""
import numpy as np, multiprocessing as mp
from scipy.optimize import minimize
import alg

n = 3; N, op, perm, rand_GL, step = alg.tools(n); X, Y, Z = alg.X, alg.Y, alg.Z
rng = np.random.default_rng(5)
H = sum(rng.normal()*op(P, i)@op(Q, i+1) for i in range(n-1) for P in (X, Y, Z) for Q in (X, Y, Z))
H = H + sum(rng.normal()*op(P, i) for i in range(n) for P in (X, Y, Z)); H = (H+H.conj().T)/2
E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); eta = 0.9/ell
lam = E[1]; v = V[:, 1]; gap = E[1]-E[0]; thr = lam - gap/4
r2 = np.random.default_rng(11)
W = np.linalg.qr(r2.normal(size=(N, N))+1j*r2.normal(size=(N, N)))[0]; W[:, 0] = v; W = np.linalg.qr(W)[0]
W = W*np.exp(-1j*np.angle(np.vdot(v, W[:, 0])))                        # W|0> = v exactly
ds = V[:, 2:] @ (r2.normal(size=N-2)+1j*r2.normal(size=N-2)); ds *= 0.05/np.linalg.norm(ds)
TMAX = 1500
frames = [(perm(rand_GL(r2)), int(r2.integers(n-1))) for _ in range(TMAX)]
from scipy.linalg import expm
def U0(z):
    d = ds + z*V[:, 0]; return expm(np.outer(d, v.conj()) - np.outer(v, d.conj())) @ W
def coords(psi):
    """tube coordinate delta of the state (lambda nondegenerate): phase-fixed component orthogonal to v."""
    a = np.vdot(v, psi); ph = np.exp(-1j*np.angle(a)); d = (psi - a*v)*ph
    s = np.linalg.norm(d); d = d*(np.arcsin(min(s, 1))/s if s > 0 else 1)
    return np.linalg.norm(V[:, 2:].conj().T @ d), np.abs(np.vdot(V[:, 0], d))
def escape(z):
    U = U0(z)
    for t in range(TMAX):
        if np.real(np.vdot(U[:, 0], H@U[:, 0])) < thr: return t
        U = step(U, H, *frames[t], eta)
    return TMAX
def du_at(zr, T=260):
    U = U0(zr[0]+1j*zr[1])
    for t in range(T): U = step(U, H, *frames[t], eta)
    return coords(U[:, 0])[1]
def portrait(z, T=600):
    U = U0(z); out = []
    for t in range(T):
        out.append(coords(U[:, 0])); U = step(U, H, *frames[t], eta)
    return np.array(out)

if __name__ == "__main__":
    print("levels", np.round(E, 3), "eta", eta, "gap", gap)
    G = 101; L = 0.08; xs = np.linspace(-L, L, G)
    with mp.Pool(54) as p:
        esc = np.array(p.map(escape, [x+1j*y for y in xs for x in xs], chunksize=8)).reshape(G, G)
    i, j = np.unravel_index(np.argmax(esc), esc.shape); z0 = xs[j]+1j*xs[i]
    print("grid max", esc.max(), "at", z0)
    best = None
    for T in (200, 320, 440):
        res = minimize(du_at, [z0.real, z0.imag] if best is None else best, args=(T,), method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-16, maxiter=600))
        best = res.x; print("T", T, "z*", best, "|du_T|", res.fun)
    zs = best[0]+1j*best[1]
    dirs = np.exp(1j*np.linspace(0, 2*np.pi, 9)[:-1]); dists = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7]
    starts = [zs] + [zs + r*0.05*u for r in dists for u in dirs[::2]]
    with mp.Pool(len(starts)) as p:
        por = np.array(p.map(portrait, starts)); et = np.array(p.map(escape, starts))
    print("escape times of portrait starts", et[:1], et[1:].reshape(len(dists), -1).mean(1))
    np.savez("fiber.npz", esc=esc, xs=xs, zs=zs, por=por, et=et, dists=np.array(dists), lam=lam, E=E, eta=eta,
             ds_norm=0.05)
