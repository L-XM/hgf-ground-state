"""Shared setup for the fiber figure: real 3-qubit Hamiltonian, first excited level, one completion, one frame sequence."""
import numpy as np, alg
from scipy.linalg import expm
n = 3; N, op, perm, rand_GL, step = alg.tools(n); X, Z = alg.X, alg.Z; Yp = alg.Y
rng = np.random.default_rng(2)
H = sum(rng.normal()*op(P, i)@op(Q, i+1) for i in range(n-1) for P, Q in ((X, X), (Yp, Yp), (Z, Z), (X, Z), (Z, X)))
H = np.real(H + sum(rng.normal()*op(P, i) for i in range(n) for P in (X, Z)))
E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); eta = 0.5/ell
k = 1; lam = E[k]; v = V[:, k]; vu = V[:, 0]                     # saddle and its unstable direction (real slice)
us = V[:, 2:] @ np.random.default_rng(7).normal(size=N-2); us /= np.linalg.norm(us)   # a stable direction
W = np.linalg.qr(np.random.default_rng(3).normal(size=(N, N)))[0]; W[:, 0] = v; W = np.linalg.qr(W)[0]
W *= np.sign(W[:, 0] @ v)
fr = np.random.default_rng(11); T = 2600
FRAMES = [(perm(rand_GL(fr)), fr.integers(n-1)) for _ in range(T)]
def start(x, s):
    d = x*us + s*vu; Xd = np.outer(d, v) - np.outer(v, d)
    return (expm(Xd) @ W).astype(complex)
def run(x, s, T=T, keep=False, stop=None):
    U = start(x, s); out = []
    for t in range(T):
        psi = U[:, 0].real
        if keep: out.append(psi.copy())
        if stop is not None and abs(psi @ vu) > stop: return t, out
        c, e = FRAMES[t]; U = step(U, H, c, e, eta)
    return T, out
