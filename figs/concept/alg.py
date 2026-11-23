"""The algorithm for n qubits, same conventions as figs/data_extra.py (perm frames, six horizontal generators)."""
import numpy as np
from functools import reduce
from scipy.linalg import expm
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex)


def tools(n):
    N = 2**n; op = lambda P, i: reduce(np.kron, [P if j == i else I2 for j in range(n)])
    bits = lambda x: np.array([(x >> (n-1-k)) & 1 for k in range(n)]); idx = lambda b: int(''.join(map(str, b)), 2)
    def perm(A):
        P = np.zeros((N, N), complex)
        for x in range(N): P[idx((A @ bits(x)) % 2), x] = 1
        return P
    def rand_GL(rng):
        while True:
            A = rng.integers(0, 2, (n, n))
            if round(np.linalg.det(A)) % 2: return A
    loc = []
    for ab in (1, 2, 3):
        for ph in (1, 1j):
            M = np.zeros((4, 4), complex); M[ab, 0] = ph; M[0, ab] = -np.conj(ph); loc.append(M)
    embed = lambda M, e: reduce(np.kron, [np.eye(2**e), M, np.eye(2**(n-e-2))])
    e0 = np.eye(N)[0].astype(complex)
    def step(U, H, c, e, eta):
        psi = U[:, 0]; Fv = np.real(np.vdot(psi, H @ psi)); g = 2*(H @ psi - Fv*psi)
        x = [-eta*np.real(np.vdot(g, U @ c @ embed(Mj, e) @ e0)) for Mj in loc]
        return U @ c @ embed(expm(sum(xj*Mj for xj, Mj in zip(x, loc))), e) @ c.conj().T
    return N, op, perm, rand_GL, step
