"""对比实验：局部 Pauli 投影（输出侧，Wiersema–Killoran 2023 第 V 节）vs 本文的水平门 + CNOT 标架。
n=4，三个哈密顿量：横场 Ising 周期链 g=1（WK 图 7 的例子）、海森堡环、随机 2-local。
初态：|+>^4（WK 的初始化）和 6 个 Haar 随机电路。
变体：
  ours      本文，独立标架，eta=1/ell
  lp_full   每步把梯度投影到全部 1-local 与最近邻 2-local Pauli 上，输出侧一步 U <- exp(i*eta*G) U
  lp_rand   每步随机取一个局部 Pauli，输出侧
  noframe   本文的两比特水平门，但不夹 CNOT
记录 δ_t = F - E0、|g|^2 = 4 Var 以及局部投影梯度的平方 Σ c_j^2，输出 localpauli.csv。"""
import numpy as np, itertools, sys, zlib
from functools import reduce
from scipy.linalg import expm

n, N = 4, 16
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex)
op = lambda P, i: reduce(np.kron, [P if j == i else I2 for j in range(n)])

def hamiltonians():
    tfim = -sum(op(Z, i) @ op(Z, (i + 1) % n) + op(X, i) for i in range(n))
    heis = sum(op(P, i) @ op(P, (i + 1) % n) for i in range(n) for P in (X, Y, Z)) / 4
    rng = np.random.default_rng(7)
    rand = sum(rng.normal() * op(P, i) @ op(Q, i + 1) for i in range(n - 1) for P in (X, Y, Z) for Q in (X, Y, Z))
    rand = rand + sum(rng.normal() * op(P, i) for i in range(n) for P in (X, Y, Z))
    return {"tfim": tfim, "heisenberg": heis, "random2local": rand}

# 局部 Pauli 集：1-local 和周期最近邻 2-local（与 WK 的 P^{1-local} ∪ P^{2-local}_{n.n.} 一致）
local = [op(P, i) for i in range(n) for P in (X, Y, Z)]
local += [op(P, i) @ op(Q, (i + 1) % n) for i in range(n) for P in (X, Y, Z) for Q in (X, Y, Z)]

bits = lambda x: np.array([(x >> (n - 1 - k)) & 1 for k in range(n)])
idx = lambda b: int(''.join(map(str, b)), 2)
def perm(A):
    P = np.zeros((N, N), complex)
    for x in range(N): P[idx((A @ bits(x)) % 2), x] = 1
    return P
def rand_GL(rng):
    while True:
        A = rng.integers(0, 2, (n, n))
        if round(np.linalg.det(A)) % 2: return A
gens = []
for ab in (1, 2, 3):
    for ph in (1, 1j):
        M = np.zeros((4, 4), complex); M[ab, 0] = ph; M[0, ab] = -np.conj(ph); gens.append(M)
embed = lambda M, e: reduce(np.kron, [np.eye(2 ** e), M, np.eye(2 ** (n - e - 2))])
e0 = np.eye(N)[0].astype(complex)

def run(H, U0, var, T, rng, eta_out):
    E = np.linalg.eigvalsh(H); E0 = E[0]; ell = 2 * (E[-1] - E[0]); eta = 1 / ell
    F = lambda U: np.real(np.vdot(U[:, 0], H @ U[:, 0]))
    U = U0.copy(); out = []
    for t in range(T + 1):
        psi = U[:, 0]; f = F(U); g = 2 * (H @ psi - f * psi)
        Hpsi = H @ psi
        c = np.array([-2 * np.imag(np.vdot(P @ psi, Hpsi)) for P in local])  # dF/dθ along e^{-iθP} ψ
        if t % 10 == 0 or t == T:
            out.append((t, max(f - E0, 1e-16), np.vdot(g, g).real, float(c @ c)))
        if var == "ours":
            cfr = perm(rand_GL(rng)); e = rng.integers(n - 1)
            x = [-eta * np.real(np.vdot(g, U @ cfr @ embed(Mj, e) @ e0)) for Mj in gens]
            U = U @ cfr @ embed(expm(sum(xj * Mj for xj, Mj in zip(x, gens))), e) @ cfr.conj().T
        elif var == "noframe":
            e = rng.integers(n - 1)
            x = [-eta * np.real(np.vdot(g, U @ embed(Mj, e) @ e0)) for Mj in gens]
            U = U @ embed(expm(sum(xj * Mj for xj, Mj in zip(x, gens))), e)
        elif var == "lp_full":
            G = sum(cj * P for cj, P in zip(c, local))
            U = expm(1j * eta_out * G) @ U
        elif var == "lp_rand":
            j = rng.integers(len(local))
            U = expm(1j * eta_out * c[j] * local[j]) @ U
    return out

if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    Hs = hamiltonians()
    plus = reduce(np.kron, [np.array([[1, 1], [1, -1]]) / np.sqrt(2)] * n).astype(complex)  # Hadamard^{⊗4}, 输出 |+>^4
    rows = []
    for hname, H in Hs.items():
        E = np.linalg.eigvalsh(H); ell = 2 * (E[-1] - E[0])
        starts = [] if hname == "heisenberg" else [("plus", plus)]  # |+>^4 是海森堡环的本征态，跳过
        for s in range(6):
            r = np.random.default_rng(300 + s)
            starts.append((f"haar{s}", np.linalg.qr(r.normal(size=(N, N)) + 1j * r.normal(size=(N, N)))[0]))
        for sname, U0 in starts:
            for var, eta_out in (("ours", None), ("noframe", None), ("lp_full", 0.5 / ell), ("lp_full", 0.1 / ell), ("lp_rand", 1 / ell)):
                rng = np.random.default_rng(zlib.crc32(f"{hname}|{sname}|{var}|{eta_out}".encode()))
                for (t, d, g2, c2) in run(H, U0, var, T, rng, eta_out):
                    rows.append((hname, sname, var if eta_out is None else f"{var}@{eta_out*ell:.1f}/ell", t, d, g2, c2))
            print(hname, sname, "done", flush=True)
    with open("localpauli.csv", "w") as f:
        f.write("H,start,variant,t,delta,g2,localproj2\n")
        for r in rows: f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6e},{r[5]:.6e},{r[6]:.6e}\n")
