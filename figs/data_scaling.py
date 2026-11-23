"""规模实验：n=3..10，步数和局部速率随 N 的变化，对比定理 3 的 ρ=2βκΔ（κ=3/(N-1)）。
哈密顿量：横场 Ising 环（g=1，基态非简并）和海森堡环（奇数 n 基态 4 重简并，定理 3 仍适用，Δ 取到下一个不同本征值）。
每个运行到 δ<1e-8 停止。
变体：iid（独立标架）、acc（单 CNOT 懒游走）。η=1/ℓ，每组 5 个 Haar 初态。
更新用列运算：U <- U c (I⊗e^M⊗I) c†，e^M 是 |00> 与 v 张成平面里的转动，不形成 N×N 稠密矩阵乘法。
输出 scaling.csv（每个运行的 δ 轨迹，按步长抽样）和 scaling_summary.csv。"""
import numpy as np, sys, zlib, os
from functools import reduce
from multiprocessing import Pool

def paulis(n):
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1]).astype(complex)
    op = lambda P, i: reduce(np.kron, [P if j == i else I2 for j in range(n)])
    return X, Y, Z, op

def hamiltonian(name, n):
    X, Y, Z, op = paulis(n)
    if name == "tfim": return -sum(op(Z, i) @ op(Z, (i + 1) % n) + op(X, i) for i in range(n))
    if name == "heisenberg": return sum(op(P, i) @ op(P, (i + 1) % n) for i in range(n) for P in (X, Y, Z)) / 4

def gen_GL(n, rng):
    while True:
        A = rng.integers(0, 2, (n, n))
        if round(np.linalg.det(A)) % 2: return A

def colperm(A, n):
    """返回数组 p，p[x] = 整数编码的 A x（比特顺序：qubit 0 是最高位）。"""
    x = np.arange(2 ** n); b = (x[:, None] >> (n - 1 - np.arange(n))) & 1
    y = (b @ A.T) % 2
    return y @ (1 << (n - 1 - np.arange(n)))

def step(U, H, eta, A, e, n):
    """一步：U <- U c E c†，c|x>=|Ax>，E = 边 e 上的 e^M。返回新的 U。"""
    N = 2 ** n; psi = U[:, 0]; Hpsi = H @ psi; f = np.vdot(psi, Hpsi).real; g = 2 * (Hpsi - f * psi)
    p = colperm(A, n)
    # 边 e 上 ab 串（其余比特为 0）的整数编码
    ab = [(a << (n - 1 - e)) | (b << (n - 2 - e)) for a, b in ((0, 1), (1, 0), (1, 1))]
    v = -eta * np.array([np.vdot(U[:, p[y]], g) for y in ab])   # M = |v><00| - |00><v|，v_ab = -η <U|Ab>, g>
    th = np.linalg.norm(v)
    if th == 0: return U, f, g
    vh = v / th
    # 4×4 的 e^M，基 (00,01,10,11)
    R = np.eye(4, dtype=complex); w = np.concatenate([[0], vh])
    R += (np.cos(th) - 1) * (np.outer(np.eye(4)[0], np.eye(4)[0]) + np.outer(w, w.conj()))
    R += np.sin(th) * (np.outer(w, np.eye(4)[0]) - np.outer(np.eye(4)[0], w.conj()))
    V = U[:, p].reshape(N, 2 ** e, 4, 2 ** (n - e - 2))
    V = np.einsum("iakb,kl->ialb", V, R).reshape(N, N)
    Un = np.empty_like(U); Un[:, p] = V
    return Un, f, g

def run(args):
    name, n, var, seed, Tmax = args
    N = 2 ** n; H = hamiltonian(name, n); E = np.linalg.eigvalsh(H); E0 = E[0]
    ell = 2 * (E[-1] - E0); eta = 1 / ell
    rng = np.random.default_rng(zlib.crc32(f"{name}|{n}|{var}|{seed}".encode()))
    r0 = np.random.default_rng(zlib.crc32(f"U0|{name}|{n}|{seed}".encode()))
    U = np.linalg.qr(r0.normal(size=(N, N)) + 1j * r0.normal(size=(N, N)))[0]
    moves = [(np.eye(n, dtype=int) + np.outer(np.eye(n, dtype=int)[i], np.eye(n, dtype=int)[j])) % 2
             for i in range(n) for j in range(n) if i != j]
    A = gen_GL(n, rng); stride = max(1, N // 32); traj = []; t = 0
    while t <= Tmax:
        if var == "iid": A = gen_GL(n, rng)
        elif rng.random() < 0.5: A = (A @ moves[rng.integers(len(moves))]) % 2
        U, f, g = step(U, H, eta, A, rng.integers(n - 1), n)
        d = max(f - E0, 1e-16)
        if t % stride == 0: traj.append((t, d))
        if d < 1e-8: traj.append((t, d)); break
        t += 1
    return name, n, var, seed, traj

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    Tmax = int(sys.argv[1]) if len(sys.argv) > 1 else 1500000
    ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(3, 11))
    jobs = [(h, n, v, s, Tmax) for h in ("tfim", "heisenberg") for n in ns
            for v in ("iid", "acc") for s in range(5)]
    jobs.sort(key=lambda j: -j[1])                     # 大的先跑
    out = sys.argv[3] if len(sys.argv) > 3 else "scaling.csv"
    with Pool(min(len(jobs), int(os.environ.get("NPROC", 50)))) as pool, open(out, "w") as fo:
        fo.write("H,n,variant,seed,t,delta\n")
        for name, n, var, seed, traj in pool.imap_unordered(run, jobs):
            for t, d in traj: fo.write(f"{name},{n},{var},{seed},{t},{d:.6e}\n")
            fo.flush(); print(name, n, var, seed, "steps", traj[-1][0], "final", traj[-1][1], flush=True)
