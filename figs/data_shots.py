"""有限 shot 实验（回应审稿意见 5）。噪声模型：每个能量估计 = 精确值 + 高斯噪声，方差 σ_E² = Σ_k h_k² / S，
h_k 是 H 的 Pauli 系数（每项各用 S 个 shot 独立测量时方差的上界）。导数 ∂_j = E(π/4) − E(−π/4)，噪声方差 2σ_E²。
(A) n=4 海森堡环和横场 Ising 环，独立标架，η=1/ℓ，S = 1e2..1e6，每组 5 个 Haar 初态，4000 步。
(B) 平台高度随 n：横场 Ising 环，n=3..8，S=1e4，独立标架，400N 步，每组 3 个初态。
(C) 陷阱加噪声：n=4，H=Σ Z_iZ_{i+1}（开边界），命题 1 的陷阱电路，独立 CNOT 标架，η=1/8，S=1e3、1e5、∞。
输出 shots.csv：expt,H,n,S,seed,t,delta"""
import numpy as np, sys, zlib, os
from multiprocessing import Pool
from functools import reduce
from data_scaling import hamiltonian, gen_GL, colperm, paulis


def zz_open(n):
    X, Y, Z, op = paulis(n)
    return sum(op(Z, i) @ op(Z, i + 1) for i in range(n - 1))


def noisy_step(U, H, eta, A, e, n, sd, rng):
    """与 data_scaling.step 相同，但六个导数各加 N(0, sd²) 噪声。"""
    N = 2 ** n; psi = U[:, 0]; Hpsi = H @ psi; f = np.vdot(psi, Hpsi).real; g = 2 * (Hpsi - f * psi)
    p = colperm(A, n)
    ab = [(a << (n - 1 - e)) | (b << (n - 2 - e)) for a, b in ((0, 1), (1, 0), (1, 1))]
    d = np.array([np.vdot(U[:, p[y]], g) for y in ab])
    if sd > 0: d = d + sd * (rng.normal(size=3) + 1j * rng.normal(size=3))
    v = -eta * d; th = np.linalg.norm(v)
    if th == 0: return U, f
    vh = v / th
    R = np.eye(4, dtype=complex); w = np.concatenate([[0], vh]); e0 = np.eye(4)[0]
    R += (np.cos(th) - 1) * (np.outer(e0, e0) + np.outer(w, w.conj()))
    R += np.sin(th) * (np.outer(w, e0) - np.outer(e0, w.conj()))
    V = U[:, p].reshape(N, 2 ** e, 4, 2 ** (n - e - 2))
    V = np.einsum("iakb,kl->ialb", V, R).reshape(N, N)
    Un = np.empty_like(U); Un[:, p] = V
    return Un, f


def run(args):
    expt, name, n, S, seed, T = args
    N = 2 ** n
    H = zz_open(n) if name == "zz" else hamiltonian(name, n)
    E = np.linalg.eigvalsh(H); E0 = E[0]; ell = 2 * (E[-1] - E0)
    eta = 1 / 8 if expt == "C" else 1 / ell
    sig2 = np.trace(H @ H).real / N - (np.trace(H).real / N) ** 2      # Σ_k h_k²
    sd = 0.0 if S == 0 else np.sqrt(2 * sig2 / S)
    rng = np.random.default_rng(zlib.crc32(f"{expt}|{name}|{n}|{S}|{seed}".encode()))
    r0 = np.random.default_rng(zlib.crc32(f"U0|{expt}|{name}|{n}|{seed}".encode()))
    if expt == "C":
        # 陷阱：U0 span{|0>,|1>} = span{|0000>,|0001>}，其余列在补空间里 Haar 随机
        a = r0.normal(size=2) + 1j * r0.normal(size=2); a /= np.linalg.norm(a)
        U = np.zeros((N, N), complex); U[:2, 0] = a; U[:2, 1] = [-a[1].conj(), a[0].conj()]
        Q = np.linalg.qr(r0.normal(size=(N - 2, N - 2)) + 1j * r0.normal(size=(N - 2, N - 2)))[0]
        U[2:, 2:] = Q
    else:
        U = np.linalg.qr(r0.normal(size=(N, N)) + 1j * r0.normal(size=(N, N)))[0]
    stride = max(1, T // 400); traj = []
    for t in range(T + 1):
        U, f = noisy_step(U, H, eta, gen_GL(n, rng), rng.integers(n - 1), n, sd, rng)
        if t % stride == 0: traj.append((t, max(f - E0, 1e-16)))
    return expt, name, n, S, seed, traj


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [("A", h, 4, S, s, 4000) for h in ("heisenberg", "tfim") for S in (0, 100, 1000, 10**4, 10**5, 10**6) for s in range(5)]
    jobs += [("B", "tfim", n, 10**4, s, 400 * 2 ** n) for n in range(3, 9) for s in range(3)]
    jobs += [("C", "zz", 4, S, s, 3000) for S in (0, 1000, 10**5) for s in range(5)]
    jobs.sort(key=lambda j: -j[5])
    with Pool(min(len(jobs), int(os.environ.get("NPROC", 40)))) as pool, open("shots.csv", "w") as fo:
        fo.write("expt,H,n,S,seed,t,delta\n")
        for expt, name, n, S, seed, traj in pool.imap_unordered(run, jobs):
            for t, d in traj: fo.write(f"{expt},{name},{n},{S},{seed},{t},{d:.6e}\n")
            fo.flush(); print(expt, name, n, S, seed, "final", traj[-1][1], flush=True)
