"""定理 G′（法向完全双曲）的数值核对：n=3，随机 2-local H，iid CNOT 标架，η = 0.5/ℓ。
已证明：Lyapunov 谱里至少 2k 个 >= γ̃，至少 p 个 <= -γ̃+，2k + p = 法向实维数，所以没有零指数。
  γ̃  = (p_m/(2m)) log(1 + 4βμ /C_m)，  C_m  = (N-1)(1+ηℓ√m)²
  γ̃+ = -(p_m/(2m)) log(1 - 4βμ+/C_m)（正锥里 q 在每个好块上至多乘 1 - 4βμ+/C_m；4βμ+/C_m <= 1/3）
另外看谱对补全（隐藏列 z）的依赖：5 个随机补全。"""
import numpy as np
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(11)
n = 3; N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0])
eta = 0.5/ell; beta = eta*(1-eta*ell/2); m = 12
def covers():
    cov = set()
    for _ in range(m): cov |= set(Aset(rand_GL(n), rng.integers(n-1), n))
    return len(cov) == N-1
p_m = np.mean([covers() for _ in range(20000)])
Cm = (N-1)*(1+eta*ell*np.sqrt(m))**2
print(f"m={m}, p_m≈{p_m:.3f}, C_m={Cm:.1f}")
for k in (1, 3, 6):
    lam = E[k]; mu = lam-E[k-1]; mup = E[k+1]-lam; p = 2*(N-1-k)
    g = p_m/(2*m)*np.log(1+4*beta*mu/Cm); gp = -p_m/(2*m)*np.log(1-4*beta*mup/Cm)
    rows = []
    for trial in range(5):
        U = haar(N); U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
        Hc = 2*(U[:,1:].conj().T@H@U[:,1:] - lam*np.eye(N-1))
        sp = lyap_spectrum(n, Hc, eta, 15000)
        rows.append((sp[2*k-1], sp[2*k], sp[-p] if p else np.nan, np.sum(sp > 0), np.sum(sp < 0)))
    r = np.array(rows)
    print(f"k={k}: 2k={2*k}, p={p}；γ̃={g:.1e}, γ̃+={gp:.1e}")
    print(f"   5 个补全：正指数个数 {r[:,3].astype(int)}，负指数个数 {r[:,4].astype(int)}")
    print(f"   第 2k 大 ∈ [{r[:,0].min():.4f}, {r[:,0].max():.4f}] (>= γ̃: {bool(np.all(r[:,0] >= g))})；"
          f"第 2k+1 大 ∈ [{r[:,1].min():.4f}, {r[:,1].max():.4f}] (<= -γ̃+: {bool(np.all(r[:,1] <= -gp))})")
