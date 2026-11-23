"""猜想 E 的周期覆盖版本（定理 E_per）的数值核对。n=3，随机 2-local H，η = 0.9/ℓ。
标架按固定的周期表 (A_1,e_1),...,(A_m,e_m) 循环，表内各步的 A_s 之并覆盖全部非零串。
(i)  一个周期的复合映射 G 在鞍点不动点处的线性化 L = Π(I - η P_s 𝓗)：模 > 1 的特征值个数（证明给出 >= 1；观察 = 2k）。
(ii) PL 不等式 |g|² >= μ+ (F-λ)（λ <= F <= λ+μ+/2）。
(iii) 30 个 Haar 初始电路 + 每个鞍点附近 3 个初态：终点能量是否都是基态。"""
import numpy as np
from functools import reduce
from scipy.linalg import expm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(21)
n = 3; N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); eta = 0.9/ell
sched, cov = [], set()
while len(cov) < N-1:
    A, e = rand_GL(n), rng.integers(n-1); sched.append((A, e)); cov |= set(Aset(A, e, n))
m = len(sched); print(f"周期表长度 m = {m}")
loc = []
for ab in (1,2,3):
    for ph in (1, 1j):
        M = np.zeros((4,4),complex); M[ab,0] = ph; M[0,ab] = -np.conj(ph); loc.append(M)
embed = lambda M,e: reduce(np.kron, [np.eye(2**e), M, np.eye(2**(n-e-2))])
e0 = np.eye(N)[0].astype(complex)
def perm(A):
    P = np.zeros((N,N), complex)
    for x in range(N): P[idx((A@bits(x,n))%2), x] = 1
    return P
frames = [(perm(A), e) for A, e in sched]
def step(U, c, e):
    psi = U[:,0]; F = np.real(np.vdot(psi, H@psi)); g = 2*(H@psi - F*psi)
    x = [-eta*np.real(np.vdot(g, U@c@embed(Mj,e)@e0)) for Mj in loc]
    return U @ c @ embed(expm(sum(xj*Mj for xj,Mj in zip(x,loc))), e) @ c.conj().T
F = lambda U: np.real(np.vdot(U[:,0], H@U[:,0]))

print("(i) 鞍点处一个周期的线性化")
for k in range(1, N):
    U = haar(N); U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
    Hc = realify(2*(U[:,1:].conj().T@H@U[:,1:] - E[k]*np.eye(N-1))); L = np.eye(2*(N-1))
    for A, e in sched:
        p = np.zeros(N-1); p[[y-1 for y in Aset(A, e, n)]] = 1
        L = (np.eye(2*(N-1)) - eta*np.r_[p,p][:,None]*Hc) @ L
    ev = np.abs(np.linalg.eigvals(L))
    print(f"  k={k}: |特征值|>1 的个数 {np.sum(ev>1+1e-9):2d}（2k={2*k}），最大模 {ev.max():.3f}，=1 的个数 {np.sum(abs(ev-1)<1e-9)}")

print("(ii) PL 不等式")
worst = np.inf
for k in range(N-1):
    lam, mup = E[k], E[k+1]-E[k]
    for _ in range(2000):
        psi = V[:,k] + 0.3*rng.random()*(rng.normal(size=N)+1j*rng.normal(size=N)); psi /= np.linalg.norm(psi)
        f = np.real(np.vdot(psi, H@psi)) - lam
        if 0 < f <= mup/2:
            g = 2*(H@psi - (f+lam)*psi); worst = min(worst, np.vdot(g,g).real/(mup*f))
print(f"  min |g|²/(μ+(F-λ)) = {worst:.3f}（应 >= 1）")

print("(iii) 终点能量 - E0")
def run(U, T=600):
    for t in range(T):
        for c, e in frames: U = step(U, c, e)
    return F(U) - E[0]
res = [run(haar(N)) for _ in range(30)]
print(f"  Haar 初态 30 个：max(F-E0) = {max(res):.1e}")
for k in range(1, N):
    out = []
    for _ in range(3):
        U = haar(N); psi = V[:,k] + 1e-4*(rng.normal(size=N)+1j*rng.normal(size=N)); U[:,0] = psi/np.linalg.norm(psi)
        out.append(run(np.linalg.qr(U)[0]))
    print(f"  鞍点 k={k} 附近 3 个初态：max(F-E0) = {max(out):.1e}")
