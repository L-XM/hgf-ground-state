"""定理 E 证明（第 7.4.7–7.4.9 节）的数值核对。n=3，随机非简并对角 H，命题 C 的陷阱，iid CNOT 标架，η = 0.5/ℓ。
(1) 嵌入后的生成元 G = c·(M⊗I)·c† 相对 Stab|0> 的竖直分量：‖G_k‖_F/‖G_m‖_F 应 = sqrt(2^{n-2}-1)，所以 k 有一阶漂移。
(2) 引理 R：陷阱轨迹上，每个好块（长 m，覆盖全部非零串）让 F-λ 至少缩小一个固定比例；|δ_t| 指数衰减。
(3) 圆盘引理：在陷阱轨迹第 T 步沿 E^u 加扰动 u（k 不变），两条轨迹用同一串标架。
    a_t = δ'_t - δ_t，Φ_t = -q(a_t)。应有 Φ_t/Φ_T >= 1/2 直到离开邻域；|a_t|^2 <= L_t Φ_t；扰动轨迹最终到基态。"""
import numpy as np
from functools import reduce
from scipy.linalg import expm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(23)
n = 3; N = 2**n
vals = np.sort(rng.uniform(-1, 1, N)); rest = list(rng.permutation([vals[i] for i in range(N) if i not in (2, 5)]))
Ed = np.array([vals[5], vals[2]] + rest)            # |001> 是第 3 低的能级（下面有 2 个），|000> 在它上面
H = np.diag(Ed).astype(complex); lam = Ed[1]; E0 = Ed.min(); ell = 2*(Ed.max()-Ed.min())
eta = 0.5/ell; beta = eta*(1-eta*ell/2); m = 12
below = np.sort(Ed[Ed < lam]); mu = lam - below[-1]; mup = np.min(Ed[Ed > lam]) - lam
Hc = 2*(H - lam*np.eye(N)); q = lambda a: np.real(np.vdot(a, Hc@a))
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
def gen(U, c, e):
    psi = U[:,0]; Fv = np.real(np.vdot(psi, H@psi)); g = 2*(H@psi - Fv*psi)
    return sum(-eta*np.real(np.vdot(g, U@c@embed(Mj,e)@e0))*Mj for Mj in loc)
step = lambda U, c, e: U @ c @ embed(expm(gen(U, c, e)), e) @ c.conj().T
F = lambda U: np.real(np.vdot(U[:,0], H@U[:,0]))
v = np.eye(N)[1].astype(complex)
dl = lambda U: U[:,0]/U[1,0] - v
z = np.eye(N, dtype=complex)[:, [1,0]+list(range(2,N))]   # z|0> = |001>
def Xof(d):
    X = np.zeros((N,N), complex); X[1:,0] = d; X[0,1:] = -d.conj(); return X
def sigma(d):
    a = np.linalg.norm(d)
    return z if a == 0 else z@expm(Xof(np.arctan(a)*(z[:,1:].conj().T@d)/a))
kof = lambda U: np.linalg.inv(sigma(dl(U)))@U
print(f"能级 λ = {lam:.3f}（下面 {np.sum(Ed<lam)} 个），μ = {mu:.3f}，μ+ = {mup:.3f}，E0 = {E0:.3f}")

print("(1) ‖G_k‖_F/‖G_m‖_F，应为", np.sqrt(2**(n-2)-1))
P0 = np.outer(e0, e0)
rat = []
for _ in range(20):
    U = haar(N); A = rand_GL(n); e = rng.integers(n-1); c = perm(A)
    G = c@embed(gen(U, c, e), e)@c.conj().T
    Gm = (np.eye(N)-P0)@G@P0 + P0@G@(np.eye(N)-P0); rat.append(np.linalg.norm(G-Gm)/np.linalg.norm(Gm))
print(f"  20 个随机点：[{min(rat):.6f}, {max(rat):.6f}]")

frames = [(rand_GL(n), rng.integers(n-1)) for _ in range(6000)]
good = lambda j: len(set().union(*[set(Aset(*frames[s], n)) for s in range(j*m, (j+1)*m)])) == N-1
a0 = 0.7; U = np.eye(N, dtype=complex); U[0,0] = U[1,1] = np.cos(a0); U[1,0] = np.sin(a0); U[0,1] = -np.sin(a0)
traj = [U]
for t in range(1200): traj.append(step(traj[-1], perm(frames[t][0]), frames[t][1]))
Fs = np.array([F(W) for W in traj]) - lam; ds = np.array([np.linalg.norm(dl(W)) for W in traj])
print("(2) 陷阱轨迹：好块上 (F-λ) 的缩小比例（只看 F-λ <= μ+/2 的块）")
r = [Fs[(j+1)*m]/Fs[j*m] for j in range(1200//m - 1) if good(j) and Fs[j*m] <= mup/2 and Fs[j*m] > 1e-28]
ng = [Fs[(j+1)*m]/Fs[j*m] for j in range(1200//m - 1) if not good(j) and Fs[j*m] > 1e-28]
print(f"  好块 {len(r)} 个，比例最大 {max(r):.3f}；非好块 {len(ng)} 个，比例最大 {max(ng) if ng else float('nan'):.3f}（应 <= 1）")
tt = np.arange(100, 600); print(f"  min(F-λ) = {Fs.min():.1e}（>= 0），log|δ_t| 斜率 {np.polyfit(tt, np.log(ds[tt]), 1)[0]:.4f}")

print("(3) 圆盘引理：第 T 步沿 E^u 扰动")
T = int(np.argmax(ds < 1e-3)); UT = traj[T]; kT = kof(UT); Eu = np.eye(N)[:, Ed < lam].astype(complex)
th0 = 2*mu; print(f"  T = {T}，|δ_T| = {ds[T]:.1e}")
for eps in (1e-5, 1e-8, 1e-11):
    for trial in range(2):
        u = Eu@(rng.normal(size=Eu.shape[1])+1j*rng.normal(size=Eu.shape[1])); u *= eps/np.linalg.norm(u)
        U1, U2 = UT.copy(), sigma(dl(UT)+u)@kT
        Phi0 = -q(dl(U2)-dl(U1)); mn, wB = np.inf, 0; t = T; out = None
        while t < 6000:
            a = dl(U2)-dl(U1); Phi = -q(a)
            if max(np.linalg.norm(dl(U1)), np.linalg.norm(dl(U2))) > 1e-2:
                if out is None: out = t
            if out is None:
                mn = min(mn, Phi/Phi0); wB = max(wB, np.vdot(a,a).real/((8/th0 + 4*eta**2*(t-T)/beta)*Phi))
            c = perm(frames[t][0]); e = frames[t][1]; U1, U2 = step(U1, c, e), step(U2, c, e); t += 1
        print(f"  |u|={eps:.0e}: 离开邻域步 {out-T if out else '无'}，此前 min Φ_t/Φ_T = {mn:.3f}，max |a|²/(L_tΦ_t) = {wB:.3f}；"
              f"6000 步时 F-E0：扰动 {F(U2)-E0:.1e}，陷阱 {F(U1)-E0:.3f}")
