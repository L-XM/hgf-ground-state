"""定理 E（iid / 累积标架）证明里几个新引理的数值核对。n=3，随机 2-local H，η = 0.5/ℓ。
(R) 开口恢复引理（线性）：δ_0 在闭负锥 {q<=0} 里，经过一个好块（长 m）后
    -q(δ_m)/|δ_m|² >= θ*_m = (2βσ²/C_m)(1+ηℓ)^{-2m}，σ = 𝓗 的最小 |本征值|。
(P) 成对轨迹的 q 单调（非线性）：两条轨迹用同一串标架，管状坐标 U = z e^{X(δ)}（z ∈ Fix）。
    起初 Δδ 在不稳定锥里，Φ_t = -q_{z_t}(Δδ_t) 在离开鞍点邻域之前应不减（允许 O(r) 的相对误差）。
(D) 累积标架（懒游走）：从任何标架出发，接下来 m 步是好块的概率，对起点取最小值。"""
import numpy as np, itertools
from functools import reduce
from scipy.linalg import expm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(13)
n = 3; N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0])
eta = 0.5/ell; beta = eta*(1-eta*ell/2); L = eta*ell
Hz = lambda z, lam: 2*(z[:,1:].conj().T@H@z[:,1:] - lam*np.eye(N-1))
q = lambda Hc, d: np.real(np.vdot(d, Hc@d))

print("(R) 开口恢复：min (-q(δ_m)/|δ_m|²) / θ*_m（应 >= 1）")
for k in (1, 3, 5):
    worst = np.inf
    for trial in range(400):
        z = haar(N); z[:,0] = V[:,k]; z = np.linalg.qr(z)[0]; Hc = Hz(z, E[k])
        w, Q = np.linalg.eigh(Hc); sig = np.min(abs(w))
        neg, pos = Q[:, w < 0], Q[:, w > 0]
        a = neg@(rng.normal(size=neg.shape[1])+1j*rng.normal(size=neg.shape[1]))
        b = pos@(rng.normal(size=pos.shape[1])+1j*rng.normal(size=pos.shape[1]))
        tau = rng.choice([0.0, rng.random()])          # tau=0：正好在锥边界 q=0 上
        # 取 s 使 q(a + s b) = -tau |a|^2 q(a)/q(a)... 直接解：q(a)+s²q(b) = tau*q(a)
        s = np.sqrt((tau-1)*q(Hc,a)/q(Hc,b)); d = a + s*b
        cov, m = set(), 0; d0 = d.copy()
        while len(cov) < N-1:
            ys = [y-1 for y in Aset(rand_GL(n), rng.integers(n-1), n)]; cov |= set(ys); m += 1
            p = np.zeros(N-1); p[ys] = 1; d = d - eta*p*(Hc@d)
        Cm = (N-1)*(1+L*np.sqrt(m))**2; th = 2*beta*sig**2/Cm*(1+L)**(-2*m)
        worst = min(worst, (-q(Hc,d)/np.vdot(d,d).real)/th)
    print(f"  k={k}: {worst:.2f}")

print("(P) 成对轨迹：Φ_t/Φ_0 的最小值、离开邻域时的值、期间的好块数")
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
def step(U, c, e):
    psi = U[:,0]; Fv = np.real(np.vdot(psi, H@psi)); g = 2*(H@psi - Fv*psi)
    x = [-eta*np.real(np.vdot(g, U@c@embed(Mj,e)@e0)) for Mj in loc]
    return U @ c @ embed(expm(sum(xj*Mj for xj,Mj in zip(x,loc))), e) @ c.conj().T
def Xof(d):
    X = np.zeros((N,N), complex); X[1:,0] = d; X[0,1:] = -d.conj(); return X
def tube(U, v):
    """U = z e^{X(δ)}，z|0> ∝ v。闭式：a = U†v，cos|δ| = |a_0|"""
    a = U.conj().T@v; ph = a[0]/abs(a[0]); r = np.arccos(min(1, abs(a[0])))
    d = -(r/np.sin(r) if r > 1e-14 else 1.0)*a[1:]*np.conj(ph)
    return U@expm(-Xof(d)), d
for k in (1, 3, 5):
    v = V[:,k]; out = []
    for trial in range(4):
        z = haar(N); z[:,0] = v; z = np.linalg.qr(z)[0]
        Hc = Hz(z, E[k]); w, Q = np.linalg.eigh(Hc)
        d0 = 1e-6*(rng.normal(size=N-1)+1j*rng.normal(size=N-1))
        Dd = 1e-8*Q[:, 0]*np.exp(2j*np.pi*rng.random())          # 最负的本征方向
        Y = np.zeros((N,N), complex); B = rng.normal(size=(N-1,N-1))+1j*rng.normal(size=(N-1,N-1))
        Y[1:,1:] = 1e-8*(B - B.conj().T)/2                       # 切向（隐藏列）扰动，Y|0> = 0
        U1 = z@expm(Xof(d0)); U2 = z@expm(Y)@expm(Xof(d0+Dd))
        Phi0 = None; ratios = []; good = 0; cov = set(); t = 0
        while t < 4000:
            z1, d1 = tube(U1, v); z2, d2 = tube(U2, v)
            r = max(np.linalg.norm(d1), np.linalg.norm(d2))
            if r > 1e-2: break
            Phi = -q(Hz(z1, E[k]), d1 - d2)
            if Phi0 is None: Phi0 = Phi
            ratios.append(Phi/Phi0)
            A = rand_GL(n); e = rng.integers(n-1); cov |= set(Aset(A, e, n))
            if len(cov) == N-1: good += 1; cov = set()
            c = perm(A); U1 = step(U1, c, e); U2 = step(U2, c, e); t += 1
        out.append((min(ratios), ratios[-1], good, t))
    print(f"  k={k}: " + "；".join(f"min {a:.3f}, 终值 {b:.1e}, 好块 {g}, 步数 {t}" for a,b,g,t in out))

print("(D) 懒游走：min_A Pr(接下来 m 步覆盖全部非零串)")
GL = [np.array(A).reshape(n,n) for A in itertools.product([0,1], repeat=n*n) if round(abs(np.linalg.det(np.array(A).reshape(n,n)))) % 2]
moves = [np.eye(n,dtype=int)] + [(np.eye(n,dtype=int)+np.eye(n,dtype=int)[:,[i]]@np.eye(n,dtype=int)[[j],:]) % 2 for i in range(n) for j in range(n) if i != j]
for m in (6, 12):
    probs = []
    for A0 in GL:
        hit = 0
        for _ in range(300):
            A = A0.copy(); cov = set()
            for s in range(m):
                A = (A@(moves[0] if rng.random() < 0.5 else moves[1+rng.integers(len(moves)-1)])) % 2
                cov |= set(Aset(A, rng.integers(n-1), n))
            hit += len(cov) == N-1
        probs.append(hit/300)
    print(f"  m={m}: min {min(probs):.3f}, 平均 {np.mean(probs):.3f}（168 个起点）")
