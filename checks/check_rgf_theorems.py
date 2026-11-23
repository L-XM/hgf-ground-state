"""方向 1 的定理核对（输入侧局部 SU(4)/U(3) 水平门 + 随机 CNOT 标架），统一用 CP^{N-1} 写法：
度量 <v,w> = Re<v|w>（v,w ⊥ ψ），grad F = 2(H-F)ψ，一步 ψ' = Exp_ψ(-η P g)。
生成元 M = ph|ab><00| - h.c.（ph∈{1,i}，ab∈{01,10,11}），M|0> 两两正交且长度 1。
(a) 引理 1：一步恰好是测地线步      (b) 引理 2：ℓ=2(λmax-λmin) 与必然下降
(c) 引理 3：Pr(x∈A)=3/(N-1)（n=3,4 精确枚举）
(d) 定理 2：速率                       (e) 命题 5：对角 H 的陷阱
(f) 命题 4：转置游走的混合             (g) Malvetti Lemma 3.9 Case II 反例
(h) 鞍点逃逸（猜想 6 的数值证据）"""
import numpy as np, itertools
from functools import reduce
from scipy.linalg import expm
rng = np.random.default_rng(2)
I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)

def gl_all(n):
    out = []
    for v in range(2**(n*n)):
        A = np.array([(v>>k)&1 for k in range(n*n)]).reshape(n,n)
        if round(np.linalg.det(A)) % 2: out.append(A)
    return out

class Sys:
    def __init__(s, n):
        s.n = n; s.N = 2**n
        s.bits = lambda x: np.array([(x>>(n-1-k))&1 for k in range(n)])
        s.idx = lambda b: int(''.join(map(str,b)),2)
        s.edges = list(range(n-1))
        s.trans = [(np.eye(n,dtype=int) + np.outer(np.eye(n,dtype=int)[i], np.eye(n,dtype=int)[j]))%2 for i in range(n) for j in range(n) if i!=j]
        s.e0 = np.zeros(s.N,complex); s.e0[0] = 1
        s.loc = []
        for ab in range(1,4):
            for ph in (1,1j):
                M = np.zeros((4,4),complex); M[ab,0] = ph; M[0,ab] = -np.conj(ph); s.loc.append(M)
    def op(s,P,i): return reduce(np.kron,[P if j==i else I2 for j in range(s.n)])
    def embed(s,M,e): return reduce(np.kron,[np.eye(2**e),M,np.eye(2**(s.n-e-2))])
    def perm(s,A):
        P = np.zeros((s.N,s.N))
        for x in range(s.N): P[s.idx((A@s.bits(x))%2), x] = 1
        return P
    def rand_GL(s):
        while True:
            A = rng.integers(0,2,(s.n,s.n))
            if round(np.linalg.det(A)) % 2: return A
    def heis(s): return sum(s.op(P,i)@s.op(P,(i+1)%s.n) for i in range(s.n) for P in (X,Y,Z))/4
    def rand2(s):
        n = s.n
        H = sum(rng.normal()*s.op(P,i)@s.op(Q,i+1) for i in range(n-1) for P in (X,Y,Z) for Q in (X,Y,Z))
        return (H + sum(rng.normal()*s.op(P,i) for i in range(n) for P in (X,Y,Z)))/4
    def haar(s): return np.linalg.qr(rng.normal(size=(s.N,s.N))+1j*rng.normal(size=(s.N,s.N)))[0]
    def step(s, U, H, eta, c, e, which=None):
        """返回 (U', Pg, g)。x_j = -η ∂_j f(0) = -η <g, U c M_j|0>>"""
        psi = U[:,0]; F = np.real(psi.conj()@H@psi); g = 2*(H@psi - F*psi)
        js = range(6) if which is None else [which]
        m = {j: U@(c@(s.embed(s.loc[j],e)@s.e0)) for j in js}
        x = {j: -eta*np.real(np.vdot(g, m[j])) for j in js}
        Pg = -sum(x[j]*m[j] for j in js)/eta
        return U@c@s.embed(expm(sum(x[j]*s.loc[j] for j in js)),e)@c.T, Pg, g
F_ = lambda U,H: np.real(U[:,0].conj()@H@U[:,0])

S = Sys(4); N = S.N
H = S.rand2(); E = np.linalg.eigvalsh(H); ell = 2*(E[-1]-E[0]); eta = 1/ell; kap = 3/(N-1)

# (a)(b)
errs, slack, curv = [], [], []
U = S.haar()
for t in range(400):
    c = S.perm(S.rand_GL()); e = rng.integers(3)
    U2, Pg, g = S.step(U,H,eta,c,e); psi = U[:,0]; r = eta*np.linalg.norm(Pg)
    geo = np.cos(r)*psi - np.sin(r)*Pg/np.linalg.norm(Pg)
    errs.append(1 - abs(np.vdot(geo, U2[:,0])))
    slack.append(F_(U,H) - F_(U2,H) - eta/2*np.linalg.norm(Pg)**2)
    U = U2
for _ in range(300):
    psi = S.haar()[:,0]; v = rng.normal(size=N)+1j*rng.normal(size=N); v -= np.vdot(psi,v)*psi; v /= np.linalg.norm(v)
    ss = np.linspace(0,np.pi,401); Fs = np.array([np.real(np.vdot(q:=np.cos(a)*psi+np.sin(a)*v, H@q)) for a in ss])
    curv.append(np.max(np.abs(np.diff(Fs,2)))/(ss[1]-ss[0])**2)
print(f"(a) 一步与测地线 cos r ψ - sin r Pg/|Pg| 的最大偏差 1-|<.,.>| = {max(errs):.1e}")
print(f"(b) max|F''| 沿随机测地线 = {max(curv):.4f} ≤ ℓ = {ell:.4f}；必然下降余量 min[F-F'-(η/2)|Pg|²] = {min(slack):.2e}（≥0）")

# (c)
for n in (3,4):
    Sn = Sys(n); G = gl_all(n); cnt = np.zeros(2**n)
    for A in G:
        for e in Sn.edges:
            for ab in (1,2,3):
                b = np.zeros(n,dtype=int); b[e] = ab>>1; b[e+1] = ab&1
                cnt[Sn.idx((A@b)%2)] += 1
    p = cnt/(len(G)*len(Sn.edges))
    print(f"(c) n={n}, |GL|={len(G)}：Pr(x∈A) 对 x≠0 ∈ [{p[1:].min():.6f}, {p[1:].max():.6f}]，Pr(0∈A)={p[0]}，3/(N-1)={3/(2**n-1):.6f}")

# (d)
H = S.heis(); E = np.linalg.eigvalsh(H); E0 = E[0]; Delta = E[E>E0+1e-9][0]-E0
ell = 2*(E[-1]-E[0]); eta = 1/ell; rho = 2*eta*(1-ell*eta/2)*kap*Delta; rho_as = 4*eta*(1-ell*eta/2)*kap*Delta
sl = []
for sd in range(6):
    U = S.haar(); d = []
    for t in range(4000):
        U,_,_ = S.step(U,H,eta,S.perm(S.rand_GL()),rng.integers(3)); d.append(F_(U,H)-E0)
    d = np.array(d); tt = np.where((d<Delta/2)&(d>1e-11))[0]; sl.append(np.polyfit(tt,np.log(d[tt]),1)[0])
print(f"(d) Heisenberg 环 Δ={Delta}，η=1/ℓ={eta:.4f}：实测斜率 {np.round(sl,4)}")
print(f"    定理 2 保证 log(1-ρ) = {np.log(1-rho):.4f}，δ→0 时的保证 log(1-4η(1-ℓη/2)κΔ) = {np.log(1-rho_as):.4f}，一阶预测 log(1-4ηκΔ) = {np.log(1-4*eta*kap*Delta):.4f}")

# (e) 陷阱：H = Σ Z_i Z_{i+1}（开链），ψ0 = cos a|0000> + sin a|0001>
H = sum(S.op(Z,i)@S.op(Z,i+1) for i in range(3)).real.astype(complex); diag = np.real(np.diag(H))
ell = 2*(diag.max()-diag.min()); eta = 1/ell; x1 = 1
for mode in ('iid','acc'):
    finals, leaks = [], []
    for sd in range(10):
        a = rng.uniform(0.2,1.3); U = np.eye(N,dtype=complex)
        U[[0,x1],[0,x1]] = np.cos(a); U[x1,0] = np.sin(a); U[0,x1] = -np.sin(a)
        A = np.eye(4,dtype=int); leak = 0
        for t in range(3000):
            if mode=='iid': A = S.rand_GL()
            else: A = (A@S.trans[rng.integers(len(S.trans))])%2
            U,_,_ = S.step(U,H,eta,S.perm(A),rng.integers(3))
            leak = max(leak, 1-abs(U[0,0])**2-abs(U[x1,0])**2)
        finals.append(F_(U,H)); leaks.append(leak)
    print(f"(e) {mode}: 终点能量 {np.round(finals,6)}（基态 {diag.min()}，|0001> 能量 {diag[x1]}），漏出 span 的最大权重 {max(leaks):.1e}")

# (f) 转置随机游走
for n in (2,3,4):
    G = gl_all(n); key = lambda A: int(''.join(map(str,A.ravel())),2); pos = {key(A):i for i,A in enumerate(G)}
    tr = [(np.eye(n,dtype=int)+np.outer(np.eye(n,dtype=int)[i],np.eye(n,dtype=int)[j]))%2 for i in range(n) for j in range(n) if i!=j]
    nbr = np.array([[pos[key((A@T)%2)] for T in tr] for A in G])
    dist = np.zeros(len(G)); dist[pos[key(np.eye(n,dtype=int))]] = 1; first = None; tv = []
    for m in range(1,41):
        new = np.zeros(len(G))
        for k in range(len(tr)): np.add.at(new, nbr[:,k], dist/len(tr))
        dist = new; tv.append(0.5*np.abs(dist-1/len(G)).sum())
        if first is None and dist.min() > 0: first = (m, dist.min()*len(G))
    print(f"(f) n={n}, |GL|={len(G)}, {len(tr)} 个转置：P^m 全正的最小 m 与 β=|G|·min = {first}；TV 距离 m=5,10,20,40: {[round(tv[i],4) for i in (4,9,19,39)]}")

# (g)
Lam = np.diag([2.,-2.]); dirs = [np.array([np.cos(t),np.sin(t)]) for t in (np.pi/8,np.pi/4,3*np.pi/8)]
theta = lambda w: np.arctan2(w[1]**2, w[0]**2)
for eta in (0.1, 0.5):
    w = np.array([1.,0.3]); dth = [theta(w-eta*(u@Lam@w)*u)-theta(w) for u in dirs]
    print(f"(g) f=x²-y²（ℓ=2），η={eta}，w=(1,0.3)，θ={theta(w):.4f}：三个方向的 Δθ = {np.round(dth,4)}")

# (h) 鞍点逃逸：激发本征态 + 1e-4 扰动，标架其余列取 Haar
def escape(H, mode, trials=10, T=8000):
    E, V = np.linalg.eigh(H); lev = np.unique(np.round(E,9)); ell = 2*(E[-1]-E[0]); eta = 1/ell; res = []
    for j in range(1,len(lev)-1):
        vec = V[:, np.argmin(abs(E-lev[j]))]; ok = []
        for tr in range(trials):
            psi = vec + 1e-4*(rng.normal(size=N)+1j*rng.normal(size=N)); psi /= np.linalg.norm(psi)
            W = S.haar(); W[:,0] = psi; U = np.linalg.qr(W)[0]
            A = np.eye(4,dtype=int)
            for t in range(T):
                A = S.rand_GL() if mode=='iid' else (A@S.trans[rng.integers(len(S.trans))])%2
                U,_,_ = S.step(U,H,eta,S.perm(A),rng.integers(3))
                if F_(U,H) < lev[j]-0.1*(lev[j]-lev[j-1]): ok.append(t); break
        res.append(f"{lev[j]:+.2f}:{len(ok)}/{trials}")
    return ' '.join(res)
for name,H in [("Heisenberg 环", S.heis()), ("随机 2-local", S.rand2())]:
    for mode in ('iid','acc'): print(f"(h) {name} {mode}：", escape(H,mode))
