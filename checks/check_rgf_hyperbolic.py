"""定理 G（鞍点处线性化随机映射的双曲性）与引理 5（局部微分同胚）的数值核对。
(1) 线性化 δ -> (I - η P_t 𝓗) δ 的完整 Lyapunov 谱（QR 法，实 2(N-1) 维）：
    定理 G 证明最大指数 > 0；观察：正指数恰好 2k 个（k = 鞍点下面的能级数）。
(2) 精确恒等式 F(ψ) - λ = q(δ)/(1+|δ|²)，δ = ψ/<v,ψ> - v。
(3) 一步更新 T(U) = U c e^{M(U)} c† 的左平移导数的最小奇异值（局部微分同胚）。"""
import numpy as np
from functools import reduce
from scipy.linalg import expm
rng = np.random.default_rng(5)
I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)

def setup(n):
    N = 2**n
    op = lambda P,i: reduce(np.kron, [P if j==i else I2 for j in range(n)])
    H = sum(rng.normal()*op(P,i)@op(Q,i+1) for i in range(n-1) for P in (X,Y,Z) for Q in (X,Y,Z))
    H = H + sum(rng.normal()*op(P,i) for i in range(n) for P in (X,Y,Z))
    return N, (H+H.conj().T)/2

def rand_GL(n):
    while True:
        A = rng.integers(0,2,(n,n))
        if round(abs(np.linalg.det(A))) % 2: return A

def bits(x,n): return np.array([(x >> (n-1-k)) & 1 for k in range(n)])
def idx(b): return int(''.join(map(str, b)), 2)
def Aset(A, e, n):
    out = []
    for ab in (1,2,3):
        b = np.zeros(n,int); b[e], b[e+1] = ab>>1, ab&1; out.append(idx((A@b)%2))
    return out

def realify(M): return np.block([[M.real, -M.imag],[M.imag, M.real]])

def lyap_spectrum(n, Hc, eta, T, acc=False):
    d = Hc.shape[0]; R = realify(Hc); Qm = np.eye(2*d); s = np.zeros(2*d); A = rand_GL(n)
    for t in range(T):
        if acc:
            if rng.random() < 0.5:
                i, j = rng.choice(n, 2, replace=False); E = np.eye(n,dtype=int); E[i,j] = 1; A = (A@E) % 2
        else:
            A = rand_GL(n)
        p = np.zeros(d); p[[y-1 for y in Aset(A, rng.integers(n-1), n)]] = 1
        Qm = Qm - eta*(np.r_[p,p][:,None]*(R@Qm))
        Qm, Rr = np.linalg.qr(Qm); s += np.log(np.abs(np.diag(Rr)))
    return np.sort(s/T)[::-1]

def haar(N):
    q, r = np.linalg.qr(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N))); return q*(np.diag(r)/abs(np.diag(r)))

print("(1) 线性化的 Lyapunov 谱")
for n, T, ks in ((3, 20000, (1,3,5,7)), (4, 5000, (1,7,14))):
    N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); eta = 1/ell; kap = 3/(N-1)
    for k in ks:
        lam = E[k]
        rows = []
        for kind in ("Haar", "本征", "累积"):
            if kind == "本征": U = np.roll(V, -k, axis=1)          # 其余列是本征矢（陷阱型）
            else: U = haar(N); U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
            Hc = 2*(U[:,1:].conj().T@H@U[:,1:] - lam*np.eye(N-1))
            sp = lyap_spectrum(n, Hc, eta, T, acc=(kind=="累积"))
            neg = sp[sp < 0]
            rows.append((kind, int(np.sum(sp > 0)), sp[0], sp[sp>0].min(), neg.max() if neg.size else np.nan))
        print(f" n={n} 鞍点 k={k:2d}（2k={2*k:2d}，维数 {2*(N-1)}）")
        for kind, npos, top, mp, mn in rows:
            print(f"    {kind}补全：正指数 {npos:2d} 个，最大 {top:.4f}，最小正 {mp:.4f}，最大负 {mn:.4f}")

print("(2) 恒等式 F-λ = q(δ)/(1+|δ|²)")
N, H = setup(3); E, V = np.linalg.eigh(H); err = 0
for k in range(N):
    for _ in range(20):
        psi = rng.normal(size=N)+1j*rng.normal(size=N); psi /= np.linalg.norm(psi)
        dl = psi/np.vdot(V[:,k], psi) - V[:,k]
        lhs = np.real(np.vdot(psi, H@psi)) - E[k]; rhs = np.real(np.vdot(dl, (H-E[k]*np.eye(N))@dl))/(1+np.vdot(dl,dl).real)
        err = max(err, abs(lhs-rhs))
print(f"    最大误差 {err:.1e}")

print("(3) 一步更新的导数（n=3，u(8) 上 64 维）")
n = 3; N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); e0 = np.eye(N)[0].astype(complex)
loc = []
for ab in (1,2,3):
    for ph in (1, 1j):
        M = np.zeros((4,4),complex); M[ab,0] = ph; M[0,ab] = -np.conj(ph); loc.append(M)
embed = lambda M,e: reduce(np.kron, [np.eye(2**e), M, np.eye(2**(n-e-2))])
def perm(A):
    P = np.zeros((N,N))
    for x in range(N): P[idx((A@bits(x,n))%2), x] = 1
    return P
def step(U, eta, c, e):
    psi = U[:,0]; F = np.real(np.vdot(psi, H@psi)); g = 2*(H@psi - F*psi)
    x = [-eta*np.real(np.vdot(g, U@c@embed(M,e)@e0)) for M in loc]
    return U @ c @ embed(expm(sum(xj*M for xj,M in zip(x,loc))), e) @ c.T
basis = []
for i in range(N):
    for j in range(i, N):
        B = np.zeros((N,N),complex)
        if i == j: B[i,i] = 1j; basis.append(B)
        else:
            B[i,j], B[j,i] = 1, -1; basis.append(B/np.sqrt(2)); B = np.zeros((N,N),complex); B[i,j] = B[j,i] = 1j; basis.append(B/np.sqrt(2))
coord = lambda W: np.array([np.real(np.trace(B.conj().T@W)) for B in basis])
for fac in (0.5, 1.0, 2.0):
    eta = fac/ell; smin = []
    for trial in range(20):
        U = haar(N) if trial < 15 else np.linalg.qr(np.c_[V[:,3] + 1e-3*rng.normal(size=N), haar(N)[:,1:]])[0]
        c = perm(rand_GL(n)); e = rng.integers(n-1); T0 = step(U, eta, c, e); h = 1e-6
        J = np.array([coord(T0.conj().T@(step(U@expm(h*B), eta, c, e) - step(U@expm(-h*B), eta, c, e))/(2*h)) for B in basis]).T
        smin.append(np.linalg.svd(J, compute_uv=False).min())
    print(f"    η = {fac}/ℓ：20 个点（含 5 个鞍点附近）上最小奇异值的最小值 {min(smin):.3f}")
