"""论文正文新增论断的数值核对（n=3、4）：
(1) 两项参数移位：∂_j = E(π/4) − E(−π/4)，其中 E(θ) = <ψ(θ)|H|ψ(θ)>，ψ(θ) = U c e^{θ M_j} |0>。
(2) 引理 R(a)：好块上 |g_j|² <= C'_m S，C'_m = (N−1)(1 + 5/2 ηℓ√m)²（非线性轨迹，随机 Haar 初态）。
(3) 管状坐标线性化：δ' = A(W)δ + O(|δ|²)，A(W) = I − η Π⊥ P_W 𝓗，其中 Θ(δ,W) = e^{X(δ)} W（左乘）。"""
import numpy as np
from functools import reduce
from scipy.linalg import expm
rng = np.random.default_rng(42)
I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)
def setup(n):
    N = 2**n; op = lambda P,i: reduce(np.kron, [P if j==i else I2 for j in range(n)])
    H = sum(rng.normal()*op(P,i)@op(Q,i+1) for i in range(n-1) for P in (X,Y,Z) for Q in (X,Y,Z))
    H = H + sum(rng.normal()*op(P,i) for i in range(n) for P in (X,Y,Z)); return N, (H+H.conj().T)/2
for n in (3, 4):
    N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0])
    bits = lambda x: np.array([(x>>(n-1-k))&1 for k in range(n)]); idx = lambda b: int(''.join(map(str,b)),2)
    def perm(A):
        P = np.zeros((N,N), complex)
        for x in range(N): P[idx((A@bits(x))%2), x] = 1
        return P
    def rand_GL():
        while True:
            A = rng.integers(0,2,(n,n))
            if round(np.linalg.det(A)) % 2: return A
    def Aset(A, e):
        out = []
        for ab in (1,2,3):
            b = np.zeros(n,int); b[e], b[e+1] = ab>>1, ab&1; out.append(idx((A@b)%2))
        return out
    loc = []
    for ab in (1,2,3):
        for ph in (1,1j):
            M = np.zeros((4,4),complex); M[ab,0] = ph; M[0,ab] = -np.conj(ph); loc.append(M)
    embed = lambda M,e: reduce(np.kron, [np.eye(2**e), M, np.eye(2**(n-e-2))])
    e0 = np.eye(N)[0].astype(complex); Fv = lambda p: np.real(np.vdot(p, H@p))
    def step(U, c, e, eta):
        psi = U[:,0]; g = 2*(H@psi - Fv(psi)*psi)
        x = [-eta*np.real(np.vdot(g, U@c@embed(Mj,e)@e0)) for Mj in loc]
        return U@c@embed(expm(sum(xj*Mj for xj,Mj in zip(x,loc))), e)@c.conj().T
    haar = lambda: np.linalg.qr(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))[0]
    # (1)
    err = 0
    for _ in range(50):
        U = haar(); c = perm(rand_GL()); e = rng.integers(n-1); psi = U[:,0]; g = 2*(H@psi - Fv(psi)*psi)
        for Mj in loc:
            d = np.real(np.vdot(g, U@c@embed(Mj,e)@e0))
            Eth = lambda th: Fv(U@c@embed(expm(th*Mj),e)@e0)
            err = max(err, abs(d - (Eth(np.pi/4) - Eth(-np.pi/4))))
    # (2)
    eta = 0.9/ell; worst = 0
    for _ in range(200):
        U = haar(); S = 0; cov = set(); m = 0; g0 = None
        while len(cov) < N-1:
            A = rand_GL(); e = rng.integers(n-1); cov |= set(Aset(A, e)); m += 1
            psi = U[:,0]; g = 2*(H@psi - Fv(psi)*psi); g0 = g if g0 is None else g0
            P = U[:, Aset(A,e)]@U[:, Aset(A,e)].conj().T; S += np.linalg.norm(P@g)**2
            U = step(U, perm(A), e, eta)
        worst = max(worst, np.linalg.norm(g0)**2/((N-1)*(1+2.5*eta*ell*np.sqrt(m))**2*S))
    # (3)
    k = N//2; v = V[:,k]; lam = E[k]; Hc = 2*(H-lam*np.eye(N)); Pp = np.eye(N) - np.outer(v, v.conj())
    def tube(U):
        a = np.vdot(v, U[:,0]); ph = a/abs(a); r = np.arccos(min(1,abs(a)))
        u = Pp@U[:,0]*np.conj(ph); d = (r/np.sin(r) if r > 1e-14 else 1.0)*u
        X_ = np.outer(d, v.conj()) - np.outer(v, d.conj())
        return d, expm(-X_)@U
    res = []
    for s in (1e-2, 1e-3, 1e-4):
        wr = 0
        for _ in range(20):
            W = haar(); W[:,0] = v; W = np.linalg.qr(W)[0]; W = W*(np.vdot(W[:,0], v)/abs(np.vdot(W[:,0], v)))
            d = Pp@(rng.normal(size=N)+1j*rng.normal(size=N)); d *= s/np.linalg.norm(d)
            U = expm(np.outer(d, v.conj()) - np.outer(v, d.conj()))@W
            A = rand_GL(); e = rng.integers(n-1); ys = Aset(A, e); PW = W[:, ys]@W[:, ys].conj().T
            d2, _ = tube(step(U, perm(A), e, eta))
            wr = max(wr, np.linalg.norm(d2 - (d - eta*Pp@PW@Hc@d))/s**2)
        res.append(wr)
    print(f"n={n}: (1) 移位规则最大误差 {err:.1e}；(2) max |g|²/(C'_m S) = {worst:.3f}（<=1）；"
          f"(3) 余项/|δ|² = " + ", ".join(f"{x:.3f}" for x in res) + "（|δ| = 1e-2, 1e-3, 1e-4，应有界）")
