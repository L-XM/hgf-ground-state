"""定理 E（iid）证明的数值核对：内蕴坐标 δ = ψ/<v,ψ> - v（v 是激发本征矢），坐标 U = σ(δ) k，k ∈ Stab|0>。
(a) 引理 P 的一步展开：δ' = δ - η P(U) 𝓗 δ + O(|δ|²)，P(U) = 到 span{Π_⊥ U|y> : y ∈ 𝒜_t} 的正交投影，𝓗 = 2(H-λ)。
    隐藏列漂移 |k' - k| = O(|δ|)（一阶，不是二阶）。
(b) 同一圆盘上的两点（δ_s、k 相同，只有 δ_u 不同）：Φ_t = -q(Δδ_t) 在离开邻域前几乎不减；
    |Δk_t| / Σ_{s<t}|Δδ_s| 有界。n=3，随机 2-local H，η = 0.5/ℓ。"""
import numpy as np
from functools import reduce
from scipy.linalg import expm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(17)
n = 3; N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); eta = 0.5/ell
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
for k in (1, 3, 5):
    v = V[:,k]; lam = E[k]; Hc = 2*(H - lam*np.eye(N)); Pp = np.eye(N) - np.outer(v, v.conj())
    z = haar(N); z[:,0] = v; z = np.linalg.qr(z)[0]; z[:,0] = v
    Eu, Es = V[:, :k], V[:, k+1:]
    dl = lambda U: U[:,0]/np.vdot(v, U[:,0]) - v
    def sigma(d):
        a = np.linalg.norm(d)
        return z if a == 0 else z@expm(Xof(np.arctan(a)*(z[:,1:].conj().T@d)/a))
    def kof(U): return sigma(dl(U)).conj().T@U
    q = lambda d: np.real(np.vdot(d, Hc@d))
    # (a)
    res, drift = [], []
    for s in (1e-2, 1e-3, 1e-4):
        wr, wd = 0, 0
        for _ in range(20):
            d = Pp@(rng.normal(size=N)+1j*rng.normal(size=N)); d *= s/np.linalg.norm(d)
            kk = haar(N-1); K = np.eye(N, dtype=complex); K[0,0] = np.exp(2j*np.pi*rng.random()); K[1:,1:] = kk
            U = sigma(d)@K; A = rand_GL(n); e = rng.integers(n-1)
            B = np.array([Pp@U[:,y] for y in Aset(A, e, n)]).T; Q, _ = np.linalg.qr(B); P = Q@Q.conj().T
            U2 = step(U, perm(A), e)
            wr = max(wr, np.linalg.norm(dl(U2) - (d - eta*P@(Hc@d)))/s**2)
            wd = max(wd, np.linalg.norm(kof(U2) - kof(U), 2)/s)
        res.append(wr); drift.append(wd)
    print(f"k={k} (a) 余项/|δ|² = " + ", ".join(f"{x:.3f}" for x in res) + "（|δ| = 1e-2, 1e-3, 1e-4）；"
          f"|Δk|/|δ| = " + ", ".join(f"{x:.3f}" for x in drift))
    # (b)
    out = []
    for ds_size, du_size, gap in ((1e-6, 1e-6, 1e-7), (1e-4, 1e-8, 1e-9), (1e-4, 1e-6, 1e-10)):
        cs = Es@(rng.normal(size=Es.shape[1])+1j*rng.normal(size=Es.shape[1])); cs *= ds_size/np.linalg.norm(cs)
        cu = Eu@(rng.normal(size=k)+1j*rng.normal(size=k)); cu *= du_size/np.linalg.norm(cu)
        hu = Eu@(rng.normal(size=k)+1j*rng.normal(size=k)); hu *= gap/np.linalg.norm(hu)
        K = np.eye(N, dtype=complex); K[0,0] = np.exp(2j*np.pi*rng.random()); K[1:,1:] = haar(N-1)
        U1, U2 = sigma(cs+cu)@K, sigma(cs+cu+hu)@K
        Phi0 = q(dl(U1)-dl(U2)); mn = 1.0; sumd = 0; worstk = 0; t = 0
        while t < 6000:
            D = dl(U1) - dl(U2); r = max(np.linalg.norm(dl(U1)), np.linalg.norm(dl(U2)))
            if r > 1e-2: break
            mn = min(mn, -q(D)/(-Phi0)); sumd += np.linalg.norm(D)
            A = rand_GL(n); e = rng.integers(n-1); c = perm(A); U1, U2 = step(U1, c, e), step(U2, c, e); t += 1
            worstk = max(worstk, np.linalg.norm(kof(U1)-kof(U2), 2)/sumd)
        out.append(f"|δ_s|={ds_size:.0e},|Δδ_0|={gap:.0e}: min Φ_t/Φ_0 {mn:.3f}, 终值 {-q(dl(U1)-dl(U2))/(-Phi0):.1e}, max|Δk|/Σ|Δδ| {worstk:.3f}, 步数 {t}")
    print(f"     (b) " + "\n         ".join(out))
