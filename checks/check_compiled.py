"""汇编文档（special01-results-compiled.md）里新写法的数值核对。n=3，随机 2-local H，iid CNOT 标架。
(1) 梯度的 Lipschitz 常数：|g(ψ)-g(ψ')| <= 4(λmax-λmin)|ψ-ψ'|。
(2) 反向块不等式：好块上 q(δ_0) >= q(δ_m)/(1-4βμ+/C_m)，只要 q(δ_m) > 0。
(3) 奇异值界（定理 G、G′ 的 Courant–Fischer 写法），G_t = [0,t) 里对齐好块数：
    σ_{2k}(Φ_t) >= sqrt(2μ/ℓ) ρ^{G_t/2}，σ_{2k+1}(Φ_t) <= sqrt(ℓ/(2μ+)) (1-4βμ+/C_m)^{G_t/2}。"""
import numpy as np
from scipy.linalg import expm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(29)
n = 3; N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); w0 = E[-1]-E[0]

print("(1) max |g-g'| / (4(λmax-λmin)|ψ-ψ'|)（应 <= 1）")
def g(p): F = np.real(np.vdot(p, H@p)); return 2*(H@p - F*p)
worst = 0
for _ in range(4000):
    p = rng.normal(size=N)+1j*rng.normal(size=N); p /= np.linalg.norm(p)
    d = rng.normal(size=N)+1j*rng.normal(size=N); p2 = p + 10**rng.uniform(-6, 0)*d; p2 /= np.linalg.norm(p2)
    worst = max(worst, np.linalg.norm(g(p)-g(p2))/(4*w0*np.linalg.norm(p-p2)))
print(f"  {worst:.3f}")

eta = 0.5/ell; beta = eta*(1-eta*ell/2)
def frame_p():
    p = np.zeros(N-1); ys = [y-1 for y in Aset(rand_GL(n), rng.integers(n-1), n)]; p[ys] = 1; return p, ys
print("(2) 反向块不等式：min [q(δ_0)/q(δ_m)]·(1-4βμ+/C_m)（应 >= 1）")
for k in (1, 3, 5):
    mup = E[k+1]-E[k]; worst = np.inf
    for _ in range(300):
        U = haar(N); U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
        Hc = 2*(U[:,1:].conj().T@H@U[:,1:] - E[k]*np.eye(N-1)); q = lambda x: np.vdot(x, Hc@x).real
        d = rng.normal(size=N-1)+1j*rng.normal(size=N-1); cov = set(); ds = [d]
        while len(cov) < N-1:
            p, ys = frame_p(); cov |= set(ys); d = d - eta*p*(Hc@d); ds.append(d)
        m = len(ds)-1; Cm = (N-1)*(1+eta*ell*np.sqrt(m))**2
        if q(ds[-1]) > 0: worst = min(worst, q(ds[0])/q(ds[-1])*(1-4*beta*mup/Cm))
    print(f"  k={k}: {worst:.3f}")

print("(3) 奇异值界，m=12，t=120：σ_{2k}/下界 的最小值（>=1），σ_{2k+1}/上界 的最大值（<=1）")
m = 12; T = 120   # T 再大，SVD 的舍入误差（cond ~ e^{2λ1 T}）会淹没小奇异值
for k in (1, 3, 6):
    mu, mup = E[k]-E[k-1], E[k+1]-E[k]; lo, hi = np.inf, 0
    for _ in range(20):
        U = haar(N); U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
        R = realify(2*(U[:,1:].conj().T@H@U[:,1:] - E[k]*np.eye(N-1)))
        Phi = np.eye(2*(N-1)); G = 0
        for j in range(T//m):
            cov = set()
            for s in range(m):
                p, ys = frame_p(); cov |= set(ys); Phi = (np.eye(2*(N-1)) - eta*np.r_[p,p][:,None]*R)@Phi
            G += len(cov) == N-1
        Cm = (N-1)*(1+eta*ell*np.sqrt(m))**2; rho = 1+4*beta*mu/Cm; rhop = 1/(1-4*beta*mup/Cm)
        sv = np.linalg.svd(Phi, compute_uv=False)
        lo = min(lo, sv[2*k-1]/(np.sqrt(2*mu/ell)*rho**(G/2)))
        hi = max(hi, sv[2*k]/(np.sqrt(ell/(2*mup))*rhop**(-G/2)))
    print(f"  k={k}: {lo:.3e}, {hi:.3e}")

print("(4) 管状坐标恒等式 F-λ = ½ sinc²|δ| q(δ)，以及不等式 (★)；H 有一个二重简并能级")
Hd = H.copy(); Ed = E.copy(); Ed[3] = Ed[4] = (E[3]+E[4])/2; Hd = V@np.diag(Ed)@V.conj().T
lam = Ed[3]; Vl = V[:, [3,4]]; Pperp = np.eye(N) - Vl@Vl.conj().T; Hc = 2*(Hd - lam*np.eye(N))
q = lambda x: np.vdot(x, Hc@x).real; ellD = 2*(Ed[-1]-Ed[0]); etaD = 0.9/ellD; betaD = etaD*(1-etaD*ellD/2)
e1, e2 = 0, np.inf
for _ in range(500):
    v = Vl@(rng.normal(size=2)+1j*rng.normal(size=2)); v /= np.linalg.norm(v)
    W = haar(N); W[:,0] = v; W = np.linalg.qr(W)[0]; W = W*(np.vdot(W[:,0], v)/abs(np.vdot(W[:,0], v)))
    d = Pperp@(rng.normal(size=N)+1j*rng.normal(size=N)); d *= rng.uniform(0.01, 1.2)/np.linalg.norm(d)
    X = np.outer(d, v.conj()) - np.outer(v, d.conj()); psi = (expm(X)@W)[:,0]; r = np.linalg.norm(d)
    e1 = max(e1, abs(np.vdot(psi, Hd@psi).real - lam - 0.5*(np.sin(r)/r)**2*q(d)))
    ys = Aset(rand_GL(n), rng.integers(n-1), n); PW = W[:, ys]@W[:, ys].conj().T
    w = PW@(Hc@d); Ad = d - etaD*Pperp@w
    e2 = min(e2, q(d) - 2*betaD*np.vdot(w,w).real - q(Ad))
print(f"  恒等式最大误差 {e1:.1e}；(★) 余量 q(δ)-2β|w|²-q(Aδ) 的最小值 {e2:.2e}（应 >= 0）")
