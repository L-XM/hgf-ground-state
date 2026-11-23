"""猜想 E 各部分的数值核对（n=3，随机 2-local H，iid CNOT 标架）。
(G) 定理 G：线性化 δ -> (I - η P_t 𝓗) δ 至少有 2k 个 Lyapunov 指数 >= γ̃（已证明），
    γ̃ = (p_m/(2m)) log(1 + 4βμ/C_m)，μ = λ - (下面最近的能级)，C_m = (N-1)(1+ηℓ√m)²，
    p_m = Pr(m 步 iid 标架覆盖全部非零串)。其余 2(N-1-k) 个为负是观察，没有证明。
(L5) 引理 5：‖DT(X) - Ad X‖_op <= ηℓ ‖X‖_op（于是 η < 1/ℓ 时 DT 处处可逆）。
(H) 引理 H：收敛到激发态的轨迹（命题 C 的陷阱）F-λ、|δ| 指数收敛，U_t 收敛。"""
import numpy as np
from functools import reduce
from scipy.linalg import expm
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(7)
n = 3; N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); kap = 3/(N-1)

print("(G) Lyapunov 谱的符号和定理 G 的一致下界 γ̃，η = 0.5/ℓ")
eta = 0.5/ell; beta = eta*(1-eta*ell/2); m = 12
def covers(m):
    cov = set()
    for _ in range(m): cov |= set(Aset(rand_GL(n), rng.integers(n-1), n))
    return len(cov) == N-1
p_m = np.mean([covers(m) for _ in range(20000)]); Cm = (N-1)*(1+eta*ell*np.sqrt(m))**2
print(f"  m = {m}，p_m ≈ {p_m:.3f}，C_m = {Cm:.1f}")
for k in (1, 3, 5, 6):
    lam = E[k]; mu = lam - E[k-1]
    gt = p_m/(2*m)*np.log(1+4*beta*mu/Cm)
    for kind in ("Haar", "本征"):
        if kind == "本征": U = np.roll(V, -k, axis=1)
        else: U = haar(N); U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
        Hc = 2*(U[:,1:].conj().T@H@U[:,1:] - lam*np.eye(N-1))
        sp = lyap_spectrum(n, Hc, eta, 20000)
        pos, neg = sp[sp > 0], sp[sp < 0]
        print(f"  k={k} {kind}补全：正 {len(pos):2d} 个、负 {len(neg):2d} 个（2k={2*k}）；最小正 {pos.min():.4f}，最大负 {neg.max():.4f}；"
              f"γ̃ = {gt:.1e}，第 2k 大的指数 >= γ̃: {bool(sp[2*k-1] >= gt)}")

print("(L5) ‖DT(X) - Ad X‖_op / (ηℓ‖X‖_op) 的最大值（应 <= 1）")
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
def Mgen(U, H, eta, c, e):
    psi = U[:,0]; F = np.real(np.vdot(psi, H@psi)); g = 2*(H@psi - F*psi)
    return sum(-eta*np.real(np.vdot(g, U@c@embed(Mj,e)@e0))*Mj for Mj in loc)
def step(U, H, eta, c, e):
    return U @ c @ embed(expm(Mgen(U, H, eta, c, e)), e) @ c.conj().T
op = lambda A: np.linalg.norm(A, 2)
worst = 0
for fac in (0.3, 0.9):
    eta = fac/ell
    for trial in range(30):
        U = haar(N); c = perm(rand_GL(n)); e = rng.integers(n-1); T0 = step(U, H, eta, c, e)
        W = c @ embed(expm(Mgen(U, H, eta, c, e)), e) @ c.conj().T
        X = rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)); X = (X - X.conj().T)/2; h = 1e-6
        DT = T0.conj().T @ (step(U@expm(h*X), H, eta, c, e) - step(U@expm(-h*X), H, eta, c, e))/(2*h)
        worst = max(worst, op(DT - W.conj().T@X@W)/(eta*ell*op(X)))
print(f"  最大比值 {worst:.3f}")

print("(H) 陷阱轨迹：对角 H，U_0 把 span{|000>,|001>} 映到 span{|000>,|001>}")
Hd = sum(reduce(np.kron, [Z if j in (i,i+1) else I2 for j in range(n)]) for i in range(n-1))
Ed = np.diag(Hd).real; lam = Ed[1]                    # |001> 的能量：激发态
eta = 0.5/(2*(Ed.max()-Ed.min())); a = 0.7
U = np.eye(N, dtype=complex); U[0,0] = U[1,1] = np.cos(a); U[1,0] = np.sin(a); U[0,1] = -np.sin(a)
Fs, ds, Us = [], [], []
for t in range(600):
    U = step(U, Hd, eta, perm(rand_GL(n)), rng.integers(n-1))
    psi = U[:,0]; Fs.append(np.real(np.vdot(psi, Hd@psi)) - lam); ds.append(abs(psi[0]/psi[1])); Us.append(U.copy())
Fs, ds = np.array(Fs), np.array(ds); t = np.arange(50, 400)
print(f"  min(F-λ) = {Fs.min():.1e}（应 >= 0）；log(F-λ) 斜率 {np.polyfit(t, np.log(Fs[t]), 1)[0]:.4f}；"
      f"log|δ| 斜率 {np.polyfit(t, np.log(ds[t]), 1)[0]:.4f}")
r = Fs[300:]*(1+ds[300:]**2)/ds[300:]**2
print(f"  t>=300 时 (F-λ)(1+|δ|²)/|δ|² 的最小值 {r.min():.3f}（严格正锥：应远离 0）")
print(f"  ‖U_599 - U_300‖ = {op(Us[599]-Us[300]):.1e}（U_t 收敛）")
