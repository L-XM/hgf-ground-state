"""方向 1 定理的数值核对（输入侧局部 SU(4)/U(3) 水平门 + 随机 CNOT 标架）。
(1) 引理 1：f(U e^{sD}) = f(U e^{sD_m})（D 是套了标架的局部生成元，D_m 是它的水平部分）
(2) 引理 3：每步平均下降 = 3/(N-1) × 全梯度一步（一阶），对随机 U 检验
(3) 定理 3：进入 δ ≤ Δ/2 后，log δ_t 的斜率不慢于 log(1-2κΔ)
(4) 鞍点逃逸：从激发本征态 + 1e-4 扰动出发，i.i.d. 标架 / 累积标架 / 单方向不加 Rz
(5) Malvetti Lemma 3.9 Case II 的"单步必有方向使 θ 增加"在二维反例上是否成立"""
import numpy as np, itertools
from functools import reduce
from scipy.linalg import expm
rng = np.random.default_rng(1)
I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)
n = 4; N = 2**n
op = lambda P,i: reduce(np.kron,[P if j==i else I2 for j in range(n)])
bits = lambda x: np.array([(x>>(n-1-k))&1 for k in range(n)])
idx = lambda b: int(''.join(map(str,b)),2)
def rand_GL():
    while True:
        A = rng.integers(0,2,(n,n))
        if round(np.linalg.det(A)) % 2: return A
def perm(A):
    P = np.zeros((N,N))
    for x in range(N): P[idx((A@bits(x))%2), x] = 1
    return P
CNOT_moves = [np.eye(n,dtype=int) + np.eye(n,dtype=int)[:,[i]]@np.eye(n,dtype=int)[[j],:] for i in range(n) for j in range(n) if i!=j]
edges = list(range(n-1))
# 局部生成元：在边 e 上 M = (ph|ab><00| - h.c.)/√2，水平部分 HS 范数为 1
loc = []
for ab in range(1,4):
    for ph in (1,1j):
        M = np.zeros((4,4),complex); M[ab,0] = ph; M[0,ab] = -np.conj(ph); loc.append(M/np.sqrt(2))
embed = lambda M,e: reduce(np.kron,[np.eye(2**e),M,np.eye(2**(n-e-2))])
e0 = np.zeros(N,complex); e0[0] = 1

def heis():  return sum(op(P,i)@op(P,(i+1)%n) for i in range(n) for P in (X,Y,Z))/4
def rand2():                                           # 随机 2-local，非简并
    H = sum(rng.normal()*op(P,i)@op(Q,i+1) for i in range(n-1) for P in (X,Y,Z) for Q in (X,Y,Z))
    return (H + sum(rng.normal()*op(P,i) for i in range(n) for P in (X,Y,Z)))/4
f = lambda U,H: np.real(U[:,0].conj()@H@U[:,0])
haar = lambda: np.linalg.qr(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))[0]

# (1)
H = rand2(); U = haar(); c = perm(rand_GL()); e = 1
coef = rng.normal(size=6); M = sum(a*m for a,m in zip(coef,loc)); D = c@embed(M,e)@c.T
v = D@e0; Dm = np.outer(v,e0.conj()) - np.outer(e0,v.conj())
print("(1) max_s |f(Ue^{sD}) - f(Ue^{sD_m})| =", max(abs(f(U@expm(s*D),H)-f(U@expm(s*Dm),H)) for s in np.linspace(-3,3,13)),
      "；U e^{sD} 与 U e^{sD_m} 本身差", round(np.linalg.norm(expm(D)-expm(Dm)),3))

def step(U, H, eta, c, e, which=None):
    """输入侧一步：在边 e、标架 c 上做投影梯度步；which=None 用 6 个方向，否则只用第 which 个方向"""
    psi = U[:,0]; Hpsi = H@psi
    js = range(6) if which is None else [which]
    g = {j: 2*np.real(Hpsi.conj()@(U@(c@embed(loc[j],e)@e0))) for j in js}  # ⟨G, D_j⟩
    Mx = -eta*sum(g[j]*loc[j] for j in js)
    return U@c@embed(expm(Mx),e)@c.T

# (2) 一阶平均下降
H = rand2(); ell = 4*np.linalg.norm(H,2)
for trial in range(3):
    U = haar(); w = H@U[:,0] - f(U,H)*U[:,0]; G2 = 2*np.vdot(w,w).real   # |grad|^2 = 2 Var
    eta = 1e-4; drops = []
    for _ in range(4000):
        drops.append(f(U,H) - f(step(U,H,eta,perm(rand_GL()),rng.integers(len(edges))),H))
    print(f"(2) 平均下降/(η|grad|²) = {np.mean(drops)/(eta*G2):.4f} ± {np.std(drops)/np.sqrt(4000)/(eta*G2):.4f}，预测 3/(N-1) = {3/(N-1):.4f}")

# (3) 局部线性速率
H = heis(); E = np.linalg.eigvalsh(H); E0 = E[0]; Delta = E[E > E0+1e-9][0]-E0
ell = 4*np.linalg.norm(H,2); eta = 1/ell; kappa = eta*(1-ell*eta/2)*3/(N-1)
slopes = []
for s in range(6):
    U = haar(); d = []
    for t in range(6000):
        U = step(U,H,eta,perm(rand_GL()),rng.integers(len(edges))); d.append(f(U,H)-E0)
    d = np.array(d); tt = np.where((d < Delta/2) & (d > 1e-11))[0]
    slopes.append(np.polyfit(tt, np.log(d[tt]), 1)[0])
print(f"(3) Heisenberg 环 Δ={Delta:.3f}：实测 log δ 斜率 {np.round(slopes,4)}，定理保证 ≤ log(1-2κΔ) = {np.log(1-2*kappa*Delta):.4f}")

# (4) 鞍点逃逸
def escape_test(H, mode, trials=12, T=8000, pert=1e-4):
    E, V = np.linalg.eigh(H); lev = np.unique(np.round(E,9)); ell = 4*np.linalg.norm(H,2); eta = 1/ell
    out = {}
    for j in range(1, len(lev)-1):
        Ej = lev[j]; Elow = lev[j-1]; vec = V[:, np.argmin(abs(E-Ej))]
        times = []
        for tr in range(trials):
            psi = vec + pert*(rng.normal(size=N)+1j*rng.normal(size=N)); psi /= np.linalg.norm(psi)
            W = haar(); W[:,0] = psi; U = np.linalg.qr(W)[0]; U = U*(np.vdot(U[:,0],psi)/abs(np.vdot(U[:,0],psi)))
            Acur = np.eye(n,dtype=int); hit = None
            for t in range(T):
                if mode == 'iid':   c = perm(rand_GL()); which = None
                elif mode == 'acc':
                    Acur = (Acur@CNOT_moves[rng.integers(len(CNOT_moves))])%2; c = perm(Acur); which = None
                elif mode == 'single': c = perm(rand_GL()); which = rng.integers(6)
                U = step(U,H,eta,c,rng.integers(len(edges)),which)
                if f(U,H) < Ej - 0.1*(Ej-Elow): hit = t; break
            times.append(hit)
        ok = [t for t in times if t is not None]
        out[round(Ej,3)] = (len(ok), trials, int(np.median(ok)) if ok else None)
    return out
for name, H in [("Heisenberg 环", heis()), ("随机 2-local", rand2())]:
    for mode in ('iid','acc','single'):
        print(f"(4) {name:10s} {mode:6s} 逃逸（成功数, 试验数, 中位步数）按鞍点能级：", escape_test(H, mode))

# (5) 二维反例：f = x² - y²，方向 {π/8, π/4, 3π/8}（两两不共线，满足 Malvetti Assumption 3.1）
Lam = np.diag([2.,-2.]); dirs = [np.array([np.cos(t),np.sin(t)]) for t in (np.pi/8,np.pi/4,3*np.pi/8)]
theta = lambda w: np.arctan2(w[1]**2, w[0]**2)
w = np.array([1.,0.3]); eta = 0.1
print("(5) w=(1,0.3), η=0.1 时三个方向的 Δθ =", [round(theta(w-eta*(u@Lam@w)*u)-theta(w),4) for u in dirs])
esc = 0
for tr in range(200):
    w = np.array([1., 0.3]); 
    for t in range(3000):
        u = dirs[rng.integers(3)]; w = w - eta*(u@Lam@w)*u; w /= np.linalg.norm(w)
        if w@Lam@w < 0: esc += 1; break
print(f"    但整条轨迹仍逃逸：{esc}/200")
# 注：本文件是 check_rgf_theorems.py 的早期版本；special01 讲义 7.4 节的中位步数和"单方向慢 5–6 倍"来自它的 (4) 段（12 次/组，iid/acc/single）。
