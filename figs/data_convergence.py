"""图 2 的数据：n=4 海森堡环，η=1/ℓ，三种变体的 δ_t = F - E0。
iid CNOT 标架 / 累积（懒游走，每步至多 1 个 CNOT）/ 单个随机 Pauli 方向（输入侧，作为对照）。
每种 6 个种子，同一个 Haar 初态（每个种子不同）。输出 convergence.csv。"""
import numpy as np, itertools
from functools import reduce
from scipy.linalg import expm
n, N = 4, 16
I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)
op = lambda P,i: reduce(np.kron, [P if j==i else I2 for j in range(n)])
H = sum(op(P,i)@op(P,(i+1)%n) for i in range(n) for P in (X,Y,Z))/4
E = np.linalg.eigvalsh(H); E0 = E[0]; Delta = E[E > E0+1e-9][0]-E0; ell = 2*(E[-1]-E[0]); eta = 1/ell
beta = eta*(1-ell*eta/2); kappa = 3/(N-1); rho = 2*beta*kappa*Delta
bits = lambda x: np.array([(x>>(n-1-k))&1 for k in range(n)]); idx = lambda b: int(''.join(map(str,b)),2)
def perm(A):
    P = np.zeros((N,N), complex)
    for x in range(N): P[idx((A@bits(x))%2), x] = 1
    return P
loc = []
for ab in (1,2,3):
    for ph in (1,1j):
        M = np.zeros((4,4),complex); M[ab,0] = ph; M[0,ab] = -np.conj(ph); loc.append(M)
embed = lambda M,e: reduce(np.kron, [np.eye(2**e), M, np.eye(2**(n-e-2))])
e0 = np.eye(N)[0].astype(complex)
F = lambda U: np.real(np.vdot(U[:,0], H@U[:,0]))
def step(U, c, e):
    psi = U[:,0]; g = 2*(H@psi - F(U)*psi)
    x = [-eta*np.real(np.vdot(g, U@c@embed(Mj,e)@e0)) for Mj in loc]
    return U@c@embed(expm(sum(xj*Mj for xj,Mj in zip(x,loc))), e)@c.conj().T
paulis = [reduce(np.kron, [ (I2,X,Y,Z)[k] for k in ks]) for ks in itertools.product(range(4), repeat=n) if any(ks)]
def pauli_step(U, P):
    psi = U[:,0]; g = 2*(H@psi - F(U)*psi); G = 1j*P
    d = np.real(np.vdot(g, U@G@e0)); return U@expm(-eta*d*G)
def rand_GL(rng):
    while True:
        A = rng.integers(0,2,(n,n))
        if round(np.linalg.det(A)) % 2: return A
moves = [(np.eye(n,dtype=int)+np.outer(np.eye(n,dtype=int)[i], np.eye(n,dtype=int)[j])) % 2 for i in range(n) for j in range(n) if i != j]
T = 2500; rows = []
for seed in range(6):
    rng0 = np.random.default_rng(100+seed)
    U0 = np.linalg.qr(rng0.normal(size=(N,N))+1j*rng0.normal(size=(N,N)))[0]
    for var in ("iid", "accumulated", "pauli"):
        rng = np.random.default_rng(1000*seed + hash(var) % 997); U = U0.copy(); A = rand_GL(rng)
        for t in range(T+1):
            rows.append((var, seed, t, max(F(U)-E0, 1e-16)))
            if var == "iid": U = step(U, perm(rand_GL(rng)), rng.integers(n-1))
            elif var == "accumulated":
                if rng.random() < 0.5: A = (A@moves[rng.integers(len(moves))]) % 2
                U = step(U, perm(A), rng.integers(n-1))
            else: U = pauli_step(U, paulis[rng.integers(len(paulis))])
with open("convergence.csv","w") as f:
    f.write(f"# n=4 Heisenberg ring, eta=1/ell={eta:.6f}, Delta={Delta:.6f}, ell={ell:.6f}, kappa={kappa:.6f}, rho_TheoremB={rho:.6f}\n")
    f.write("variant,seed,t,delta\n")
    for r in rows: f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6e}\n")
# 斜率：在 1e-10 < δ < Δ/2 段拟合 log δ
import collections
print(f"Δ={Delta}, ℓ={ell}, κ={kappa:.4f}, Theorem B log(1-ρ)={np.log(1-rho):.4f}")
for var in ("iid","accumulated","pauli"):
    sl = []
    for seed in range(6):
        d = np.array([r[3] for r in rows if r[0]==var and r[1]==seed]); t = np.where((d<Delta/2)&(d>1e-10))[0]
        if len(t) > 20: sl.append(np.polyfit(t, np.log(d[t]), 1)[0])
    print(var, "斜率", np.round(sl,4), "终点 δ 中位数", np.median([ [r[3] for r in rows if r[0]==var and r[1]==s][-1] for s in range(6)]))
