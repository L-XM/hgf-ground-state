"""（只重跑 (b)）图 2、3 的补充数据。
(a) noframe.csv：n=4 海森堡环，η=1/ℓ，不加标架（c = I）的输入侧局部门，6 个种子（初态与 convergence.csv 相同）。
(b) lyapunov.csv：n=3 随机 2-local H（种子 5，同 check_rgf_hyperbolic.py），η=0.9/ℓ，每个激发能级 k=1..7 一个 Haar 补全，
    线性化 δ -> (I - η P_t 𝓗) δ 的全部 14 个 Lyapunov 指数，T=10000 步。
(c) trap.csv：n=4，H = Σ Z_i Z_{i+1}，η=1/8。三条轨迹：命题 C 的陷阱电路（CNOT 标架）；同一初态换 Haar 补全；
    陷阱电路加 1e-6 的基态分量。"""
import numpy as np
from functools import reduce
from scipy.linalg import expm
I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)
def tools(n):
    N = 2**n; op = lambda P,i: reduce(np.kron, [P if j==i else I2 for j in range(n)])
    bits = lambda x: np.array([(x>>(n-1-k))&1 for k in range(n)]); idx = lambda b: int(''.join(map(str,b)),2)
    def perm(A):
        P = np.zeros((N,N), complex)
        for x in range(N): P[idx((A@bits(x))%2), x] = 1
        return P
    def rand_GL(rng):
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
    e0 = np.eye(N)[0].astype(complex)
    def step(U, H, c, e, eta):
        psi = U[:,0]; Fv = np.real(np.vdot(psi, H@psi)); g = 2*(H@psi - Fv*psi)
        x = [-eta*np.real(np.vdot(g, U@c@embed(Mj,e)@e0)) for Mj in loc]
        return U@c@embed(expm(sum(xj*Mj for xj,Mj in zip(x,loc))), e)@c.conj().T
    return N, op, perm, rand_GL, Aset, step
# (b)
n = 3; N, op, perm, rand_GL, Aset, step = tools(n); rng = np.random.default_rng(5)
H = sum(rng.normal()*op(P,i)@op(Q,i+1) for i in range(n-1) for P in (X,Y,Z) for Q in (X,Y,Z))
H = H + sum(rng.normal()*op(P,i) for i in range(n) for P in (X,Y,Z)); H = (H+H.conj().T)/2
E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0]); eta = 0.9/ell
realify = lambda M: np.block([[M.real, -M.imag],[M.imag, M.real]])
with open("lyapunov.csv","w") as f:
    f.write("k,index,exponent\n")
    for k in range(1, N):
        U = np.linalg.qr(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))[0]; U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
        R = realify(2*(U[:,1:].conj().T@H@U[:,1:] - E[k]*np.eye(N-1))); Q = np.eye(2*(N-1)); s = np.zeros(2*(N-1))
        for t in range(10000):
            p = np.zeros(N-1); p[[y-1 for y in Aset(rand_GL(rng), rng.integers(n-1))]] = 1
            Q, Rr = np.linalg.qr(Q - eta*(np.r_[p,p][:,None]*(R@Q))); s += np.log(np.abs(np.diag(Rr)))
        sp = np.sort(s/10000)[::-1]
        for i, x in enumerate(sp): f.write(f"{k},{i},{x:.6e}\n")
        print("k", k, "正指数个数", int(np.sum(sp > 0)), "(2k =", 2*k, ")")
