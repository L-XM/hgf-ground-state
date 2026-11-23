"""表 I 的数据：鞍点逃离。n=4，η=0.9/ℓ（定理 4 的步长范围内），生成元不归一化（与正文一致）。
H：海森堡环（2 个激发鞍点能级）和随机 2-local（种子 21，14 个）。
初态：激发本征矢加 1e-4 的复高斯扰动，补全取 Haar。逃离判据：F < λ - 0.1(λ - λ_下)。
三种方式：iid CNOT 标架；单 CNOT 标架（懒游走）；iid 标架但每步只用 6 个方向中随机的 1 个。每个能级 12 次，最多 20000 步。"""
import numpy as np, json
from functools import reduce
from scipy.linalg import expm
n, N = 4, 16
I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)
op = lambda P,i: reduce(np.kron, [P if j==i else I2 for j in range(n)])
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
moves = [(np.eye(n,dtype=int)+np.outer(np.eye(n,dtype=int)[i], np.eye(n,dtype=int)[j])) % 2 for i in range(n) for j in range(n) if i != j]
rng = np.random.default_rng(21)
def rand_GL():
    while True:
        A = rng.integers(0,2,(n,n))
        if round(np.linalg.det(A)) % 2: return A
def step(U, H, eta, c, e, which=None):
    psi = U[:,0]; F = np.real(np.vdot(psi, H@psi)); g = 2*(H@psi - F*psi)
    js = range(6) if which is None else [which]
    Mx = sum(-eta*np.real(np.vdot(g, U@c@embed(loc[j],e)@e0))*loc[j] for j in js)
    return U@c@embed(expm(Mx), e)@c.conj().T
heis = sum(op(P,i)@op(P,(i+1)%n) for i in range(n) for P in (X,Y,Z))/4
r2 = sum(rng.normal()*op(P,i)@op(Q,i+1) for i in range(n-1) for P in (X,Y,Z) for Q in (X,Y,Z))
r2 = r2 + sum(rng.normal()*op(P,i) for i in range(n) for P in (X,Y,Z)); r2 = (r2+r2.conj().T)/2
out = {}
for name, H in (("heisenberg", heis), ("random2local", r2)):
    E, V = np.linalg.eigh(H); lev = np.unique(np.round(E, 9)); ell = 2*(E[-1]-E[0]); eta = 0.9/ell
    for mode in ("iid", "single_cnot", "one_direction"):
        res = []
        for j in range(1, len(lev)-1):
            vec = V[:, np.argmin(abs(E-lev[j]))]; times = []
            for tr in range(12):
                psi = vec + 1e-4*(rng.normal(size=N)+1j*rng.normal(size=N)); psi /= np.linalg.norm(psi)
                W = np.linalg.qr(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))[0]; W[:,0] = psi; U = np.linalg.qr(W)[0]
                A = rand_GL(); hit = None
                for t in range(20000):
                    if mode == "single_cnot":
                        if rng.random() < 0.5: A = (A@moves[rng.integers(len(moves))]) % 2
                        c, which = perm(A), None
                    elif mode == "iid": c, which = perm(rand_GL()), None
                    else: c, which = perm(rand_GL()), rng.integers(6)
                    U = step(U, H, eta, c, rng.integers(n-1), which)
                    if np.real(np.vdot(U[:,0], H@U[:,0])) < lev[j] - 0.1*(lev[j]-lev[j-1]): hit = t+1; break
                times.append(hit)
            ok = [t for t in times if t is not None]
            res.append(dict(level=float(lev[j]), escaped=len(ok), trials=12, median=float(np.median(ok)) if ok else None))
        out[f"{name}/{mode}"] = res
        meds = [r["median"] for r in res]
        print(f"{name:13s} {mode:13s} 能级数 {len(res)}  全部逃离 {all(r['escaped']==12 for r in res)}  中位步数 {min(meds):.0f}–{max(meds):.0f}", flush=True)
json.dump(out, open("escape.json", "w"), indent=1)
