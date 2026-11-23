"""Malvetti 2405.12039 Assumption 3.1 在输入侧（K=Stab|0>）方向集上是否成立。
条件：方向投影到 T_ψ CP^{N-1} 后张满，且去掉任意一个方向（连同它的所有实数倍）后仍张满。
在 |0> 的标架下核对（输入侧方向随标架移动，所以对每个点都一样）。"""
import numpy as np, itertools
from functools import reduce
n = 3; N = 2**n; rng = np.random.default_rng(0)
I = np.eye(2); X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1]).astype(complex)
e0 = np.zeros(N, complex); e0[0] = 1

def tangent(v):                       # 投影到 |0> 的正交补，转成实向量（去掉 |0> 分量）
    v = v - np.vdot(e0, v)*e0
    return np.concatenate([v.real[1:], v.imag[1:]])

def classes(vecs, tol=1e-9):          # 按"互为实数倍"分组
    out = []
    for v in vecs:
        if np.linalg.norm(v) < tol: continue
        u = v/np.linalg.norm(v)
        for c in out:
            if abs(abs(c[0]@u) - 1) < tol: c.append(u); break
        else: out.append([u])
    return out

def check(name, vecs):
    cl = classes(vecs); full = np.linalg.matrix_rank(np.array([c[0] for c in cl]))
    worst = min(np.linalg.matrix_rank(np.array([c[0] for j, c in enumerate(cl) if j != i])) for i in range(len(cl)))
    print(f"{name:34s} 方向类数={len(cl):4d}  张成维数={full}/{2*(N-1)}  去掉一类后最小维数={worst}  满足3.1={full==worst==2*(N-1)}")

# 局部 SU(4)/U(3) 生成元（每条边，相对 |00> 水平）
def local_gens(edge):
    gens = []
    for ab in range(1, 4):
        for ph in (1, 1j):
            M = np.zeros((4,4), complex); M[ab,0] = ph; M[0,ab] = -np.conj(ph)
            i = edge[0]
            gens.append(reduce(np.kron, [np.eye(2**i), M, np.eye(2**(n-i-2))]))
    return gens
edges = [(0,1), (1,2)]
G = [g for e in edges for g in local_gens(e)]
check("局部 SU(4)/U(3)，不加 twirl", [tangent(g@e0) for g in G])

# CNOT 电路 = GL(n,F2) 的置换矩阵（固定 |0>）
def perm(A):
    P = np.zeros((N,N))
    for x in range(N):
        bits = np.array([(x>>(n-1-k))&1 for k in range(n)]); y = (A@bits) % 2
        P[int(''.join(map(str,y)),2), x] = 1
    return P
GL = [np.array(A).reshape(n,n) for A in itertools.product([0,1], repeat=n*n)
      if round(np.linalg.det(np.array(A).reshape(n,n))) % 2]
cn = [perm(A) for A in GL]
check("+ CNOT twirl（GL(3,F2) 全部 168 个）", [tangent(c@g@c.T@e0) for c in cn for g in G])

# 再加 Z 相位 twirl：对角相位，φ_0 = 0，固定 |0>
ph = [np.diag(np.exp(1j*np.r_[0, rng.uniform(0, 2*np.pi, N-1)])) for _ in range(3)]
check("+ CNOT twirl + 3 组随机 Z 相位", [tangent(d@c@g@c.T@d.conj().T@e0) for d in ph for c in cn for g in G])

# 输入侧全 Pauli 采样（McMahon/Pervez 式）
P = {'I':I,'X':X,'Y':Y,'Z':Z}
paulis = [reduce(np.kron, [P[c] for c in s]) for s in itertools.product('IXYZ', repeat=n) if set(s) != {'I'}]
check("输入侧全 Pauli 串", [tangent(1j*p@e0) for p in paulis])

# 只用 Clifford 对角相位（S 和 CZ：相位 i^{q(x)}，q 为 F2 上的二次型提升；q(0)=0，固定 |0>）
def bits(x): return np.array([(x>>(n-1-k))&1 for k in range(n)])
def cliff_diag(lin, quad):
    return np.diag([1j**(int(lin@bits(x)) + 2*int(bits(x)@quad@bits(x))) for x in range(N)])
cds = []
for lin in itertools.product(range(4), repeat=n):
    for q in [np.zeros((n,n),int), np.triu(np.ones((n,n),int),1)]:
        cds.append(cliff_diag(np.array(lin), q))
check("+ CNOT twirl + Clifford 对角相位(S,CZ)", [tangent(d@c@g@c.T@d.conj().T@e0) for d in cds[:8] for c in cn for g in G])
check("只 S 相位（不加 CNOT）", [tangent(d@g@d.conj().T@e0) for d in cds for g in G])

# T 相位（e^{iπ/4}，非 Clifford）：每个基矢方向出现 4 个不共线的实方向
T = [np.diag(np.exp(1j*np.pi/4*np.r_[0, rng.integers(0, 8, N-1)])) for _ in range(6)]
check("+ CNOT twirl + T 相位", [tangent(d@c@g@c.T@d.conj().T@e0) for d in T for c in cn for g in G])

# 局部 T 门：相位 e^{iπ/4 Σ_k c_k x_k}，c_k∈{0..7}，只用单比特 T 的幂（可局部实现，固定 |0>）
Tloc = [np.diag([np.exp(1j*np.pi/4*int(np.array(c)@bits(x))) for x in range(N)]) for c in itertools.product(range(8), repeat=n)]
sel = [Tloc[i] for i in rng.choice(len(Tloc), 6, replace=False)]
check("+ CNOT twirl + 6 组局部 T 幂", [tangent(d@c@g@c.T@d.conj().T@e0) for d in sel for c in cn for g in G])
check("+ CNOT twirl + 全部局部 T 幂", [tangent(d@c@g@c.T@d.conj().T@e0) for d in Tloc[::37] for c in cn for g in G])
