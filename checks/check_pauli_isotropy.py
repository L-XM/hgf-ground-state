"""对照：输入侧随机单个 Pauli 串（2512.13401 / Malvetti 式）的平均投影 E[P]，
与随机 CNOT 标架 + 一个局部 SU(4)/U(3) 门比较。每个方向的 CNOT 代价另算。"""
import numpy as np, itertools
from functools import reduce
P = {'I':np.eye(2),'X':np.array([[0,1],[1,0]],complex),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1.,-1]).astype(complex)}
for n in (3, 4, 5):
    N = 2**n; e0 = np.zeros(N, complex); e0[0] = 1
    tang = lambda v: np.concatenate([v.real[1:], v.imag[1:]])
    EP = np.zeros((2*(N-1),)*2); cnt = 0
    for s in itertools.product('IXYZ', repeat=n):
        if set(s) == {'I'}: continue
        v = tang(1j*reduce(np.kron, [P[c] for c in s])@e0)
        if np.linalg.norm(v) > 1e-12: v /= np.linalg.norm(v); EP += np.outer(v, v)
        cnt += 1
    EP /= cnt
    d = EP.diagonal()
    print(f"n={n}: Pauli 单方向 E[P] 对角 ∈ [{d.min():.5f},{d.max():.5f}]，非对角 {np.abs(EP-np.diag(d)).max():.1e}；"
          f"1/(2(N-1))={1/(2*(N-1)):.5f}，局部门 3/(N-1)={3/(N-1):.5f}，比值 {3/(N-1)/d.mean():.1f}")
