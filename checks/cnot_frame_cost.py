"""Average CNOT cost quoted in Sec. III (main text).

Pauli rotation along a uniformly random non-identity Pauli string of weight w: 2(w-1) CNOTs.
Independent frame change: uniformly random element of GL(n, F_2), minimum number of CNOTs
with all-to-all connectivity, by breadth-first search over GL(n, F_2).
"""
import itertools
from collections import deque

def pauli_cost(n):
    tot = cnt = 0
    for s in itertools.product(range(4), repeat=n):
        w = sum(c != 0 for c in s)
        if w:
            tot += 2 * (w - 1); cnt += 1
    return tot / cnt

def frame_cost(n):
    # a matrix is stored as a tuple of n row bitmasks; CNOT(c -> t) adds row c to row t
    start = tuple(1 << i for i in range(n))
    dist = {start: 0}; q = deque([start])
    while q:
        A = q.popleft()
        for c in range(n):
            for t in range(n):
                if c != t:
                    B = list(A); B[t] ^= A[c]; B = tuple(B)
                    if B not in dist:
                        dist[B] = dist[A] + 1; q.append(B)
    return len(dist), sum(dist.values()) / len(dist)

for n in (3, 4):
    size, avg = frame_cost(n)
    print(f"n={n}: Pauli rotation {pauli_cost(n):.2f} CNOTs; |GL|={size}, frame change {avg:.2f} CNOTs")
