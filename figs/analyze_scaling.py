"""分析 scaling.csv：每组 (H,n,variant) 的到 δ<1e-8 步数中位数、基态附近衰减率，对比定理 3 的 -log(1-ρ)。"""
import numpy as np, csv, sys, collections
from data_scaling import hamiltonian
files = sys.argv[1:] or ["scaling.csv"]
runs = collections.defaultdict(list)
for fn in files:
    for r in csv.DictReader(open(fn)):
        runs[(r["H"], int(r["n"]), r["variant"], int(r["seed"]))].append((int(r["t"]), float(r["delta"])))
spec = {}
print(f"{'H':10s} {'n':>2s} {'var':4s} {'runs':>4s} {'steps(med)':>10s} {'rate(med)':>10s} {'thm3':>9s} {'ratio':>6s} {'rate*(N-1)':>10s}")
groups = collections.defaultdict(list)
for k, tr in runs.items(): groups[k[:3]].append(tr)
rows = []
for (h, n, v), trs in sorted(groups.items()):
    if (h, n) not in spec:
        E = np.linalg.eigvalsh(hamiltonian(h, n)); E0 = E[0]; D = E[E > E0 + 1e-9][0] - E0; ell = 2 * (E[-1] - E0)
        eta = 1 / ell; beta = eta * (1 - ell * eta / 2); rho = 2 * beta * (3 / (2 ** n - 1)) * D
        spec[(h, n)] = (D, -np.log(1 - rho))
    D, th = spec[(h, n)]
    steps, rates = [], []
    for tr in trs:
        t = np.array([a for a, _ in tr]); d = np.array([b for _, b in tr])
        if d[-1] >= 1e-8: continue
        steps.append(t[-1])
        m = (d < D / 2) & (d > 1e-8)
        if m.sum() > 5: rates.append(-np.polyfit(t[m], np.log(d[m]), 1)[0])
    if not steps: continue
    N = 2 ** n; r = np.median(rates)
    rows.append((h, n, v, len(steps), np.median(steps), r, th))
    print(f"{h:10s} {n:2d} {v:4s} {len(steps):4d} {np.median(steps):10.0f} {r:10.2e} {th:9.2e} {r/th:6.2f} {r*(N-1):10.3f}")
