"""定理 G 证明里的块覆盖引理（n=3,4，随机 2-local H，Haar 补全，iid CNOT 标架）。
一块 = 连续 m 步，块内 A_s 的并集覆盖全部非零串（"好块"）。对线性化 δ_{s+1} = (I - η P_s 𝓗) δ_s，
(C1) |𝓗δ_t|² <= (N-1)(1+ηℓ√m)² · S，S = Σ_s |P_s 𝓗 δ_s|²（对每个 δ 都成立，确定性）；
(C2) 好块上 -q(δ_{t+m}) >= -q(δ_t) + 2β|𝓗δ_t|²/C_m，对所有 δ（不只是负锥里的）；
(C3) 好块出现的频率 p_m/m（定理 G 的一致常数 γ̃± 用它）。"""
import numpy as np
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "check_rgf_hyperbolic.py")).read().split('print("(1)')[0])
rng = np.random.default_rng(3)
for n in (3, 4):
    N, H = setup(n); E, V = np.linalg.eigh(H); ell = 2*(E[-1]-E[0])
    for fac in (0.5, 0.99):
        eta = fac/ell; beta = eta*(1-eta*ell/2); r1 = r2 = 0; blocks = []
        for trial in range(300):
            k = rng.integers(1, N); U = haar(N); U[:,0] = V[:,k]; U = np.linalg.qr(U)[0]
            Hc = 2*(U[:,1:].conj().T@H@U[:,1:] - E[k]*np.eye(N-1))
            d0 = rng.normal(size=N-1)+1j*rng.normal(size=N-1); d = d0.copy(); S = 0; cov = set(); m = 0
            while len(cov) < N-1:
                ys = [y-1 for y in Aset(rand_GL(n), rng.integers(n-1), n)]; cov |= set(ys); m += 1
                p = np.zeros(N-1); p[ys] = 1; w = p*(Hc@d); S += np.vdot(w,w).real; d = d - eta*w
            Cm = (N-1)*(1+eta*ell*np.sqrt(m))**2; h0 = np.linalg.norm(Hc@d0)**2
            q = lambda x: np.vdot(x, Hc@x).real
            r1 = max(r1, h0/(Cm*S)); r2 = max(r2, (q(d) - q(d0) + 2*beta*h0/Cm)/h0)
            blocks.append(m)
        print(f"n={n} η={fac}/ℓ：(C1) max |𝓗δ|²/(C_m S) = {r1:.3f}（应 <= 1）；"
              f"(C2) max [q(δ_m)-q(δ_0)+2β|𝓗δ|²/C_m]/|𝓗δ|² = {r2:.2e}（应 <= 0）；覆盖所需步数中位数 {int(np.median(blocks))}")
