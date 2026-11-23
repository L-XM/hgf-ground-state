"""余项控制（7.4.8 节）的数值核对：误差是乘性的。沿用 check_disk.py 的陷阱和标架。
ε_t := (q(a_{t+1}) - q(a_t))_+ 是"q 的上升量"（线性部分只会让 q 下降，所以上升全是误差）。
应有 ε_t <= C r_t (|a_t|^2 + |a_t||Δk_t|)，C 与 |u| 无关；|Δk_t| <= C' Σ_{s<t} |a_s|。"""
import numpy as np
src = open("check_disk.py").read().split('print("(3)')[0]
exec(src)
T = int(np.argmax(ds < 1e-3)); UT = traj[T]; kT = kof(UT); Eu = np.eye(N)[:, Ed < lam].astype(complex)
print("(4) 误差结构")
for eps in (1e-5, 1e-8, 1e-11):
    u = Eu@(rng.normal(size=Eu.shape[1])+1j*rng.normal(size=Eu.shape[1])); u *= eps/np.linalg.norm(u)
    U1, U2 = UT.copy(), sigma(dl(UT)+u)@kT
    Cmax, Dmax, suma, t = 0, 0, 0, T
    while t < 6000:
        a = dl(U2)-dl(U1); r = max(np.linalg.norm(dl(U1)), np.linalg.norm(dl(U2)))
        if r > 1e-2: break
        dk = np.linalg.norm(kof(U2)-kof(U1)); na = np.linalg.norm(a)
        if suma > 0: Dmax = max(Dmax, dk/suma)
        c = perm(frames[t][0]); e = frames[t][1]; V1, V2 = step(U1, c, e), step(U2, c, e)
        inc = max(0, q(dl(V2)-dl(V1)) - q(a)); Cmax = max(Cmax, inc/(r*(na**2 + na*dk)))
        U1, U2 = V1, V2; suma += na; t += 1
    print(f"  |u|={eps:.0e}: {t-T} 步内 max ε_t/(r_t(|a|²+|a||Δk|)) = {Cmax:.3f}，max |Δk_t|/Σ|a_s| = {Dmax:.3f}")
