"""三维实态玩具模型：N=3，态是 R^3 里的单位向量（对径点是同一个态），H=diag(λx, λy, λz)。
世界坐标：z = 基态（能量 0），x = 鞍点（能量 1），y = 最高态（能量 2.2）。
四种一步：
  full    精确黎曼梯度：沿 -g 的大圆走 η|g|（Wiersema–Killoran 的流的离散版）
  random  随机方向：切平面里均匀随机取一个方向 u，只沿 u 走（Malvetti 等的 Case I）
  column  本文的缩影：方向只能是电路自己的另一列 U|y>，标架每步随机决定 y∈{1,2}
  noframe 同上，但永远用同一列 y=1
column/noframe 的一步就是"水平门"：在 (U|0>, U|y>) 平面里转一个角度，另一列不动。"""
import numpy as np
LAM = np.array([1.0, 2.2, 0.0])          # x, y, z 方向的能量
H = np.diag(LAM); ELL = 2*(LAM.max()-LAM.min()); ETA = 1/ELL
F = lambda p: p @ H @ p
grad = lambda p: 2*(H@p - F(p)*p)


def rot(a, b, th):
    """在 (a,b) 平面里把 a 转向 b 角度 th，返回新的 (a,b)。"""
    return np.cos(th)*a + np.sin(th)*b, -np.sin(th)*a + np.cos(th)*b


def run(U0, mode, T, rng=None, eta=ETA, fixed=1):
    U = np.array(U0, float).copy(); path = [U[:, 0].copy()]; cols = [U.copy()]
    rng = rng or np.random.default_rng(0)
    for t in range(T):
        p = U[:, 0]; g = grad(p)
        if mode == "full":
            ng = np.linalg.norm(g)
            if ng > 1e-12:
                th = eta*ng; U[:, 0] = np.cos(th)*p - np.sin(th)*g/ng
        elif mode == "random":
            a = rng.normal(size=3); a -= (a @ p)*p; a /= np.linalg.norm(a)
            d = a @ g; U[:, 0], _ = rot(p, a, -eta*d)
        else:
            y = fixed if mode == "noframe" else rng.integers(1, 3)
            d = U[:, y] @ g
            U[:, 0], U[:, y] = rot(U[:, 0], U[:, y], -eta*d)
        U = np.linalg.qr(U)[0] * np.sign(np.diag(np.linalg.qr(U)[1]))   # 消掉舍入漂移，保持正交
        path.append(U[:, 0].copy()); cols.append(U.copy())
    return np.array(path), cols


def frame_from(psi, u1):
    """以 psi 为第 0 列、u1 为第 1 列（会被正交化）补全成正交矩阵。"""
    psi = psi/np.linalg.norm(psi); u1 = u1 - (u1 @ psi)*psi; u1 /= np.linalg.norm(u1)
    return np.column_stack([psi, u1, np.cross(psi, u1)])


def densify(path, k=12):
    """每一步是一段大圆弧：在相邻两点之间插值，画出来就是弧，不是弦。"""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        c = np.clip(a @ b, -1, 1); w = np.arccos(c)
        if w < 1e-12: continue
        for s in np.linspace(0, 1, k)[1:]:
            out.append((np.sin((1-s)*w)*a + np.sin(s*w)*b)/np.sin(w))
    return np.array(out)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    psi0 = np.array([0.35, 0.92, 0.18]); psi0 /= np.linalg.norm(psi0)
    U0 = frame_from(psi0, np.array([0.2, -0.1, 0.97]))
    for m in ("full", "random", "column", "noframe"):
        P, cols = run(U0, m, 400, np.random.default_rng(1))
        print(f"{m:8s} F: {F(P[0]):.3f} -> {F(P[40]):.3f} (40 步) -> {F(P[-1]):.2e} (400 步)  |g|末={np.linalg.norm(grad(P[-1])):.2e}")
    # 陷阱：态在 x-y 大圆上，隐藏列 = z（基态）
    psiT = np.array([0.45, 0.89, 0.0]); psiT /= np.linalg.norm(psiT)
    for alpha in (0.0, 0.25):
        t = np.array([-psiT[1], psiT[0], 0.0])
        u1 = np.cos(alpha)*t + np.sin(alpha)*np.array([0, 0, 1.0])
        P, _ = run(frame_from(psiT, u1), "column", 400, np.random.default_rng(2))
        print(f"trap alpha={alpha}: F 末 = {F(P[-1]):.3e}, 末态 z 分量 = {abs(P[-1][2]):.3e}")
