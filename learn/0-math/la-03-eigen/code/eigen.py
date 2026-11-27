"""线性代数 03 里的数字：特征分解、幂迭代、随机游走、重复相乘时的增长。
运行：python eigen.py（只需 numpy）
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")


def hand():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    w, V = np.linalg.eigh(A)
    print("特征值", w, "\n特征向量（按列）\n", V)
    print("A [1,1] =", A @ [1, 1], "  A [1,-1] =", A @ [1, -1])
    print("A^5 =", np.linalg.matrix_power(A, 5).tolist())
    print("P D^5 P^T =", np.round(V @ np.diag(w ** 5) @ V.T, 6).tolist())
    for name, M in (("旋转 90°", [[0, -1], [1, 0]]), ("剪切", [[1, 1], [0, 1]])):
        w, V = np.linalg.eig(np.array(M, float))
        print(name, "特征值", w, " 特征向量矩阵的秩", np.linalg.matrix_rank(V, tol=1e-8))


def power_iteration():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    v, target = np.array([1.0, 0.0]), np.array([1.0, 1.0]) / np.sqrt(2)
    print("\n幂迭代  k   向量                 与 [1,1] 的夹角   瑞利商 vᵀAv")
    for k in range(7):
        ang = np.degrees(np.arccos(min(1.0, v @ target)))
        print(f"        {k}   [{v[0]:.4f}, {v[1]:.4f}]     {ang:7.3f}°        {v @ A @ v:.6f}")
        v = A @ v
        v /= np.linalg.norm(v)


def random_walk():
    # 第 j 列 = 从页面 j 出发的去向概率。A→B、C 各一半；B→C；C→A
    P = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.5, 1.0, 0.0]])
    w = np.linalg.eigvals(P)
    print("\n随机游走的特征值", np.round(w, 4), " 模长", np.round(np.abs(w), 4))
    p = np.array([1.0, 0.0, 0.0])
    for k in range(21):
        if k in (0, 1, 2, 3, 5, 10, 20):
            print(f"  走 {k:2d} 步后  A {p[0]:.4f}  B {p[1]:.4f}  C {p[2]:.4f}")
        p = P @ p


def growth():
    A = np.array([[0.9, 5.0], [0.0, 0.9]])
    print("\n特征值都是 0.9 的矩阵 [[0.9,5],[0,0.9]]，最大奇异值", round(float(np.linalg.svd(A, compute_uv=False)[0]), 4))
    x = np.array([0.0, 1.0])
    norms = []
    for k in range(101):
        norms.append(np.linalg.norm(x))
        x = A @ x
    for k in (0, 1, 2, 5, 10, 20, 50, 100):
        print(f"  k={k:3d}  |A^k x| = {norms[k]:.4g}")
    print("  峰值出现在 k =", int(np.argmax(norms)), " 大小", round(float(max(norms)), 3))


def random_matrices(n=64, steps=50, trials=20):
    print(f"\n{n}×{n} 随机矩阵（元素标准差 = g/√n），连乘 {steps} 次后向量长度的几何平均")
    for g in (0.9, 1.0, 1.1):
        radius, final = [], []
        for s in range(trials):
            rng = np.random.default_rng(s)
            A = rng.standard_normal((n, n)) * g / np.sqrt(n)
            x = rng.standard_normal(n)
            x /= np.linalg.norm(x)
            radius.append(np.abs(np.linalg.eigvals(A)).max())
            final.append(np.linalg.norm(np.linalg.matrix_power(A, steps) @ x))
        print(f"  g={g}  最大特征值模长平均 {np.mean(radius):.3f}   |A^{steps} x| = {np.exp(np.mean(np.log(final))):.3g}   g^{steps} = {g ** steps:.3g}")


if __name__ == "__main__":
    hand()
    power_iteration()
    random_walk()
    growth()
    random_matrices()
