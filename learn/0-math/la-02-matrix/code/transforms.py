"""线性代数 02 里的数字：矩阵乘向量、复合顺序、行列式、旋转与 RoPE、乘法次数与耗时。
运行：python transforms.py（只需 numpy；耗时和机器有关）
"""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def hand():
    A, x = np.array([[2, 1], [0, 1]]), np.array([1, 2])
    print("A x =", A @ x, " = 1×第一列 + 2×第二列 =", 1 * A[:, 0] + 2 * A[:, 1])
    R, S, e = np.array([[0, -1], [1, 0]]), np.array([[2, 0], [0, 1]]), np.array([1, 0])
    print("先转后拉 S R e =", S @ R @ e, "  先拉后转 R S e =", R @ S @ e)
    print("S R =", (S @ R).tolist(), " R S =", (R @ S).tolist())
    for M in ([[2, 1], [0, 1]], [[1, 2], [2, 4]], [[0, 1], [1, 0]]):
        M = np.array(M)
        print(M.tolist(), "行列式", round(float(np.linalg.det(M)), 6), "秩", np.linalg.matrix_rank(M))


def rope():
    q, k = np.array([1.6, 0.4]), np.array([1.0, 1.2])
    w = 0.3
    print("\n旋转后的点积只和位置差有关（位置差都是 3）：")
    for m, n in ((0, 3), (10, 13), (500, 503)):
        print(f"  q 在位置 {m:3d}，k 在位置 {n:3d}：{(rot(m * w) @ q) @ (rot(n * w) @ k):+.6f}   长度 {np.linalg.norm(rot(m * w) @ q):.6f}")


def cost(n=3000, repeat=3):
    rng = np.random.default_rng(0)
    A, B, x = rng.standard_normal((n, n)), rng.standard_normal((n, n)), rng.standard_normal(n)

    def best(f):
        ts = []
        for _ in range(repeat):
            t = time.perf_counter()
            r = f()
            ts.append(time.perf_counter() - t)
        return min(ts), r

    t1, r1 = best(lambda: (A @ B) @ x)
    t2, r2 = best(lambda: A @ (B @ x))
    print(f"\nn = {n}")
    print(f"(AB)x  乘法 {n**3 + n**2:.3g} 次  耗时 {t1 * 1000:.1f} ms")
    print(f"A(Bx)  乘法 {2 * n**2:.3g} 次  耗时 {t2 * 1000:.1f} ms")
    print("结果最大相对差", float(np.max(np.abs(r1 - r2)) / np.max(np.abs(r1))))


if __name__ == "__main__":
    hand()
    rope()
    cost()
