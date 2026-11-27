"""线性代数 04 里的数字：手算例子、图片的低秩近似、把迷你 GPT 的权重矩阵截成低秩。
运行：python lowrank.py（只需 numpy，约几十秒）
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")
GPT = Path(__file__).resolve().parents[3] / "3-llm/tf-04-mini-gpt"


def rank_k(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def hand():
    A = np.array([[3.0, 0.0], [4.0, 5.0]])
    U, s, Vt = np.linalg.svd(A)
    print("A^T A =", (A.T @ A).tolist(), " 奇异值", s, " 平方", s ** 2)
    print("U =\n", U, "\nV^T =\n", Vt)
    A1 = rank_k(A, 1)
    print("秩 1 近似 =", np.round(A1, 6).tolist(), " 误差（F 范数）", np.linalg.norm(A - A1))


def picture(n=64):
    """和互动页里的“示例图”用同一公式生成：渐变底 + 圆 + 方框 + 斜线。"""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = 0.25 * x
    img[(x - 0.32) ** 2 + (y - 0.36) ** 2 < 0.045] = 0.9
    img[(x > 0.55) & (x < 0.9) & (y > 0.5) & (y < 0.85)] = 0.65
    img[np.abs(x + y - 1.25) < 0.025] = 1.0
    return img


def image_table():
    img = picture()
    s = np.linalg.svd(img, compute_uv=False)
    m, n = img.shape
    print(f"\n{m}×{n} 示例图（秩 {np.linalg.matrix_rank(img)}）")
    print("  k   存储的数字   占原图   相对误差   公式 sqrt(尾部平方和)/总")
    for k in (1, 2, 4, 8, 16, 32):
        err = np.linalg.norm(img - rank_k(img, k)) / np.linalg.norm(img)
        formula = np.sqrt((s[k:] ** 2).sum() / (s ** 2).sum())
        print(f" {k:2d}   {k * (m + n + 1):7d}    {k * (m + n + 1) / (m * n):6.1%}   {err:.4f}     {formula:.4f}")


def load_gpt():
    sys.path.insert(0, str(GPT / "code"))
    from minigpt import MiniGPT

    src = (GPT / "demo/model.js").read_text()
    blob = json.loads(src[src.index("{"):src.rindex("}") + 1])
    cfg = blob["cfg"]
    m = MiniGPT(**cfg)
    for k in m.p:
        m.p[k] = np.array(blob["weights"][k], np.float32).reshape(m.p[k].shape)
    return m


def gpt_table():
    sys.path.insert(0, str(GPT / "code"))
    from train import accuracy_on_facts, val_loss

    base = load_gpt()
    names = [f"{l}.{n}" for l in range(2) for n in ("Wq", "Wk", "Wv", "Wo", "W1", "W2")]
    print("\n各矩阵：保留多少“能量”（奇异值平方和）需要几个奇异值（满秩 32）")
    for k in names:
        s = np.linalg.svd(base.p[k].astype(float), compute_uv=False)
        c = np.cumsum(s ** 2) / (s ** 2).sum()
        print(f"  {k:5s} {str(base.p[k].shape):10s} 90% 需要 {int(np.searchsorted(c, 0.9)) + 1:2d} 个，99% 需要 {int(np.searchsorted(c, 0.99)) + 1:2d} 个")
    full = {k: v.copy() for k, v in base.p.items()}
    print("\n把两层的 12 个权重矩阵全部截成秩 k（词向量和位置向量不动）")
    print("  k   验证损失   知识题    参数（按两个小矩阵存）")
    for k in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        for n in names:
            base.p[n] = rank_k(full[n].astype(float), k).astype(np.float32)
        ok, total = accuracy_on_facts(base)
        stored = sum(min(full[n].size, k * sum(full[n].shape)) for n in names)
        print(f" {k:2d}   {val_loss(base):.4f}     {ok:2d}/{total}    {stored:6d} / {sum(full[n].size for n in names)}")


def lora_params():
    d = 4096
    for r in (4, 8, 16, 64):
        print(f"d={d} r={r:2d}：BA 共 {2 * d * r:,} 个参数，完整矩阵 {d * d:,}，占 {2 * d * r / d / d:.2%}")


if __name__ == "__main__":
    hand()
    image_table()
    gpt_table()
    lora_params()
