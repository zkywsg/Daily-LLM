"""线性代数 01 里的数字：向量、内积、范数、高维夹角、迷你 GPT 词向量的相似度。
运行：python vectors.py（只需 numpy）
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# 部分 macOS 的 NumPy 会在结果正常时对 matmul 报虚假的溢出警告
warnings.filterwarnings("ignore", message=".*encountered in matmul")
GPT = Path(__file__).resolve().parents[3] / "3-llm/tf-04-mini-gpt"


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def hand():
    a, b = np.array([3.0, 4.0]), np.array([4.0, 3.0])
    print("a·b =", a @ b, " |a| =", np.linalg.norm(a), " |b| =", np.linalg.norm(b), " cos =", cos(a, b))
    print("角度（度） =", np.degrees(np.arccos(cos(a, b))))
    print("欧氏距离 |a-b| =", np.linalg.norm(a - b))
    print("a 在 b 上的投影 =", (a @ b) / (b @ b) * b)
    for p in (1, 2, np.inf):
        print(f"a 的 L{p} 范数 =", np.linalg.norm(a, p))


def high_dim(pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    print("\n维度   cos 的标准差   1/sqrt(d)   点积的标准差   sqrt(d)   夹角在 80~100 度的比例")
    for d in (2, 3, 10, 100, 1000):
        x, y = rng.standard_normal((pairs, d)), rng.standard_normal((pairs, d))
        dot = (x * y).sum(1)
        c = dot / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
        near = np.mean(np.abs(np.degrees(np.arccos(c)) - 90) <= 10)
        print(f"{d:5d}   {c.std():.4f}         {1 / np.sqrt(d):.4f}      {dot.std():7.2f}       {np.sqrt(d):6.2f}    {near:.1%}")


def embeddings():
    """迷你 GPT（见 3-llm/tf-04-mini-gpt）训练出来的词向量。"""
    src = (GPT / "demo/model.js").read_text()
    m = json.loads(src[src.index("{"):src.rindex("}") + 1])
    E = np.array(m["weights"]["E"]).reshape(m["cfg"]["vocab"], m["cfg"]["d"])
    idx = {c: i for i, c in enumerate(m["chars"])}
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    for group in ("猫狗鸟兔熊", "鱼骨虫胡竹"):
        print(f"\n“{group}”两两余弦相似度")
        for c in group:
            print(c, " ".join(f"{U[idx[c]] @ U[idx[o]]:+.2f}" for o in group))
    off = (U @ U.T)[~np.eye(len(E), dtype=bool)]
    print(f"\n全部 {len(E)} 个字两两相似度：均值 {off.mean():+.3f}，标准差 {off.std():.3f}")
    for c in "猫熊住吃鱼。":
        s = U @ U[idx[c]]
        top = [i for i in np.argsort(-s) if i != idx[c]][:5]
        print(c, "最像的字：", " ".join(f"{m['chars'][i]}({s[i]:+.2f})" for i in top))


def seeds(n=3):
    """换随机种子重新训练，看分组是否稳定（每个种子约 40 秒）。"""
    sys.path.insert(0, str(GPT / "code"))
    import corpus
    from train import train

    chars = corpus.vocab()
    idx = {c: i for i, c in enumerate(chars)}
    pairs = ["猫狗", "猫鸟", "狗鸟", "兔熊", "猫兔", "狗熊", "熊小", "住喜", "住最"]
    print("种子 " + "  ".join(pairs))
    for s in range(1, n + 1):
        m = train(steps=3000, seed=s)[0]
        E = m.p["E"].astype(float)
        U = E / np.linalg.norm(E, axis=1, keepdims=True)
        print(f"{s:3d}  " + "  ".join(f"{U[idx[a]] @ U[idx[b]]:+.2f}" for a, b in pairs))


if __name__ == "__main__":
    if "--seeds" in sys.argv:
        seeds()
    else:
        hand()
        high_dim()
        embeddings()
