"""最优预测器：知道造句规则的情况下，同样的验证批次上损失最低能做到多少。
它只看窗口里的字（和模型一样），用贝叶斯更新维护“现在处在哪句话的第几个字”。"""
import numpy as np

from corpus import ANIMALS, TEMPLATES, sentence
from train import CHARS, batch, val_ids

SENTS = [sentence(s, t) for s in ANIMALS for t in range(len(TEMPLATES))]


def optimal_loss(n_batches=20, ctx=24):
    states = [(i, k) for i, s in enumerate(SENTS) for k in range(len(s))]
    ix = {st: n for n, st in enumerate(states)}
    char = np.array([CHARS.index(SENTS[i][k]) for i, k in states])
    T = np.zeros((len(states), len(states)))  # 状态转移
    for (i, k), n in ix.items():
        if k + 1 < len(SENTS[i]):
            T[n, ix[(i, k + 1)]] = 1
        else:
            for j in range(len(SENTS)):
                T[n, ix[(j, 0)]] = 1 / len(SENTS)
    start = np.full(len(states), 1 / len(states))  # 窗口起点随机落在语料任意位置
    rng = np.random.default_rng(123)
    total, count = 0.0, 0
    for _ in range(n_batches):
        X, Y = batch(val_ids, rng, 64, ctx)
        for x, y in zip(X, Y):
            b = start
            for t in range(ctx):
                b = b * (char == x[t])
                b = b / b.sum()
                b = np.einsum("i,ij->j", b, T)
                total -= np.log(b[char == y[t]].sum())
                count += 1
    return total / count


if __name__ == "__main__":
    print(f"同样验证批次上的最优损失：{optimal_loss():.4f}")
