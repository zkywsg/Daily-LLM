"""梯度检查：手写反向传播 vs 数值差分。运行：python test_grad.py"""
import numpy as np
from minigpt import MiniGPT


def check(**kw):
    m = MiniGPT(vocab=11, d=8, n_layers=2, n_heads=2, d_ff=16, ctx=6, seed=1, dtype=np.float64, **kw)
    rng = np.random.default_rng(0)
    for k in m.p:  # 放大初始值，让梯度不至于太小
        m.p[k] = m.p[k] + rng.standard_normal(m.p[k].shape) * 0.3
    x, y = rng.integers(0, 11, (3, 6)), rng.integers(0, 11, (3, 6))
    _, G = m.loss_and_grads(x, y)
    assert all(np.isfinite(g).all() for g in G.values()), "梯度里出现了 inf / nan"
    worst = 0.0
    for k, v in m.p.items():
        for _ in range(6):
            i = tuple(rng.integers(0, s) for s in v.shape)
            old, h = v[i], 1e-5
            v[i] = old + h; a = m.loss_and_grads(x, y)[0]
            v[i] = old - h; b = m.loss_and_grads(x, y)[0]
            v[i] = old
            num, ana = (a - b) / (2 * h), G[k][i]
            err = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
            worst = max(worst, err)
    return worst


if __name__ == "__main__":
    for name, kw in [("默认", {}), ("无因果掩码", {"causal": False}), ("无位置向量", {"use_pos": False}), ("无残差", {"residual": False})]:
        w = check(**kw)
        print(f"{'通过' if w < 1e-4 else '失败'}  {name}：最大相对误差 {w:.1e}")
        assert w < 1e-4
    # 因果性：改动后面的字，前面位置的输出不应变化
    m = MiniGPT(vocab=11, d=8, n_layers=2, n_heads=2, d_ff=16, ctx=6, dtype=np.float64)
    x = np.array([[1, 2, 3, 4, 5, 6]]); x2 = x.copy(); x2[0, 4] = 9
    assert np.allclose(m.forward(x)[0][0, :4], m.forward(x2)[0][0, :4])
    print("通过  因果性：改第 5 个字，前 4 个位置输出不变")
