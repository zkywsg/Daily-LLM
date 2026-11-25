"""测试：用性质与朴素写法检验 attention.py。运行：python test_attention.py"""
import numpy as np
from attention import (softmax, causal_mask, scaled_dot_product_attention,
                       multi_head_attention, sinusoidal_positions)

rng = np.random.default_rng(0)


def test_softmax_rows_sum_to_one_and_shift_invariant():
    z = rng.normal(size=(4, 7))
    assert np.allclose(softmax(z).sum(-1), 1)
    assert np.allclose(softmax(z), softmax(z + 123.4))          # 平移不变


def test_softmax_no_overflow():
    z = np.array([[1000.0, 1001.0, 999.0]])
    out = softmax(z)
    assert np.all(np.isfinite(out)) and np.isclose(out.sum(), 1)  # 不加“减最大值”会得到 nan


def test_matches_hand_calculation():
    # 与 attn-02 手算题一致
    X = np.array([[1, 0], [0, 1], [1, 1]], float)
    Wq = np.array([[1, 0], [1, 1]], float)
    Wk = np.array([[1, 1], [0, 1]], float)
    Wv = np.array([[1, 0], [0, 2]], float)
    O, A = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)
    A_hand = np.array([[.401, .198, .401], [.284, .140, .576], [.306, .074, .620]])
    O_hand = np.array([[.802, 1.198], [.860, 1.432], [.926, 1.389]])
    assert np.allclose(A, A_hand, atol=1e-3) and np.allclose(O, O_hand, atol=1e-3)


def naive_multihead(X, Wq, Wk, Wv, Wo, h):
    """朴素写法：逐句、逐头用循环算，再拼接。用来对照向量化版本。"""
    B, n, d = X.shape
    dk = d // h
    out = np.zeros((B, n, d))
    for b in range(B):
        heads = []
        for i in range(h):
            sl = slice(i * dk, (i + 1) * dk)
            q, k, v = X[b] @ Wq[:, sl], X[b] @ Wk[:, sl], X[b] @ Wv[:, sl]
            s = q @ k.T / np.sqrt(dk)
            a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            heads.append(a @ v)
        out[b] = np.concatenate(heads, axis=-1) @ Wo
    return out


def test_multihead_matches_naive():
    B, n, d, h = 3, 5, 8, 4
    X = rng.normal(size=(B, n, d))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) for _ in range(4))
    out, A = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    assert out.shape == (B, n, d) and A.shape == (B, h, n, n)
    assert np.allclose(out, naive_multihead(X, Wq, Wk, Wv, Wo, h))


def test_causal_mask_zero_future_and_no_leak():
    B, n, d, h = 2, 6, 8, 2
    X = rng.normal(size=(B, n, d))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) for _ in range(4))
    m = causal_mask(n)
    out, A = multi_head_attention(X, Wq, Wk, Wv, Wo, h, mask=m)
    assert np.allclose(A[..., m], 0)                               # 未来位置权重为 0
    assert np.allclose(A.sum(-1), 1)                               # 每行仍和为 1
    X2 = X.copy(); X2[:, 4:, :] += rng.normal(size=(B, n - 4, d))  # 只改“未来”的词
    out2, _ = multi_head_attention(X2, Wq, Wk, Wv, Wo, h, mask=m)
    assert np.allclose(out[:, :4], out2[:, :4])                    # 前 4 个位置的输出完全不变


def test_permutation_equivariance_without_position():
    B, n, d, h = 1, 6, 8, 2
    X = rng.normal(size=(B, n, d))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) for _ in range(4))
    perm = rng.permutation(n)
    out, _ = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    out_p, _ = multi_head_attention(X[:, perm], Wq, Wk, Wv, Wo, h)
    assert np.allclose(out[:, perm], out_p)                        # 打乱输入 = 输出同样被打乱
    pe = sinusoidal_positions(n, d)
    out_pe, _ = multi_head_attention(X + pe, Wq, Wk, Wv, Wo, h)
    out_pe_p, _ = multi_head_attention(X[:, perm] + pe, Wq, Wk, Wv, Wo, h)
    assert not np.allclose(out_pe[:, perm], out_pe_p)              # 加了位置编码，顺序就“看得见”了


def test_scaling_flattens_weights_for_large_d():
    d, n, trials = 256, 8, 400
    un = sc = 0.0
    for _ in range(trials):
        q, K = rng.normal(size=d), rng.normal(size=(n, d))
        s = (K * q).sum(-1)   # 等价于 K @ q；这里用逐元素求和以避开部分 macOS NumPy 的误报警告
        un += softmax(s).max(); sc += softmax(s / np.sqrt(d)).max()
    assert un / trials > 0.85 and sc / trials < 0.5               # 不缩放几乎“一家独大”


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("通过", t.__name__)
    print(f"全部 {len(tests)} 个测试通过")
