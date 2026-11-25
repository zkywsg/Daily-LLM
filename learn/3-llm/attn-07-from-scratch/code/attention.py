"""从零实现注意力（NumPy）· learn/3-llm/attn-07-from-scratch/code
只依赖 numpy。每个函数的形状写在注释里。对应讲解见上一级 README.md。
"""
import numpy as np


def softmax(z, axis=-1):
    """数值稳定的 softmax：先减去最大值，结果不变但不会溢出。"""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """[n, n] 布尔矩阵，True 表示“要屏蔽”（位置 j > i，即看未来）。"""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K: [..., n, dk]；V: [..., n, dv]；mask: 可广播到 [..., n, n]，True=屏蔽。
    返回 (输出 [..., n, dv], 权重 [..., n, n])。"""
    dk = Q.shape[-1]
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)      # [..., n, n]
    if mask is not None:
        S = np.where(mask, -np.inf, S)                # 未来位置设为 -inf
    A = softmax(S, axis=-1)                           # 每行和为 1
    return A @ V, A


def multi_head_attention(X, Wq, Wk, Wv, Wo, h, mask=None):
    """X: [B, n, d]；Wq/Wk/Wv/Wo: [d, d]；h: 头数。返回 (输出 [B, n, d], 权重 [B, h, n, n])。"""
    B, n, d = X.shape
    assert d % h == 0, "d_model 必须能被头数整除"
    dk = d // h

    def split(M):                                      # [B, n, d] -> [B, h, n, dk]
        return M.reshape(B, n, h, dk).transpose(0, 2, 1, 3)

    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    O, A = scaled_dot_product_attention(Q, K, V, mask)  # O: [B, h, n, dk]
    O = O.transpose(0, 2, 1, 3).reshape(B, n, d)        # 拼接各头 -> [B, n, d]
    return O @ Wo, A


def sinusoidal_positions(n, d, base=10000.0):
    """论文式正弦位置编码，返回 [n, d]（d 为偶数）。"""
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / np.power(base, 2 * i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe
