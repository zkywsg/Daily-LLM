"""验证：concat(heads) @ Wo == sum_h  (A_h @ X @ Wv_h) @ Wo_h，并检查单头注意力矩阵的秩 <= d_k。
用法：python check_equivalence.py   （只需要 numpy）"""
import numpy as np

rng = np.random.default_rng(0)
n, d, h = 6, 8, 2
dk = d // h
X = rng.standard_normal((n, d))
Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))
Wo = rng.standard_normal((d, d))

def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)

heads, parts, scores = [], [], []
for i in range(h):
    sl = slice(i * dk, (i + 1) * dk)
    Q, K, V = X @ Wq[:, sl], X @ Wk[:, sl], X @ Wv[:, sl]
    S = Q @ K.T / np.sqrt(dk)
    A = softmax(S)
    O = A @ V                                   # [n, dk]
    heads.append(O)
    parts.append(O @ Wo[sl, :])                 # 第 i 个头对输出的贡献 [n, d]
    scores.append(S)

concat_out = np.concatenate(heads, axis=-1) @ Wo
sum_out = sum(parts)
print("max |concat·Wo - sum of heads| =", np.abs(concat_out - sum_out).max())
print("rank of each head's score matrix S (<= dk =", dk, "):", [int(np.linalg.matrix_rank(S)) for S in scores])

# 消融：去掉头 1，看输出变化
abl = parts[1]
print("ablate head 0: relative change =", np.linalg.norm(concat_out - abl) / np.linalg.norm(concat_out))
