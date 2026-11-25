"""分块在线 softmax：不一次性看到整行分数，也能得到与普通 softmax 完全相同的结果。
这是 FlashAttention 的核心数学技巧（Milakov & Gimelshein 2018；Dao et al. 2022 用它做分块）。
用法：python online_softmax.py   （只需要 numpy）"""
import numpy as np

def softmax_attend(s, v):
    e = np.exp(s - s.max())
    return (e / e.sum()) @ v

def online_attend(s, v, block):
    m, l, o = -np.inf, 0.0, np.zeros_like(v[0])
    for a in range(0, len(s), block):
        sb, vb = s[a:a + block], v[a:a + block]
        m_new = max(m, sb.max())
        scale = np.exp(m - m_new)              # 旧的累计量按新最大值重新缩放
        p = np.exp(sb - m_new)
        l = l * scale + p.sum()
        o = o * scale + p @ vb
        m = m_new
    return o / l

rng = np.random.default_rng(0)
s = rng.standard_normal(64) * 5
v = rng.standard_normal((64, 8))
ref = softmax_attend(s, v)
for b in (1, 4, 16, 64):
    print(f"block={b:>2}  max|diff| = {np.abs(online_attend(s, v, b) - ref).max():.2e}")

# KV cache 次数对比：生成 n 个词，每步每个头
n = 100
with_cache = sum(range(1, n + 1))                 # 第 t 步：新 q 对 t 个 k
without = sum(t * t for t in range(1, n + 1))     # 第 t 步：重算 t×t 的分数
print("q·k 次数  有缓存:", with_cache, " 无缓存:", without, " 比值:", round(without / with_cache, 1))
