"""PyTorch 版：并与官方 torch.nn.functional.scaled_dot_product_attention 对照。
注意：作者的机器没有安装 torch，这个文件【没有运行过】。第一次运行时请自己确认通过。
运行：python attention_torch.py
"""
import math
import torch
import torch.nn.functional as F


def attention(q, k, v, causal=False):
    # q, k, v: [B, h, n, dk]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))       # [B, h, n, n]
    if causal:
        n = q.size(-2)
        s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    return s.softmax(dim=-1) @ v                              # [B, h, n, dk]


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 6, 16) for _ in range(3))
    for causal in (False, True):
        mine = attention(q, k, v, causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        print("causal =", causal, "最大误差", (mine - ref).abs().max().item())
        assert torch.allclose(mine, ref, atol=1e-5)
    print("与官方实现一致")
