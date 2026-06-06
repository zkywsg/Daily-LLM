"""注意力机制 · tracks/language/attention-mechanisms/src/attention.py · Scaled Dot-Product Attention 与 Multi-Head Attention 的生产级实现 · torch>=2.0"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# 按相关性做加权聚合
# softmax(QK^T / √d_k) @ V
# 时间 O(n²d)，空间 O(n²)
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    use_flash: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Scaled Dot-Product Attention。

    Args:
        q:         (batch, heads, seq_q, d_k)
        k:         (batch, heads, seq_k, d_k)
        v:         (batch, heads, seq_k, d_v)
        mask:      (batch, 1, 1, seq_k)，0 表示遮蔽
        dropout_p: 训练时 attention dropout 概率
        use_flash: True 时使用 PyTorch 内置 Flash Attention（不返回 weights）

    Returns:
        context: (batch, heads, seq_q, d_v)
        weights: (batch, heads, seq_q, seq_k) 或 None（use_flash=True 时）
    """
    if use_flash:
        # Flash Attention：显存 O(n)，速度快 2-4×，不返回中间权重
        attn_mask = None
        if mask is not None:
            # scaled_dot_product_attention 接受 bool mask（True=保留）
            attn_mask = mask.bool()
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        return context, None

    # 手写版：方便调试和可视化
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (batch, heads, seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v, weights


# ─── Multi-Head Attention ──────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    多头注意力。

    将 d_model 切分为 num_heads 个子空间，各自独立计算注意力后拼接输出。
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O
    其中 head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        """
        Args:
            d_model:   隐藏维度，必须能被 num_heads 整除
            num_heads: 注意力头数
            dropout:   attention dropout 概率
            use_flash: 使用 Flash Attention（PyTorch 2.0+）
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.use_flash = use_flash

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 初始化，稳定训练早期的梯度分布。"""
        for layer in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query:  (batch, seq_q, d_model)
            key:    (batch, seq_k, d_model)
            value:  (batch, seq_k, d_model)
            mask:   (batch, 1, 1, seq_k) 或 None
        Returns:
            out:     (batch, seq_q, d_model)
            weights: (batch, heads, seq_q, seq_k) 或 None
        """
        bsz, seq_q, _ = query.shape

        # 线性投影 + reshape 成多头形式
        q = self.w_q(query).view(bsz, seq_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)

        dp = self.dropout if self.training else 0.0
        context, weights = scaled_dot_product_attention(q, k, v, mask, dp, self.use_flash)

        # 拼接各头输出并投影
        context = context.transpose(1, 2).contiguous().view(bsz, seq_q, self.d_model)
        return self.w_o(context), weights

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"d_k={self.d_k}, flash={self.use_flash}"
        )


# ─── Causal Mask 工具 ─────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    生成因果掩码（下三角矩阵），用于自回归生成。

    Args:
        seq_len: 序列长度
        device:  目标设备
    Returns:
        mask: (1, 1, seq_len, seq_len)，1=可见，0=遮蔽
    """
    return torch.tril(torch.ones(1, 1, seq_len, seq_len, device=device))


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    根据实际序列长度生成 padding 掩码。

    Args:
        lengths: (batch,)，每条序列的实际长度
        max_len: 最大序列长度
    Returns:
        mask: (batch, 1, 1, max_len)，1=有效 token，0=padding
    """
    # arange (max_len,) < lengths (batch, 1) → (batch, max_len)
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, max_len)


# ─── 快速验证（直接运行此文件时执行）────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    D_MODEL = 512
    NUM_HEADS = 8
    BATCH = 2
    SEQ = 16

    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS, dropout=0.1)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    causal = make_causal_mask(SEQ)

    out, _ = mha(x, x, x, mask=causal)

    assert out.shape == (BATCH, SEQ, D_MODEL), f"shape 错误: {out.shape}"
    print(f"✓ 输出 shape 正确: {out.shape}")
    print(f"✓ 参数量: {sum(p.numel() for p in mha.parameters()):,}")
    print(mha)
