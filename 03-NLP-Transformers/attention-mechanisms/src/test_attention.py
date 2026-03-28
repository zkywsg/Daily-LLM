"""
注意力机制单元测试 · 03-NLP-Transformers/attention-mechanisms

依赖：
    torch>=2.0, pytest>=7.0
运行：
    pytest src/test_attention.py -v
"""

import math
import pytest
import torch
from attention import (
    MultiHeadAttention,
    make_causal_mask,
    make_padding_mask,
    scaled_dot_product_attention,
)

torch.manual_seed(42)


# ─── scaled_dot_product_attention ────────────────────────────────────────────

class TestScaledDotProductAttention:
    def test_output_shape(self):
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 10, 32)
        v = torch.randn(2, 4, 10, 64)
        ctx, w = scaled_dot_product_attention(q, k, v, use_flash=False)
        assert ctx.shape == (2, 4, 8, 64)
        assert w.shape == (2, 4, 8, 10)

    def test_weights_sum_to_one(self):
        q = torch.randn(1, 1, 6, 16)
        k = torch.randn(1, 1, 6, 16)
        v = torch.randn(1, 1, 6, 16)
        _, w = scaled_dot_product_attention(q, k, v, use_flash=False)
        assert torch.allclose(w.sum(dim=-1), torch.ones(1, 1, 6), atol=1e-5)

    def test_causal_mask_zeroes_upper_triangle(self):
        SEQ = 5
        mask = make_causal_mask(SEQ)
        q = torch.randn(1, 1, SEQ, 16)
        k = torch.randn(1, 1, SEQ, 16)
        v = torch.randn(1, 1, SEQ, 16)
        _, w = scaled_dot_product_attention(q, k, v, mask=mask, use_flash=False)
        upper = torch.triu(torch.ones(SEQ, SEQ), diagonal=1).bool()
        assert (w[0, 0][upper] == 0).all(), "Causal mask 未正确遮蔽上三角"

    def test_scaling_prevents_softmax_saturation(self):
        """高维时，缩放前后 softmax 熵应有显著差异。"""
        torch.manual_seed(0)
        d_k = 512
        q = torch.randn(1, 1, 4, d_k)
        k = torch.randn(1, 1, 4, d_k)
        v = torch.randn(1, 1, 4, d_k)

        scores_unscaled = q @ k.transpose(-2, -1)
        scores_scaled   = scores_unscaled / math.sqrt(d_k)

        entropy = lambda s: -(torch.softmax(s, -1) * torch.log_softmax(s, -1)).sum(-1).mean()
        assert entropy(scores_scaled) > entropy(scores_unscaled), "缩放应提高 softmax 熵"


# ─── MultiHeadAttention ───────────────────────────────────────────────────────

class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(d_model=512, num_heads=8, use_flash=False)

    def test_output_shape(self, mha):
        x = torch.randn(2, 10, 512)
        out, w = mha(x, x, x)
        assert out.shape == (2, 10, 512)
        assert w.shape == (2, 8, 10, 10)

    def test_cross_attention_shape(self, mha):
        q = torch.randn(2, 6, 512)
        kv = torch.randn(2, 12, 512)
        out, w = mha(q, kv, kv)
        assert out.shape == (2, 6, 512)
        assert w.shape == (2, 8, 6, 12)

    def test_invalid_d_model_raises(self):
        with pytest.raises(ValueError, match="整除"):
            MultiHeadAttention(d_model=100, num_heads=8)

    def test_parameter_count(self, mha):
        # 4 个线性层，每个 d_model × d_model = 512 × 512
        expected = 4 * 512 * 512
        actual = sum(p.numel() for p in mha.parameters())
        assert actual == expected, f"参数量 {actual} ≠ {expected}"

    def test_different_inputs_give_different_outputs(self, mha):
        x1 = torch.randn(1, 8, 512)
        x2 = torch.randn(1, 8, 512)
        out1, _ = mha(x1, x1, x1)
        out2, _ = mha(x2, x2, x2)
        assert not torch.allclose(out1, out2)


# ─── Mask 工具 ────────────────────────────────────────────────────────────────

class TestMasks:
    def test_causal_mask_shape(self):
        mask = make_causal_mask(10)
        assert mask.shape == (1, 1, 10, 10)

    def test_causal_mask_is_lower_triangular(self):
        mask = make_causal_mask(5)[0, 0]
        expected = torch.tril(torch.ones(5, 5))
        assert torch.equal(mask, expected)

    def test_padding_mask_shape(self):
        lengths = torch.tensor([3, 5, 4])
        mask = make_padding_mask(lengths, max_len=6)
        assert mask.shape == (3, 1, 1, 6)

    def test_padding_mask_values(self):
        lengths = torch.tensor([2, 4])
        mask = make_padding_mask(lengths, max_len=4)[..., 0, :]
        # batch 0: [1,1,0,0], batch 1: [1,1,1,1]
        assert mask[0, 0].tolist() == [True, True, False, False]
        assert mask[1, 0].tolist() == [True, True, True, True]
