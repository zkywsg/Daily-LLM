"""轻量化卷积与注意力 · tracks/vision/lightweight-vision/exercises/depthwise_separable.py · 深度可分离卷积、SE-Block 到 MobileNet 模块 · torch"""

import torch
import torch.nn as nn

torch.manual_seed(42)


# ══════════════════════════════════════════════
# Step 1  标准卷积 vs 深度可分离卷积
# ══════════════════════════════════════════════

def count_params(module):
    return sum(p.numel() for p in module.parameters())


def step1_depthwise_separable():
    """对比标准卷积与深度可分离卷积的参数量。"""
    IN_CH, OUT_CH, K = 64, 128, 3

    # 标准卷积: 每个 out_ch 都与所有 in_ch * K * K 的权重连接
    std_conv = nn.Conv2d(IN_CH, OUT_CH, K, padding=1, bias=False)
    std_params = count_params(std_conv)
    expected_std = OUT_CH * IN_CH * K * K  # 128 * 64 * 9 = 73728
    assert std_params == expected_std

    # 深度可分离卷积 = Depthwise Conv + Pointwise Conv
    #   Depthwise: 每个 in_ch 独立用一个 K*K 卷积核
    #   Pointwise: 1x1 conv 跨通道混合
    depthwise = nn.Conv2d(IN_CH, IN_CH, K, padding=1, groups=IN_CH, bias=False)
    pointwise = nn.Conv2d(IN_CH, OUT_CH, 1, bias=False)
    ds_params = count_params(depthwise) + count_params(pointwise)
    expected_ds = IN_CH * K * K + IN_CH * OUT_CH  # 64*9 + 64*128 = 576 + 8192 = 8768
    assert ds_params == expected_ds

    # 深度可分离的参数量远小于标准卷积
    ratio = ds_params / std_params
    assert ratio < 0.2, f"Depthwise separable should use <20% params, got {ratio:.2%}"

    # 功能验证: 输出 shape 相同
    x = torch.randn(2, IN_CH, 16, 16)
    out_std = std_conv(x)
    out_ds = pointwise(depthwise(x))
    assert out_std.shape == out_ds.shape == (2, OUT_CH, 16, 16)

    print(f"[Step 1] depthwise separable  std_params={std_params:,}  ds_params={ds_params:,}  ratio={ratio:.2%}")


# ══════════════════════════════════════════════
# Step 2  逐点卷积 — 通道降维与升维
# ══════════════════════════════════════════════
def step2_pointwise_conv():
    """验证 1x1 卷积的通道变换能力（瓶颈结构的核心）。"""
    IN_CH = 256
    MID_CH = 64   # 降维 4x

    # 瓶颈: 256 → 64 → 256，通过 1x1 卷积实现
    bottleneck = nn.Sequential(
        nn.Conv2d(IN_CH, MID_CH, 1, bias=False),  # 降维
        nn.BatchNorm2d(MID_CH),
        nn.ReLU(inplace=True),
        nn.Conv2d(MID_CH, IN_CH, 1, bias=False),  # 升维
        nn.BatchNorm2d(IN_CH),
    )

    x = torch.randn(2, 256, 16, 16)
    out = bottleneck(x)
    assert out.shape == (2, 256, 16, 16), f"Shape should be preserved, got {out.shape}"

    # 瓶颈参数量远小于直接 3x3 卷积
    bn_params = count_params(bottleneck)
    conv3x3_params = IN_CH * IN_CH * 9  # 假设无 bias
    ratio = bn_params / conv3x3_params
    assert ratio < 0.15, f"Bottleneck should be much cheaper, ratio={ratio:.2%}"

    print(f"[Step 2] pointwise bottleneck  params={bn_params:,}  vs 3x3={conv3x3_params:,}  ratio={ratio:.2%}")


# ══════════════════════════════════════════════
# Step 3  SE-Block — Squeeze-and-Excitation 通道注意力
# ══════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 全局池化 → FC 降维 → ReLU → FC 升维 → Sigmoid → 重标定。

    让网络自适应地学习"哪些通道更重要"。
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)       # (B, C, H, W) -> (B, C, 1, 1)
        self.excite = nn.Sequential(
            nn.Flatten(),                              # (B, C)
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),                              # (B, C) 每个通道一个权重
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.excite(s)
        return x * s[:, :, None, None]  # 通道级乘法重标定


def step3_se_block():
    """验证 SE-Block 的通道注意力行为。"""
    se = SEBlock(channels=64, reduction=4)

    x = torch.randn(2, 64, 8, 8)
    out = se(x)
    assert out.shape == x.shape, f"SE should preserve shape: {out.shape} vs {x.shape}"

    # 输出应不同于输入（经过注意力重标定）
    assert not torch.allclose(out, x), "SE output should differ from input"

    # 验证注意力权重在 (0, 1) 之间
    with torch.no_grad():
        squeezed = se.squeeze(x)  # (B, C, 1, 1)
        weights = se.excite(squeezed)  # full: flatten → fc1 → relu → fc2 → sigmoid
    assert (weights >= 0).all() and (weights <= 1).all(), "Sigmoid weights should be in [0,1]"

    # SE 参数量很小
    se_params = count_params(se)
    print(f"[Step 3] SE block  params={se_params:,}  weights_range=[{weights.min():.3f}, {weights.max():.3f}]")


# ══════════════════════════════════════════════
# Step 4  MobileNet 风格模块
# ══════════════════════════════════════════════
class InvertedResidual(nn.Module):
    """MobileNetV2 逆残差块: 1x1 扩展 → Depthwise 3x3 → 1x1 压缩 + SE + 残差。

    核心: 先扩展通道（增加表达力），再用深度卷积提取空间特征，
    最后压缩回低维，比标准残差块参数更少。
    """

    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1, use_se=True):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        # 1. 扩展 (Pointwise)
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        ) if expand_ratio > 1 else nn.Identity()

        # 2. 深度卷积 (Depthwise)
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        )

        # 3. SE 注意力
        self.se = SEBlock(mid_ch, reduction=4) if use_se else nn.Identity()

        # 4. 压缩 (Pointwise Linear)
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class MiniMobileNet(nn.Module):
    """迷你 MobileNet 风格分类网络。"""

    def __init__(self, n_class=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            InvertedResidual(16, 16, expand_ratio=2, stride=1),
            InvertedResidual(16, 32, expand_ratio=3, stride=2),
            InvertedResidual(32, 32, expand_ratio=3, stride=1),
            InvertedResidual(32, 64, expand_ratio=4, stride=2),
            InvertedResidual(64, 64, expand_ratio=4, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_class),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def step4_mobile_net():
    """验证迷你 MobileNet 的参数效率和训练能力。"""
    mobilenet = MiniMobileNet(n_class=10)
    params_mobilenet = count_params(mobilenet)

    # 对比一个标准 CNN（3 个 conv3x3 块 + GAP + FC）的参数量
    standard_cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(128, 10),
    )
    params_standard = count_params(standard_cnn)

    print(f"[Step 4] MobileNet params={params_mobilenet:,}  vs Standard CNN params={params_standard:,}")

    # 训练验证
    optimizer = torch.optim.Adam(mobilenet.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))

    losses = []
    mobilenet.train()
    for _ in range(30):
        logits = mobilenet(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"         training  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


# ══════════════════════════════════════════════
# 运行所有步骤
# ══════════════════════════════════════════════
if __name__ == "__main__":
    step1_depthwise_separable()
    step2_pointwise_conv()
    step3_se_block()
    step4_mobile_net()
    print("\nAll steps passed!")
