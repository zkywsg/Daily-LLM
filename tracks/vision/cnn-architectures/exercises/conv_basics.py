"""CNN 积木 · tracks/vision/cnn-architectures/exercises/conv_basics.py · 从单层卷积到迷你 VGG 的渐进练习 · torch"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ══════════════════════════════════════════════
# Step 1  单层卷积 — shape 与参数共享
# ══════════════════════════════════════════════
def step1_single_conv():
    """验证卷积核如何改变通道数和空间尺寸。"""
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(4, 3, 32, 32)  # (B, C, H, W)
    out = conv(x)

    # shape: 通道 3→16，H/W 不变（padding=1 补偿了 kernel=3 的缩小）
    assert out.shape == (4, 16, 32, 32), f"Expected (4,16,32,32), got {out.shape}"

    # 参数量: weight (out_ch, in_ch, kH, kW) + bias (out_ch)
    n_params = sum(p.numel() for p in conv.parameters())
    expected = 16 * 3 * 3 * 3 + 16  # 448
    assert n_params == expected, f"Expected {expected} params, got {n_params}"

    # 验证参数共享: 同一卷积核在所有空间位置复用
    # 不同位置使用相同的 kernel 权重
    print(f"[Step 1] conv2d  in={x.shape}  out={out.shape}  params={n_params}")


# ══════════════════════════════════════════════
# Step 2  卷积块 — Conv + BN + ReLU + 下采样
# ══════════════════════════════════════════════
def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """标准卷积块: Conv3x3 → BatchNorm → ReLU。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def step2_conv_block():
    """验证卷积块的组合效果和下采样行为。"""
    net = nn.Sequential(
        conv_block(3, 32),             # (B, 3, 32, 32) -> (B, 32, 32, 32)
        conv_block(32, 64, stride=2),  # -> (B, 64, 16, 16)
        nn.AdaptiveAvgPool2d(1),       # -> (B, 64, 1, 1)
        nn.Flatten(),                  # -> (B, 64)
    )

    x = torch.randn(4, 3, 32, 32)
    out = net(x)
    assert out.shape == (4, 64), f"Expected (4,64), got {out.shape}"

    # 统计参数量（不含 BN 的 running_mean/var 缓冲区）
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[Step 2] conv blocks  in={x.shape}  out={out.shape}  params={n_params}")


# ══════════════════════════════════════════════
# Step 3  感受野计算
# ══════════════════════════════════════════════
def compute_receptive_field(layers: list) -> int:
    """计算堆叠卷积层的感受野大小。

    每层贡献: r_new = r_old + (k - 1) * jump_old
    其中 jump = 所有 stride 的累积乘积。

    Args:
        layers: [(kernel_size, stride), ...]
    Returns:
        感受野大小（像素）
    """
    rf = 1
    jump = 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf


def step3_receptive_field():
    """验证感受野随层数增长。"""
    # 两个 3x3 stride=1 的卷积等效感受野 = 5
    rf_2layers = compute_receptive_field([(3, 1), (3, 1)])
    assert rf_2layers == 5, f"Expected rf=5, got {rf_2layers}"

    # 三个 3x3 stride=1 的卷积等效感受野 = 7
    rf_3layers = compute_receptive_field([(3, 1), (3, 1), (3, 1)])
    assert rf_3layers == 7, f"Expected rf=7, got {rf_3layers}"

    # 加入 stride=2 后感受野快速增长
    # (3,1): rf=1+2=3, jump=1; (3,2): rf=3+2=5, jump=2; (3,1): rf=5+2*2=9
    rf_stride = compute_receptive_field([(3, 1), (3, 2), (3, 1)])
    assert rf_stride == 9, f"Expected rf=9, got {rf_stride}"

    # VGG 风格: 4 个 3x3 + 2 个 stride=2
    rf_vgg = compute_receptive_field([
        (3, 1), (3, 1),        # block1
        (3, 2), (3, 1),        # block2 + downsample
        (3, 2), (3, 1),        # block3 + downsample
    ])
    print(f"[Step 3] receptive field  2-layer={rf_2layers}  3-layer={rf_3layers}  with-stride={rf_stride}  vgg-style={rf_vgg}")


# ══════════════════════════════════════════════
# Step 4  迷你 VGG — 端到端分类
# ══════════════════════════════════════════════
class MiniVGG(nn.Module):
    """简化版 VGG: 3 个卷积块 + 全局池化 + 分类头。"""

    def __init__(self, n_class: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # block1: 2 x conv3-32
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # block2: 2 x conv3-64
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # block3: 2 x conv3-128
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2, 2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def step4_mini_vgg():
    """验证迷你 VGG 的前向传播和训练收敛。"""
    model = MiniVGG(n_class=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 合成数据
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))

    # 训练 20 步，验证 loss 下降
    losses = []
    model.train()
    for _ in range(20):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Step 4] mini VGG  params={n_params:,}  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


# ══════════════════════════════════════════════
# 运行所有步骤
# ══════════════════════════════════════════════
if __name__ == "__main__":
    step1_single_conv()
    step2_conv_block()
    step3_receptive_field()
    step4_mini_vgg()
    print("\nAll steps passed!")
