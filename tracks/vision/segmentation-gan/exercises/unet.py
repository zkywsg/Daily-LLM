"""U-Net 编码器-解码器 · tracks/vision/segmentation-gan/exercises/unet.py · 转置卷积、跳跃连接到完整 U-Net · torch"""

import torch
import torch.nn as nn

torch.manual_seed(42)


# ══════════════════════════════════════════════
# Step 1  转置卷积 — 上采样
# ══════════════════════════════════════════════
def step1_transposed_conv():
    """理解转置卷积如何实现上采样。"""
    # 普通 Conv2d: (B, C_in, H, W) -> (B, C_out, H', W')  其中 H' <= H
    # ConvTranspose2d: 反向操作，空间尺寸增大

    # 下采样再上采样: 验证尺寸恢复
    down = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    up = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    x = torch.randn(1, 3, 64, 64)
    x_down = down(x)
    x_up = up(x_down)

    assert x_down.shape == (1, 16, 32, 32), f"Downsampled shape wrong: {x_down.shape}"
    assert x_up.shape == (1, 3, 64, 64), f"Upsampled shape wrong: {x_up.shape}"

    # 对比双线性插值上采样
    bilinear_up = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)
    x_bilinear = bilinear_up(x_down)
    assert x_bilinear.shape == (1, 16, 64, 64)

    # 转置卷积有可学习参数，双线性插值没有
    assert sum(p.numel() for p in up.parameters()) > 0
    assert sum(p.numel() for p in bilinear_up.parameters()) == 0

    print(f"[Step 1] transposed conv  {x.shape} -> down {x_down.shape} -> up {x_up.shape}")


# ══════════════════════════════════════════════
# Step 2  编码器-解码器骨架
# ══════════════════════════════════════════════
class EncoderBlock(nn.Module):
    """编码器块: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool2x2。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """返回 (特征图, 池化后特征图)。"""
        feat = self.conv(x)
        return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    """解码器块: 上采样 → Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


def step2_encoder_decoder():
    """验证编码器-解码器能恢复空间分辨率。"""
    enc1 = EncoderBlock(3, 16)
    enc2 = EncoderBlock(16, 32)
    dec1 = DecoderBlock(32, 16)
    dec2 = DecoderBlock(16, 3)

    x = torch.randn(2, 3, 64, 64)

    # 编码
    feat1, pooled1 = enc1(x)    # feat1: (2,16,64,64)  pooled1: (2,16,32,32)
    feat2, pooled2 = enc2(pooled1)  # feat2: (2,32,32,32)  pooled2: (2,32,16,16)

    assert feat1.shape == (2, 16, 64, 64), f"feat1 shape wrong: {feat1.shape}"
    assert pooled2.shape == (2, 32, 16, 16), f"pooled2 shape wrong: {pooled2.shape}"

    # 解码
    out_dec1 = dec1(pooled2)    # (2,16,32,32)
    out_dec2 = dec2(out_dec1)   # (2,3,64,64)

    assert out_dec1.shape == (2, 16, 32, 32), f"dec1 shape wrong: {out_dec1.shape}"
    assert out_dec2.shape == (2, 3, 64, 64), f"dec2 shape wrong: {out_dec2.shape}"

    print(f"[Step 2] encoder-decoder  {x.shape} -> bottleneck {pooled2.shape} -> output {out_dec2.shape}")


# ══════════════════════════════════════════════
# Step 3  跳跃连接 — U-Net 的核心
# ══════════════════════════════════════════════
class UNetDecoderBlock(nn.Module):
    """带跳跃连接的解码器块: 上采样 → concat(skip) → Conv。"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # in_ch for conv = out_ch (from up) + skip_ch (from skip connection)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不完全匹配的情况（中心裁剪 skip）
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            skip = skip[:, :, diff_h // 2: skip.shape[2] - (diff_h - diff_h // 2),
                               diff_w // 2: skip.shape[3] - (diff_w - diff_w // 2)]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def step3_skip_connection():
    """验证跳跃连接能融合多尺度信息。"""
    enc1 = EncoderBlock(3, 16)
    enc2 = EncoderBlock(16, 32)

    # 解码器带跳跃连接: dec1 从 bottleneck 上采样后与 feat2 (32x32) 拼接
    dec1 = UNetDecoderBlock(in_ch=32, skip_ch=32, out_ch=16)
    dec2 = UNetDecoderBlock(in_ch=16, skip_ch=16, out_ch=8)

    x = torch.randn(2, 3, 64, 64)

    # 编码
    feat1, pooled1 = enc1(x)      # feat1: (2,16,64,64)  pooled1: (2,16,32,32)
    feat2, pooled2 = enc2(pooled1) # feat2: (2,32,32,32)  pooled2: (2,32,16,16)

    # 解码（带跳跃连接）
    out1 = dec1(pooled2, feat2)    # up: (2,16,32,32) + skip feat2 (2,32,32,32) -> (2,16,32,32)
    out2 = dec2(out1, feat1)       # up: (2,8,64,64) + skip feat1 (2,16,64,64) -> (2,8,64,64)

    # 验证上采样后尺寸恢复到对应编码器级别
    assert out1.shape[2:] == feat2.shape[2:], f"dec1 output {out1.shape} should match feat2 spatial {feat2.shape}"
    assert out2.shape[2:] == feat1.shape[2:], f"dec2 output {out2.shape} should match feat1 spatial {feat1.shape}"

    # 对比有/无跳跃连接的信息量
    dec_no_skip = DecoderBlock(32, 16)
    out_no_skip = dec_no_skip(pooled2)

    # 跳跃连接版通道数更多（融合了 skip 信息）
    # 因为 cat 操作: out_ch + skip_ch -> out_ch
    print(f"[Step 3] skip connection  with_skip={out1.shape}  without_skip={out_no_skip.shape}")


# ══════════════════════════════════════════════
# Step 4  完整迷你 U-Net
# ══════════════════════════════════════════════
class MiniUNet(nn.Module):
    """3 层 U-Net: 编码→瓶颈→解码（带跳跃连接），输出像素级分类。"""

    def __init__(self, in_ch=3, n_class=5):
        super().__init__()
        # 编码器
        self.enc1 = EncoderBlock(in_ch, 16)
        self.enc2 = EncoderBlock(16, 32)

        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 解码器（带跳跃连接）
        self.dec2 = UNetDecoderBlock(in_ch=64, skip_ch=32, out_ch=32)
        self.dec1 = UNetDecoderBlock(in_ch=32, skip_ch=16, out_ch=16)

        # 分类头
        self.head = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        # 编码
        feat1, p1 = self.enc1(x)   # feat1: (B,16,H,W)
        feat2, p2 = self.enc2(p1)  # feat2: (B,32,H/2,W/2)

        # 瓶颈
        b = self.bottleneck(p2)    # (B,64,H/4,W/4)

        # 解码
        d2 = self.dec2(b, feat2)   # (B,32,H/2,W/2)
        d1 = self.dec1(d2, feat1)  # (B,16,H,W)

        return self.head(d1)       # (B,n_class,H,W)


def step4_mini_unet():
    """验证完整 U-Net 的端到端训练。"""
    model = MiniUNet(in_ch=3, n_class=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 合成分割数据: 输入图像 + 像素级标签
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 5, (4, 64, 64))  # (B, H, W)

    losses = []
    model.train()
    for _ in range(20):
        logits = model(x)                # (B, 5, 64, 64)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Step 4] mini U-Net  params={n_params:,}  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


# ══════════════════════════════════════════════
# 运行所有步骤
# ══════════════════════════════════════════════
if __name__ == "__main__":
    step1_transposed_conv()
    step2_encoder_decoder()
    step3_skip_connection()
    step4_mini_unet()
    print("\nAll steps passed!")
