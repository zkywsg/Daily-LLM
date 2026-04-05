"""
dcgan_generator.py — DCGAN 生成器实现
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/segmentation-gan

练习目标:
  1. 实现单个转置卷积上采样块
  2. 实现完整 DCGAN Generator
  3. 实现 DCGAN Discriminator（镜像结构）
  4. 验证 GAN 训练循环的 loss 动态
"""

import torch
import torch.nn as nn
import math


# ── Step 1 · 转置卷积上采样块 ──────────────────────────────────
class UpBlock(nn.Module):
    """转置卷积上采样块: ConvTranspose2d → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Step 2 · DCGAN Generator ──────────────────────────────────
class DCGANGenerator(nn.Module):
    """DCGAN 生成器: z → (z_dim, 1, 1) → UpBlock × 4 → ConvTranspose2d(3) → Tanh

    Args:
        z_dim:   噪声向量维度
        base_ch: 基础通道数（每层翻倍/减半的基础）
    """

    def __init__(self, z_dim: int = 100, base_ch: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # (z_dim, 1, 1) → (base_ch*8, 4, 4)
            nn.ConvTranspose2d(z_dim, base_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(True),
            # → (base_ch*4, 8, 8)
            UpBlock(base_ch * 8, base_ch * 4),
            # → (base_ch*2, 16, 16)
            UpBlock(base_ch * 4, base_ch * 2),
            # → (base_ch, 32, 32)
            UpBlock(base_ch * 2, base_ch),
            # → (3, 64, 64)
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.view(z.size(0), self.z_dim, 1, 1)
        return self.net(x)


# ── Step 3 · DCGAN Discriminator ──────────────────────────────
class DCGANDiscriminator(nn.Module):
    """DCGAN 判别器: 镜像 Generator 结构。

    (3, 64, 64) → Conv(stride=2)×4 → Sigmoid
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (3, 64, 64) → (base_ch, 32, 32)
            nn.Conv2d(3, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (base_ch*2, 16, 16)
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # → (base_ch*4, 8, 8)
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # → (base_ch*8, 4, 4)
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # → (1, 1, 1) → Sigmoid
            nn.Conv2d(base_ch * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# ── Step 4 · GAN 训练循环验证 ─────────────────────────────────
def gan_train_step(
    G: nn.Module,
    D: nn.Module,
    optim_G: torch.optim.Optimizer,
    optim_D: torch.optim.Optimizer,
    real_images: torch.Tensor,
    z_dim: int,
) -> dict:
    """一步 GAN 训练: D 训练一步 + G 训练一步。

    Args:
        G: Generator
        D: Discriminator
        optim_G: Generator 优化器
        optim_D: Discriminator 优化器
        real_images: (B, 3, 64, 64) 真实图像
        z_dim: 噪声维度
    Returns:
        {"G_loss": float, "D_loss": float}
    """
    batch_size = real_images.size(0)
    device = real_images.device
    criterion = nn.BCELoss()
    real_label = torch.ones(batch_size, device=device)
    fake_label = torch.zeros(batch_size, device=device)

    # ── 训练 D ──
    optim_D.zero_grad()
    # 真实图像
    d_real = D(real_images)
    loss_d_real = criterion(d_real, real_label)
    # 生成假图像（detach 防止梯度流向 G）
    z = torch.randn(batch_size, z_dim, device=device)
    fake_images = G(z).detach()
    d_fake = D(fake_images)
    loss_d_fake = criterion(d_fake, fake_label)
    d_loss = loss_d_real + loss_d_fake
    d_loss.backward()
    optim_D.step()

    # ── 训练 G ──
    optim_G.zero_grad()
    z = torch.randn(batch_size, z_dim, device=device)
    fake_images = G(z)
    d_fake_for_g = D(fake_images)
    g_loss = criterion(d_fake_for_g, real_label)  # G 希望 D 认为假图是真的
    g_loss.backward()
    optim_G.step()

    return {"G_loss": g_loss.item(), "D_loss": d_loss.item()}


# ── 测试 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    Z_DIM = 100
    BASE_CH = 64

    # Step 1 测试: UpBlock
    up = UpBlock(512, 256)
    x = torch.randn(2, 512, 2, 2)
    out = up(x)
    assert out.shape == (2, 256, 4, 4), f"UpBlock shape mismatch: {out.shape}"
    print(f"Step 1 ✓ UpBlock: {x.shape} → {out.shape}")

    # Step 2 测试: Generator
    G = DCGANGenerator(z_dim=Z_DIM, base_ch=BASE_CH)
    z = torch.randn(2, Z_DIM)
    fake = G(z)
    assert fake.shape == (2, 3, 64, 64), f"Generator shape mismatch: {fake.shape}"
    assert fake.min() >= -1.0 and fake.max() <= 1.0, "Output should be in [-1, 1] (Tanh)"
    print(f"Step 2 ✓ Generator: z{tuple(z.shape)} → {tuple(fake.shape)}, range=[{fake.min():.2f}, {fake.max():.2f}]")

    # Step 3 测试: Discriminator
    D = DCGANDiscriminator(base_ch=BASE_CH)
    real = torch.randn(2, 3, 64, 64)
    d_out = D(real)
    assert d_out.shape == (2,), f"Discriminator shape mismatch: {d_out.shape}"
    assert (d_out >= 0).all() and (d_out <= 1).all(), "D output should be in [0, 1]"
    print(f"Step 3 ✓ Discriminator: {tuple(real.shape)} → {tuple(d_out.shape)}, range=[{d_out.min():.3f}, {d_out.max():.3f}]")

    # Step 4 测试: GAN 训练步
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    real_imgs = torch.randn(4, 3, 64, 64)
    result = gan_train_step(G, D, optim_G, optim_D, real_imgs, Z_DIM)
    assert math.isfinite(result["G_loss"]), f"G_loss not finite: {result['G_loss']}"
    assert math.isfinite(result["D_loss"]), f"D_loss not finite: {result['D_loss']}"
    assert result["G_loss"] > 0, "G_loss should be positive"
    assert result["D_loss"] > 0, "D_loss should be positive"
    print(f"Step 4 ✓ GAN train step: G_loss={result['G_loss']:.4f}, D_loss={result['D_loss']:.4f}")
    print("\n所有测试通过 ✓")
