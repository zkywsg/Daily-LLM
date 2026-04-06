"""
vae.py — 变分自编码器（VAE）实现
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/segmentation-gan

练习目标:
  1. 实现 VAE 编码器（输出 μ 和 log_var）
  2. 实现重参数化技巧（让采样可微）
  3. 实现 VAE 解码器
  4. 组装完整 VAE 并实现 ELBO 损失（重建 + KL 散度）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Step 1 · VAE 编码器 ──────────────────────────────────
class VAEEncoder(nn.Module):
    """编码器: 输入图像 → μ, log_var（潜空间分布参数）

    结构: Conv 下采样 → Flatten → FC → μ / log_var

    Args:
        in_channels: 输入图像通道数
        latent_dim:  潜空间维度
        base_ch:     基础通道数
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32, base_ch: int = 32):
        super().__init__()
        # 卷积下采样: 28×28 → 14×14 → 7×7
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 4, 2, 1),      # → (base_ch, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),      # → (base_ch*2, 7, 7)
            nn.ReLU(True),
        )
        # 7×7×base_ch*2 = 7*7*64 = 3136（base_ch=32 时）
        self.flat_dim = base_ch * 2 * 7 * 7
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = h.view(h.size(0), -1)            # Flatten
        mu = self.fc_mu(h)                    # 均值
        log_var = self.fc_log_var(h)          # 对数方差（数值稳定，避免负数）
        return mu, log_var


# ── Step 2 · 重参数化技巧 ──────────────────────────────────
def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """重参数化: z = μ + σ ⊙ ε, ε ~ N(0, I)

    把随机性转移到外部变量 ε，使 μ 和 σ 可参与梯度计算。
    log_var 而非 var 是为了数值稳定性（保证 σ > 0）。
    """
    std = torch.exp(0.5 * log_var)           # σ = exp(0.5 · log(σ²))
    eps = torch.randn_like(std)               # ε ~ N(0, I)
    return mu + std * eps                      # z = μ + σε


# ── Step 3 · VAE 解码器 ──────────────────────────────────
class VAEDecoder(nn.Module):
    """解码器: 潜变量 z → 重建图像

    结构: FC → Reshape → 转置卷积上采样

    Args:
        latent_dim:   潜空间维度
        out_channels: 输出图像通道数
        base_ch:      基础通道数
    """

    def __init__(self, latent_dim: int = 32, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.flat_dim = base_ch * 2 * 7 * 7
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.base_ch = base_ch

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),  # → (base_ch, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch, out_channels, 4, 2, 1),  # → (out_channels, 28, 28)
            nn.Sigmoid(),  # 输出 [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self.base_ch * 2, 7, 7)  # Reshape
        return self.deconv(h)


# ── Step 4 · 完整 VAE + ELBO 损失 ──────────────────────────
class VAE(nn.Module):
    """变分自编码器: Encoder → Reparameterize → Decoder

    Args:
        in_channels: 输入图像通道数
        latent_dim:  潜空间维度
        base_ch:     基础通道数
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 32, base_ch: int = 32):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, base_ch)
        self.decoder = VAEDecoder(latent_dim, in_channels, base_ch)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor,
                      mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        """ELBO 损失 = 重建损失 + KL 散度

        - 重建损失: BCE（逐像素二值交叉熵）
        - KL 散度: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
          推导: D_KL(N(μ,σ²) || N(0,1)) 的解析解
        """
        # 重建损失: 逐像素 BCE，sum over pixels
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL 散度: 对每个潜变量维度求和
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        total_loss = recon_loss + kl_loss
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }


# ── 自测 ─────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 8
    x = torch.rand(batch_size, 1, 28, 28)  # 模拟 MNIST 图像

    # Step 1: 编码器
    encoder = VAEEncoder(in_channels=1, latent_dim=32, base_ch=32)
    mu, log_var = encoder(x)
    assert mu.shape == (batch_size, 32), f"μ shape 错误: {mu.shape}"
    assert log_var.shape == (batch_size, 32), f"log_var shape 错误: {log_var.shape}"
    print(f"Step 1 ✓ 编码器输出: μ {mu.shape}, log_var {log_var.shape}")

    # Step 2: 重参数化
    z = reparameterize(mu, log_var)
    assert z.shape == (batch_size, 32), f"z shape 错误: {z.shape}"
    assert z.requires_grad, "重参数化后梯度丢失"
    print(f"Step 2 ✓ 重参数化: z {z.shape}, 可微={z.requires_grad}")

    # Step 3: 解码器
    decoder = VAEDecoder(latent_dim=32, out_channels=1, base_ch=32)
    x_recon = decoder(z)
    assert x_recon.shape == (batch_size, 1, 28, 28), f"重建 shape 错误: {x_recon.shape}"
    assert x_recon.min() >= 0 and x_recon.max() <= 1, "Sigmoid 输出应在 [0,1]"
    print(f"Step 3 ✓ 解码器输出: {x_recon.shape}, 范围 [{x_recon.min():.3f}, {x_recon.max():.3f}]")

    # Step 4: 完整 VAE + 损失
    vae = VAE(in_channels=1, latent_dim=32, base_ch=32)
    x_recon, mu, log_var = vae(x)
    losses = vae.loss_function(x, x_recon, mu, log_var)
    assert 'loss' in losses and 'recon_loss' in losses and 'kl_loss' in losses
    assert losses['loss'].requires_grad, "总损失应可微"
    print(f"Step 4 ✓ VAE 损失: total={losses['loss'].item():.1f}, "
          f"recon={losses['recon_loss'].item():.1f}, kl={losses['kl_loss'].item():.1f}")

    # 验证梯度可以回传
    losses['loss'].backward()
    assert vae.encoder.fc_mu.weight.grad is not None, "编码器梯度为空"
    assert vae.decoder.fc.weight.grad is not None, "解码器梯度为空"
    print("Step 4 ✓ 梯度回传正常")

    # 验证从潜空间采样生成
    with torch.no_grad():
        z_sample = torch.randn(4, 32)
        generated = vae.decoder(z_sample)
        assert generated.shape == (4, 1, 28, 28)
    print(f"Step 4 ✓ 采样生成: {generated.shape}")

    print("\n全部测试通过 ✓")
