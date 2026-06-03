"""CycleGAN 循环一致性 · tracks/vision/gan-advanced/exercises/cyclegan_cycle_consistency.py · CycleGAN 残差块、PatchGAN、循环一致性损失 · torch"""

import torch
import torch.nn as nn
import itertools


# ─── Residual Block ────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Residual block used in CycleGAN generator"""

    def __init__(self, channels):
        super().__init__()
        # TODO: Implement the residual block
        # Hint: Conv → InstanceNorm → ReLU → Conv → InstanceNorm
        #       Add input to output (skip connection)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ─── Generator ─────────────────────────────────────────────────
class CycleGANGenerator(nn.Module):
    """CycleGAN Generator: encoder → residual blocks → decoder"""

    def __init__(self, in_ch=3, out_ch=3, base_ch=64, n_residuals=9):
        super().__init__()
        # Encoder
        encoder = [
            nn.Conv2d(in_ch, base_ch, 7, padding=3, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True),
            # Downsampling
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(True),
        ]

        # Residual blocks
        residuals = [ResidualBlock(base_ch * 4) for _ in range(n_residuals)]

        # Decoder
        decoder = [
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True),
            nn.Conv2d(base_ch, out_ch, 7, padding=3, bias=False),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*encoder, *residuals, *decoder)

    def forward(self, x):
        return self.net(x)


# ─── PatchGAN Discriminator ────────────────────────────────────
class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN Discriminator for CycleGAN"""

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        # TODO: Build the PatchGAN discriminator
        # Hint: 3 conv layers with InstanceNorm + LeakyReLU
        #       Final layer outputs 1 channel (no norm, no activation)
        #       Input: (B, 3, 256, 256) → Output: (B, 1, 30, 30)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─── Loss Functions ────────────────────────────────────────────
class CycleGANLoss:
    """CycleGAN loss: adversarial + cycle consistency + identity"""

    def __init__(self, lambda_cyc=10.0, lambda_identity=5.0):
        self.lambda_cyc = lambda_cyc
        self.lambda_identity = lambda_identity
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def adversarial_loss(self, prediction, target_is_real):
        """LSGAN loss: least squares GAN for more stable training"""
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.mse(prediction, target)

    def cycle_consistency(self, real_A, real_B, G_A2B, G_B2A):
        """
        Cycle consistency: A→B→A ≈ A and B→A→B ≈ B

        TODO: Implement the cycle consistency loss
        Hint:
        1. Generate fake_B from real_A: fake_B = G_A2B(real_A)
        2. Reconstruct A from fake_B: rec_A = G_B2A(fake_B)
        3. Generate fake_A from real_B: fake_A = G_B2A(real_B)
        4. Reconstruct B from fake_A: rec_B = G_A2B(fake_A)
        5. L1(real_A, rec_A) + L1(real_B, rec_B)
        """
        # Forward cycle: A → B → A
        fake_B = G_A2B(real_A)
        rec_A = G_B2A(fake_B)

        # Backward cycle: B → A → B
        fake_A = G_B2A(real_B)
        rec_B = G_A2B(fake_A)

        loss_cycle = self.l1(rec_A, real_A) + self.l1(rec_B, real_B)
        return loss_cycle * self.lambda_cyc

    def identity_loss(self, real_A, real_B, G_A2B, G_B2A):
        """
        Identity loss: G_A2B(B) ≈ B and G_B2A(A) ≈ A

        TODO: Implement the identity loss
        Hint:
        1. Feed real_B into G_A2B → should stay the same
        2. Feed real_A into G_B2A → should stay the same
        3. L1(real_B, G_A2B(real_B)) + L1(real_A, G_B2A(real_A))
        """
        idt_B = G_A2B(real_B)  # B input to G_A2B should stay in B domain
        idt_A = G_B2A(real_A)  # A input to G_B2A should stay in A domain

        loss_idt = self.l1(idt_B, real_B) + self.l1(idt_A, real_A)
        return loss_idt * self.lambda_identity


# ─── Image Buffer ──────────────────────────────────────────────
class ImagePool:
    """
    Stores previously generated images for training stability.
    Randomly returns either the input image or one from the pool.
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        """Return images from the pool with 50% probability"""
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    idx = torch.randint(0, self.pool_size, (1,)).item()
                    old = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(old)
                else:
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


# ─── Quick Test ────────────────────────────────────────────────
def test_components():
    """Test that all components work correctly"""
    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W)

    # Test Generator
    G = CycleGANGenerator()
    y = G(x)
    print(f"Generator: {x.shape} → {y.shape}")
    assert y.shape == x.shape, f"Generator output shape mismatch: {y.shape}"

    # Test Discriminator
    D = PatchGANDiscriminator()
    d_out = D(x)
    print(f"Discriminator: {x.shape} → {d_out.shape}")

    # Test Cycle Consistency Loss
    loss_fn = CycleGANLoss()
    G_A2B = CycleGANGenerator()
    G_B2A = CycleGANGenerator()

    real_A = torch.randn(B, C, H, W)
    real_B = torch.randn(B, C, H, W)

    cyc_loss = loss_fn.cycle_consistency(real_A, real_B, G_A2B, G_B2A)
    print(f"Cycle consistency loss: {cyc_loss.item():.4f}")

    idt_loss = loss_fn.identity_loss(real_A, real_B, G_A2B, G_B2A)
    print(f"Identity loss: {idt_loss.item():.4f}")

    # Test Image Pool
    pool = ImagePool(pool_size=50)
    fake = torch.randn(B, C, H, W)
    pooled = pool.query(fake)
    print(f"Image pool: {fake.shape} → {pooled.shape}")

    print("\nAll components working correctly!")


# ─── Exercise Tasks ────────────────────────────────────────────
"""
TODO Exercises:
1. Implement the full CycleGAN training loop:
   - Alternate between training D_A, D_B and G_A2B, G_B2A
   - Use ImagePool for generated images
   - Apply learning rate decay after a fixed number of epochs

2. Experiment with different numbers of residual blocks (6, 9, 12):
   - How does depth affect training time?
   - How does depth affect output quality?

3. Replace LSGAN loss with vanilla GAN loss (BCE):
   - Observe training stability differences
   - Which converges faster? Which is more stable?

4. Implement the full horse↔zebra translation:
   - Download the CycleGAN dataset
   - Train for 200 epochs with lr=0.0002, decay after 100 epochs
   - Visualize A→B→A cycle results

5. Add a perceptual loss (VGG feature matching) alongside cycle consistency:
   - Does it improve visual quality?
   - What's the trade-off in training time?
"""

if __name__ == "__main__":
    test_components()
