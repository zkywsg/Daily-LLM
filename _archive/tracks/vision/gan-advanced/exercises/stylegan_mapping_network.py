"""StyleGAN 核心组件 · tracks/vision/gan-advanced/exercises/stylegan_mapping_network.py · Mapping Network、AdaIN、Style Block · torch"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Mapping Network ───────────────────────────────────────────
class MappingNetwork(nn.Module):
    """
    Maps latent z to intermediate latent w through 8 FC layers.
    The W space is smoother and more disentangled than Z space.
    """

    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        # TODO: Build the mapping network
        # Hint: Stack num_layers of (Linear + LeakyReLU(0.2))
        #       Each layer maps z_dim → w_dim (after first layer, z_dim = w_dim)
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2, inplace=True)]
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # z: (B, z_dim) → w: (B, w_dim)
        return self.net(z)


# ─── AdaIN (Adaptive Instance Normalization) ───────────────────
class AdaIN(nn.Module):
    """
    Injects style by normalizing features then scaling/biasing with w.

    AdaIN(x, y) = y_s * (x - mean(x)) / std(x) + y_b
    where y_s and y_b are learned from w.
    """

    def __init__(self, w_dim, num_features):
        super().__init__()
        # TODO: Implement AdaIN
        # Hint:
        # 1. InstanceNorm2d (without learnable affine params)
        # 2. Two Linear layers: w → scale, w → bias
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale = nn.Linear(w_dim, num_features)
        self.style_bias = nn.Linear(w_dim, num_features)

    def forward(self, x, w):
        # x: (B, C, H, W), w: (B, w_dim)
        normalized = self.norm(x)
        scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        return scale * normalized + bias


# ─── Style Block ───────────────────────────────────────────────
class StyleBlock(nn.Module):
    """
    StyleGAN synthesis network block: Conv → Noise → AdaIN → LeakyReLU
    """

    def __init__(self, in_ch, out_ch, w_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adain = AdaIN(w_dim, out_ch)
        # Learnable noise scaling (initialized to 0)
        self.noise_weight = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        x = self.conv(x)
        # Add per-pixel Gaussian noise (broadcast over batch and channels)
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        x = x + self.noise_weight * noise
        x = self.adain(x, w)
        return self.activation(x)


# ─── Synthesis Network (Simplified) ────────────────────────────
class SynthesisNetwork(nn.Module):
    """
    Simplified StyleGAN synthesis network.
    Starts from a learned constant, progressively upsamples with style injection.
    """

    def __init__(self, w_dim=512, base_ch=512):
        super().__init__()
        # Learned constant input (4×4)
        self.const = nn.Parameter(torch.randn(1, base_ch, 4, 4))

        # Progressive synthesis blocks
        # TODO: Implement the synthesis blocks
        # Hint: Each block upsamples 2× and applies a StyleBlock
        #       Resolution: 4→8→16→32→64→128→256
        self.block_4 = StyleBlock(base_ch, base_ch, w_dim)      # 4×4
        self.block_8 = StyleBlock(base_ch, base_ch, w_dim)      # 8×8
        self.block_16 = StyleBlock(base_ch, base_ch // 2, w_dim)  # 16×16
        self.block_32 = StyleBlock(base_ch // 2, base_ch // 4, w_dim)  # 32×32
        self.block_64 = StyleBlock(base_ch // 4, base_ch // 8, w_dim)  # 64×64
        self.block_128 = StyleBlock(base_ch // 8, base_ch // 16, w_dim)  # 128×128
        self.block_256 = StyleBlock(base_ch // 16, base_ch // 32, w_dim)  # 256×256

        # To RGB
        self.to_rgb = nn.Conv2d(base_ch // 32, 3, 1)

        self.blocks = [
            self.block_4, self.block_8, self.block_16,
            self.block_32, self.block_64, self.block_128, self.block_256,
        ]

    def forward(self, w, mix_layer=None, w2=None):
        """
        Args:
            w: (B, w_dim) — primary style vector
            mix_layer: int or None — layer at which to switch to w2 (style mixing)
            w2: (B, w_dim) — secondary style vector for mixing
        """
        batch_size = w.size(0)
        x = self.const.expand(batch_size, -1, -1, -1)

        for i, block in enumerate(self.blocks):
            # Style mixing: use w2 after mix_layer
            current_w = w2 if (mix_layer is not None and i >= mix_layer) else w
            x = block(x, current_w)
            # Upsample (except last block)
            if i < len(self.blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode="bilinear",
                                  align_corners=False)

        return torch.tanh(self.to_rgb(x))


# ─── Style Mixing Demo ─────────────────────────────────────────
def demo_style_mixing():
    """Demonstrate style mixing: coarse layers from w1, fine layers from w2"""
    w_dim = 512
    G = SynthesisNetwork(w_dim=w_dim)
    G.eval()

    # Two random latent codes
    z1 = torch.randn(1, w_dim)
    z2 = torch.randn(1, w_dim)

    # For this demo, bypass MappingNetwork and use z directly as w
    # (In full implementation, pass through MappingNetwork first)

    print("Style Mixing Demo")
    print("=" * 50)

    # Generate with w1 only (all layers)
    with torch.no_grad():
        img_full_w1 = G(z1)
        print(f"Full w1: {img_full_w1.shape}")

    # Generate with w2 only
    with torch.no_grad():
        img_full_w2 = G(z2)
        print(f"Full w2: {img_full_w2.shape}")

    # Style mixing: coarse from w1, fine from w2 (mix at layer 4)
    with torch.no_grad():
        img_mixed = G(z1, mix_layer=4, w2=z2)
        print(f"Mixed (coarse=w1, fine=w2): {img_mixed.shape}")

    print("\nStyle mixing at different layers:")
    for mix_at in [2, 3, 4, 5]:
        with torch.no_grad():
            img = G(z1, mix_layer=mix_at, w2=z2)
            print(f"  Mix at layer {mix_at}: {img.shape}")

    print("\nInterpretation:")
    print("  Layers 0-1 (4×4 - 8×8):   Coarse — pose, face shape")
    print("  Layers 2-3 (16×16 - 32×32): Middle — facial features, hair")
    print("  Layers 4-6 (64×64 - 256×256): Fine — skin, texture, color")


# ─── Latent Space Interpolation ────────────────────────────────
def demo_interpolation():
    """Demonstrate latent space interpolation in W space"""
    z_dim = 512
    mapping = MappingNetwork(z_dim=z_dim)
    mapping.eval()

    z1 = torch.randn(1, z_dim)
    z2 = torch.randn(1, z_dim)

    print("\nLatent Space Interpolation")
    print("=" * 50)

    with torch.no_grad():
        w1 = mapping(z1)
        w2 = mapping(z2)

        print(f"w1 stats: mean={w1.mean():.4f}, std={w1.std():.4f}")
        print(f"w2 stats: mean={w2.mean():.4f}, std={w2.std():.4f}")

        # Linear interpolation in W space
        print("\nInterpolation steps (α from 0.0 to 1.0):")
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            w_interp = (1 - alpha) * w1 + alpha * w2
            print(f"  α={alpha:.2f}: mean={w_interp.mean():.4f}, "
                  f"std={w_interp.std():.4f}")


# ─── Exercise Tasks ────────────────────────────────────────────
"""
TODO Exercises:
1. Implement the full Mapping Network with different depths (4, 8, 16 layers):
   - Visualize the W space distribution vs Z space
   - Does deeper mapping = better disentanglement?

2. Compare AdaIN vs Weight Modulation/Demodulation (StyleGAN v2):
   - Implement the weight demodulation technique
   - Generate images and compare artifact levels

3. Implement path length regularization:
   - Measure how much the image changes for small perturbations in w
   - Apply the regularization loss during training

4. Build the full StyleGAN training loop:
   - Use progressive growing (start at 4×4, increase to 256×256)
   - Implement R1 regularization for the discriminator
   - Track FID during training

5. Experiment with noise injection:
   - Train with and without noise
   - Observe the effect on细节 variation (hair, pores, freckles)
   - Can you control the level of detail by scaling noise_weight?

6. Implement W+ space exploration:
   - Use different w at every layer (not just mixing at one point)
   - Find directions in W+ that control specific attributes
     (age, smile, glasses, pose)
"""

if __name__ == "__main__":
    demo_style_mixing()
    demo_interpolation()
