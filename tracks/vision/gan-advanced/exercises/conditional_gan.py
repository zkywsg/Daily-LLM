"""Conditional GAN · tracks/vision/gan-advanced/exercises/conditional_gan.py · 条件 GAN 生成 MNIST 数字 · torch"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# ─── Hyperparameters ───────────────────────────────────────────
Z_DIM = 100
NUM_CLASSES = 10
BASE_CH = 64
BATCH_SIZE = 128
LR = 0.0002
BETAS = (0.5, 0.999)
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Generator ─────────────────────────────────────────────────
class ConditionalGenerator(nn.Module):
    """Conditional GAN Generator: noise z + label y → image"""

    def __init__(self, z_dim=Z_DIM, num_classes=NUM_CLASSES, base_ch=BASE_CH):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        input_dim = z_dim + num_classes
        self.net = nn.Sequential(
            # TODO: Build the generator architecture
            # Hint: Use ConvTranspose2d layers to upsample from (B, input_dim, 1, 1)
            #       to (B, 1, 28, 28) for MNIST
            # Structure: input_dim → base_ch*4 → base_ch*2 → base_ch → 1
            # Use BatchNorm2d + ReLU between layers, Tanh at the output
            nn.ConvTranspose2d(input_dim, base_ch * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch, 1, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # z: (B, z_dim), labels: (B,)
        y = self.label_embed(labels)  # (B, num_classes)
        x = torch.cat([z, y], dim=1)  # (B, z_dim + num_classes)
        x = x.view(x.size(0), -1, 1, 1)
        return self.net(x)


# ─── Discriminator ─────────────────────────────────────────────
class ConditionalDiscriminator(nn.Module):
    """Conditional GAN Discriminator: image x + label y → real/fake"""

    def __init__(self, num_classes=NUM_CLASSES, base_ch=BASE_CH):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        # TODO: Build the discriminator architecture
        # Hint: Input is (B, 1 + num_classes, 28, 28)
        #       Concatenate the label embedding (projected to image spatial dims)
        #       with the input image at the channel dimension
        self.label_proj = nn.Sequential(
            nn.Linear(num_classes, 28 * 28),
            nn.LeakyReLU(0.2, True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(1 + 1, base_ch, 4, 2, 1),  # 1(ch image) + 1(ch label)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch * 2, 1, 7, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        # x: (B, 1, 28, 28), labels: (B,)
        y = self.label_embed(labels)  # (B, num_classes)
        y = self.label_proj(y)  # (B, 28*28)
        y = y.view(y.size(0), 1, 28, 28)  # (B, 1, 28, 28)
        xy = torch.cat([x, y], dim=1)  # (B, 2, 28, 28)
        return self.net(xy).view(-1, 1).squeeze(1)


# ─── Training Loop ─────────────────────────────────────────────
def train():
    """Train conditional GAN on MNIST"""
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Models
    G = ConditionalGenerator().to(DEVICE)
    D = ConditionalDiscriminator().to(DEVICE)

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    # Loss
    criterion = nn.BCELoss()

    print(f"Training cGAN on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Labels for real and fake
            real_label = torch.ones(batch_size, device=DEVICE)
            fake_label = torch.zeros(batch_size, device=DEVICE)

            # ─── Train Discriminator ────────────────────────────
            opt_D.zero_grad()

            # Real images
            output_real = D(real_imgs, labels)
            loss_D_real = criterion(output_real, real_label)

            # Fake images
            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_imgs = G(z, labels)
            output_fake = D(fake_imgs.detach(), labels)
            loss_D_fake = criterion(output_fake, fake_label)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            opt_D.step()

            # ─── Train Generator ────────────────────────────────
            opt_G.zero_grad()

            output = D(fake_imgs, labels)
            loss_G = criterion(output, real_label)  # G wants D to say "real"
            loss_G.backward()
            opt_G.step()

            if i % 200 == 0:
                print(f"[{epoch}/{EPOCHS}] [{i}/{len(dataloader)}] "
                      f"D_loss: {loss_D.item():.4f} G_loss: {loss_G.item():.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(G.state_dict(), "checkpoints/cgan_generator.pth")
    print("Generator saved to checkpoints/cgan_generator.pth")


# ─── Exercise Tasks ────────────────────────────────────────────
"""
TODO Exercises:
1. Modify the Generator to produce 64×64 images (change architecture accordingly)
2. Add a label smoothing trick: use 0.9 instead of 1.0 for real labels
3. Implement class-conditional generation: generate specific digits on demand
4. Add FID evaluation metric to track generation quality during training
5. Compare training with and without condition information in the Discriminator
   (remove label from D and observe the difference in generation controllability)
"""

if __name__ == "__main__":
    train()
