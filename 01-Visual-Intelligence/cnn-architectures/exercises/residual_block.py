"""ResNet 从零搭建 · 01-Visual-Intelligence/cnn-architectures/exercises/residual_block.py · BasicBlock、Bottleneck 到迷你 ResNet · torch"""

import torch
import torch.nn as nn

torch.manual_seed(42)


# ══════════════════════════════════════════════
# Step 1  残差连接的直觉
# ══════════════════════════════════════════════
def step1_residual_intuition():
    """对比有/无残差连接时，恒等映射的学习难度。"""
    torch.manual_seed(42)

    # 目标: 学习恒等映射 f(x) = x
    x_target = torch.randn(32, 64)

    # --- 无残差: 网络需要直接学 x = W2*relu(W1*x) ---
    plain = nn.Sequential(
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 64),
    )
    opt_plain = torch.optim.SGD(plain.parameters(), lr=0.01)

    # --- 有残差: 网络只需学残差 F(x) = x - x = 0 ---
    class ResidualLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 64),
            )

        def forward(self, x):
            return self.block(x) + x

    res_net = ResidualLayer()
    opt_res = torch.optim.SGD(res_net.parameters(), lr=0.01)

    loss_fn = nn.MSELoss()

    for _ in range(100):
        # plain
        out_plain = plain(x_target)
        loss_plain = loss_fn(out_plain, x_target)
        opt_plain.zero_grad()
        loss_plain.backward()
        opt_plain.step()

        # residual
        out_res = res_net(x_target)
        loss_res = loss_fn(out_res, x_target)
        opt_res.zero_grad()
        loss_res.backward()
        opt_res.step()

    # 残差版本应该更容易逼近恒等映射
    assert loss_res.item() < loss_plain.item(), (
        f"Residual should learn identity faster: res={loss_res.item():.6f} vs plain={loss_plain.item():.6f}"
    )
    print(f"[Step 1] identity learning  plain_loss={loss_plain.item():.6f}  res_loss={loss_res.item():.6f}")


# ══════════════════════════════════════════════
# Step 2  BasicBlock — 两层残差块 + shortcut 对齐
# ══════════════════════════════════════════════
class BasicBlock(nn.Module):
    """ResNet BasicBlock: 两个 3x3 卷积 + shortcut。

    当 stride!=1 或 in_ch!=out_ch 时，shortcut 用 1x1 conv 对齐。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.shortcut(x))


def step2_basic_block():
    """验证 BasicBlock 在不同配置下的 shape。"""
    # 同通道、无下采样: shortcut = Identity
    block_a = BasicBlock(64, 64, stride=1)
    x_a = torch.randn(2, 64, 8, 8)
    assert block_a(x_a).shape == (2, 64, 8, 8)

    # 通道变化 + 下采样: shortcut = 1x1 Conv
    block_b = BasicBlock(64, 128, stride=2)
    x_b = torch.randn(2, 64, 8, 8)
    out_b = block_b(x_b)
    assert out_b.shape == (2, 128, 4, 4), f"Expected (2,128,4,4), got {out_b.shape}"

    # 验证梯度可以流过 shortcut
    x_test = torch.randn(1, 64, 8, 8, requires_grad=True)
    block_a.train()
    loss = block_a(x_test).sum()
    loss.backward()
    assert x_test.grad is not None, "Gradient should flow through shortcut"
    assert x_test.grad.abs().sum() > 0, "Shortcut gradient should be nonzero"

    print(f"[Step 2] basic block  same-ch={block_a(x_a).shape}  downsample={out_b.shape}  grad_ok=True")


# ══════════════════════════════════════════════
# Step 3  Bottleneck — 1x1-3x3-1x1 三层残差块
# ══════════════════════════════════════════════
class Bottleneck(nn.Module):
    """ResNet Bottleneck: 1x1 降维 → 3x3 提取 → 1x1 升维。

    参数量远小于直接用 3x3 大通道卷积。
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),          # 降维
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False),  # 空间提取
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),         # 升维
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.shortcut(x))


def step3_bottleneck():
    """对比 BasicBlock 和 Bottleneck 的参数效率。"""
    # BasicBlock: 64→64, 两个 3x3
    basic = BasicBlock(64, 64)
    basic_params = sum(p.numel() for p in basic.parameters())

    # Bottleneck: 64→16→64, 1x1-3x3-1x1 (模拟 ResNet-50 的 4x 扩展比这里简化)
    bn = Bottleneck(64, 16, 64)
    bn_params = sum(p.numel() for p in bn.parameters())

    # Bottleneck 用更少参数实现类似变换
    assert bn_params < basic_params, (
        f"Bottleneck should use fewer params: {bn_params} vs {basic_params}"
    )

    # shape 验证
    x = torch.randn(2, 64, 8, 8)
    out = bn(x)
    assert out.shape == (2, 64, 8, 8)

    # 带下采样
    bn_ds = Bottleneck(64, 32, 128, stride=2)
    out_ds = bn_ds(x)
    assert out_ds.shape == (2, 128, 4, 4)

    print(f"[Step 3] bottleneck  basic_params={basic_params:,}  bn_params={bn_params:,}  shape={out_ds.shape}")


# ══════════════════════════════════════════════
# Step 4  迷你 ResNet — 多 stage 堆叠
# ══════════════════════════════════════════════
class MiniResNet(nn.Module):
    """迷你 ResNet: 3 个 stage，每 stage 2 个 BasicBlock。"""

    def __init__(self, n_class: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(32, 32, blocks=2, stride=1)
        self.stage2 = self._make_stage(32, 64, blocks=2, stride=2)
        self.stage3 = self._make_stage(64, 128, blocks=2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, n_class),
        )

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def step4_mini_resnet():
    """验证迷你 ResNet 可训练且 loss 下降。"""
    model = MiniResNet(n_class=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))

    losses = []
    model.train()
    for _ in range(30):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Step 4] mini ResNet  params={n_params:,}  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


# ══════════════════════════════════════════════
# 运行所有步骤
# ══════════════════════════════════════════════
if __name__ == "__main__":
    step1_residual_intuition()
    step2_basic_block()
    step3_bottleneck()
    step4_mini_resnet()
    print("\nAll steps passed!")
