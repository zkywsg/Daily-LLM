"""
exercise_training_loop.py — 渐进式训练循环练习
=================================================
从最简单的 forward-backward-step 开始，逐步加入正则化、调度、早停，
最终拼出生产级训练循环。

学习路线:
  Step 1  最小训练循环 — forward / loss / backward / step
  Step 2  加入 Dropout + BatchNorm — train/eval 模式切换
  Step 3  加入梯度裁剪 + 学习率调度 — Warmup + Cosine
  Step 4  生产级 epoch 循环 — 早停 + 检查点

依赖: torch (无需额外数据集，全部用合成数据)
运行: python exercise_training_loop.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.manual_seed(42)

# ──────────────────────────────────────────────
# 共用数据（合成分类任务：16 维 → 10 类）
# ──────────────────────────────────────────────
IN_DIM = 16
N_CLASS = 10
BATCH = 32

X_all = torch.randn(400, IN_DIM)
Y_all = torch.randint(0, N_CLASS, (400,))

train_loader = DataLoader(
    TensorDataset(X_all[:320], Y_all[:320]), batch_size=BATCH, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_all[320:], Y_all[320:]), batch_size=BATCH
)


# ══════════════════════════════════════════════
# Step 1  最小训练循环
# ══════════════════════════════════════════════
def step1_minimal_loop():
    """验证 zero_grad → backward → step 的正确顺序。"""
    model = nn.Sequential(nn.Linear(IN_DIM, 32), nn.ReLU(), nn.Linear(32, N_CLASS))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(BATCH, IN_DIM)
    y = torch.randint(0, N_CLASS, (BATCH,))

    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"
    print(f"[Step 1] minimal loop  loss = {loss.item():.4f}")


# ══════════════════════════════════════════════
# Step 2  加入 Dropout + BatchNorm
# ══════════════════════════════════════════════
class RegularizedNet(nn.Module):
    """带 BatchNorm + Dropout 的两层全连接网络。"""

    def __init__(self, in_dim: int = IN_DIM, hidden: int = 64, n_class: int = N_CLASS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def step2_regularized():
    """验证 train/eval 模式对 BN 和 Dropout 行为的影响。"""
    model = RegularizedNet()
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(BATCH, IN_DIM)
    y = torch.randint(0, N_CLASS, (BATCH,))

    # train 模式: BN 用 batch 统计，Dropout 生效
    model.train()
    out_train1 = model(x)
    out_train2 = model(x)
    # Dropout 使得两次前向结果不同
    assert not torch.allclose(out_train1, out_train2), "Dropout should cause stochastic outputs in train mode"

    # eval 模式: BN 用 running stats，Dropout 关闭
    model.eval()
    with torch.no_grad():
        out_eval1 = model(x)
        out_eval2 = model(x)
    assert torch.allclose(out_eval1, out_eval2), "eval mode should be deterministic"

    print(f"[Step 2] regularized net  train stochastic={not torch.allclose(out_train1, out_train2)}  eval deterministic={torch.allclose(out_eval1, out_eval2)}")


# ══════════════════════════════════════════════
# Step 3  梯度裁剪 + 学习率调度 (Warmup + Cosine)
# ══════════════════════════════════════════════
def step3_schedule_and_clip():
    """验证梯度裁剪和余弦学习率调度的行为。"""
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    WARMUP = 100
    TOTAL = 1000
    MAX_NORM = 1.0

    model = nn.Sequential(
        nn.Linear(IN_DIM, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, N_CLASS),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP)
    cosine = CosineAnnealingLR(optimizer, T_max=TOTAL - WARMUP, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP])

    x = torch.randn(BATCH, IN_DIM)
    y = torch.randint(0, N_CLASS, (BATCH,))

    # 记录 warmup 阶段前几步的 lr
    lrs = []
    for _ in range(5):
        model.train()
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # lr 在 warmup 阶段应该逐步上升
    assert lrs[-1] > lrs[0], f"Warmup lr should increase: {lrs[0]:.2e} -> {lrs[-1]:.2e}"
    # clip_grad_norm_ returns the total norm BEFORE clipping.
    # After clipping, the actual gradient norm is <= MAX_NORM.
    post_clip_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))
    assert post_clip_norm <= MAX_NORM + 1e-4, f"Post-clip norm should be <= {MAX_NORM}: {post_clip_norm:.4f}"
    print(f"[Step 3] schedule+clip  lr warmup: {lrs[0]:.2e} -> {lrs[-1]:.2e}  pre_clip={total_norm:.4f}  post_clip={post_clip_norm:.4f}")


# ══════════════════════════════════════════════
# Step 4  生产级训练循环 — 早停 + 检查点
# ══════════════════════════════════════════════
class ProductionNet(nn.Module):
    """Kaiming 初始化 + BN + Dropout 的生产级网络。"""

    def __init__(self, in_dim: int = IN_DIM, n_class: int = N_CLASS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_class),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def step4_production_training():
    """完整 epoch 循环 + early stopping + 最优模型保存。"""
    NUM_EPOCHS = 20
    PATIENCE = 5
    MAX_NORM = 1.0

    model = ProductionNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(loader, train: bool = True):
        model.train() if train else model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            with torch.set_grad_enabled(train):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += yb.size(0)
        return total_loss / n, correct / n

    best_val_acc = 0.0
    bad_epochs = 0
    stopped_epoch = NUM_EPOCHS

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            # 保存最优状态到内存（不写磁盘）
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                stopped_epoch = epoch + 1
                break

    # 恢复最优模型
    model.load_state_dict(best_state)
    final_loss, final_acc = run_epoch(val_loader, train=False)

    assert best_val_acc > 0, "Should achieve some accuracy"
    assert stopped_epoch <= NUM_EPOCHS, "Early stop or normal finish"
    print(f"[Step 4] production training  stopped@epoch{stopped_epoch}  best_val_acc={best_val_acc:.3f}  final_val_acc={final_acc:.3f}")


# ══════════════════════════════════════════════
# 运行所有步骤
# ══════════════════════════════════════════════
if __name__ == "__main__":
    step1_minimal_loop()
    step2_regularized()
    step3_schedule_and_clip()
    step4_production_training()
    print("\nAll steps passed!")
