"""YOLO 损失函数 · 01-Visual-Intelligence/object-detection/exercises/yolo_loss.py · YOLO v1 坐标、置信度、分类损失实现 · torch>=2.0"""

import torch
import torch.nn as nn
import math


# ── Step 1 · 坐标损失 ──────────────────────────────────────────
def coord_loss(
    pred_xy: torch.Tensor,
    pred_wh: torch.Tensor,
    target_xy: torch.Tensor,
    target_wh: torch.Tensor,
    obj_mask: torch.Tensor,
) -> torch.Tensor:
    """坐标损失：只计算有物体的网格位置，宽高用平方根。

    Args:
        pred_xy:   (B, S, S, B, 2)  预测的中心坐标
        pred_wh:   (B, S, S, B, 2)  预测的宽高
        target_xy: (B, S, S, B, 2)  目标中心坐标
        target_wh: (B, S, S, B, 2)  目标宽高
        obj_mask:  (B, S, S, B)     有物体为 True
    Returns:
        标量损失
    """
    mask = obj_mask.unsqueeze(-1).expand_as(pred_xy).float()  # (B,S,S,B,2)
    xy_loss = ((pred_xy - target_xy) ** 2) * mask
    wh_loss = ((torch.sqrt(pred_wh.clamp(min=1e-6))
                - torch.sqrt(target_wh.clamp(min=1e-6))) ** 2) * mask
    return (xy_loss.sum() + wh_loss.sum()) / obj_mask.float().sum().clamp(min=1)


# ── Step 2 · 置信度损失 ────────────────────────────────────────
def conf_loss(
    pred_conf: torch.Tensor,
    target_conf: torch.Tensor,
    obj_mask: torch.Tensor,
    lambda_noobj: float = 0.5,
) -> torch.Tensor:
    """置信度损失：有物体位置权重 1，无物体位置权重 λ_noobj。

    Args:
        pred_conf:   (B, S, S, B)  预测的 IoU 置信度
        target_conf: (B, S, S, B)  目标置信度
        obj_mask:    (B, S, S, B)  有物体为 True
        lambda_noobj: 无物体位置的权重
    Returns:
        标量损失
    """
    bce = nn.functional.binary_cross_entropy(pred_conf, target_conf, reduction="none")
    obj_weight = obj_mask.float()
    noobj_weight = (~obj_mask).float() * lambda_noobj
    weight = obj_weight + noobj_weight
    return (bce * weight).sum() / pred_conf.shape[0]


# ── Step 3 · 分类损失 ──────────────────────────────────────────
def class_loss(
    pred_class: torch.Tensor,
    target_class: torch.Tensor,
    obj_mask: torch.Tensor,
) -> torch.Tensor:
    """分类损失：只在有物体的网格位置计算 CrossEntropy。

    Args:
        pred_class:   (B, S, S, C)  类别 logits
        target_class: (B, S, S)     类别索引 (int)
        obj_mask:     (B, S, S)     有物体为 True（取第一个 box 的 mask）
    Returns:
        标量损失
    """
    B, S, _, C = pred_class.shape
    # 只取有物体的位置
    mask = obj_mask[:, :, :, 0] if obj_mask.dim() == 4 else obj_mask  # (B,S,S)
    logits = pred_class[mask]           # (N_obj, C)
    targets = target_class[mask]        # (N_obj,)
    if logits.numel() == 0:
        return torch.tensor(0.0, device=pred_class.device)
    return nn.functional.cross_entropy(logits, targets)


# ── Step 4 · 完整 YOLO Loss ────────────────────────────────────
class YOLOLoss(nn.Module):
    """YOLO v1 损失函数。

    输入张量形状: (B, S, S, B*5+C)
      - B: batch size
      - S: 网格大小 (7)
      - B: 每个网格预测框数 (2)
      - C: 类别数 (20)
    """

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向计算。

        Args:
            preds:   (B, S, S, B*5+C) 预测
            targets: (B, S, S, B*5+C) 目标
        Returns:
            标量损失
        """
        B_batch = preds.shape[0]
        box_len = self.B * 5

        # 解析预测
        pred_xy = preds[..., 0:box_len:5].reshape(B_batch, self.S, self.S, self.B, 1)
        # 扩展到2维以匹配
        pred_xy = pred_xy.expand(-1, -1, -1, -1, 2)
        pred_wh = preds[..., 3:box_len:5].reshape(B_batch, self.S, self.S, self.B, 1)
        pred_wh = pred_wh.expand(-1, -1, -1, -1, 2)
        pred_conf = preds[..., 4:box_len:5]                        # (B,S,S,B)
        pred_class = preds[..., box_len:]                          # (B,S,S,C)

        # 解析目标
        tgt_xy = targets[..., 0:box_len:5].reshape(B_batch, self.S, self.S, self.B, 1)
        tgt_xy = tgt_xy.expand(-1, -1, -1, -1, 2)
        tgt_wh = targets[..., 3:box_len:5].reshape(B_batch, self.S, self.S, self.B, 1)
        tgt_wh = tgt_wh.expand(-1, -1, -1, -1, 2)
        tgt_conf = targets[..., 4:box_len:5]
        tgt_class = targets[..., box_len:].argmax(dim=-1)          # (B,S,S)

        obj_mask = tgt_conf > 0.5                                  # (B,S,S,B)

        # 各部分损失
        c_loss = self.lambda_coord * coord_loss(pred_xy, pred_wh, tgt_xy, tgt_wh, obj_mask)
        cf_loss = conf_loss(pred_conf.sigmoid(), tgt_conf, obj_mask, self.lambda_noobj)
        cl_loss = class_loss(pred_class, tgt_class, obj_mask)

        return c_loss + cf_loss + cl_loss


# ── 测试 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    S, B, C = 7, 2, 20
    batch = 2

    # Step 1 测试: 坐标损失
    pred_xy = torch.rand(batch, S, S, B, 2)
    pred_wh = torch.rand(batch, S, S, B, 2) * 0.5 + 0.1
    tgt_xy = pred_xy.clone()
    tgt_wh = pred_wh.clone()
    obj_mask = torch.zeros(batch, S, S, B, dtype=torch.bool)
    obj_mask[0, 3, 3, 0] = True
    obj_mask[1, 4, 2, 1] = True
    loss_c = coord_loss(pred_xy, pred_wh, tgt_xy, tgt_wh, obj_mask)
    assert math.isfinite(loss_c.item()), f"coord_loss not finite: {loss_c}"
    # 完美预测应该接近 0
    loss_perfect = coord_loss(tgt_xy, tgt_wh, tgt_xy, tgt_wh, obj_mask)
    assert loss_perfect.item() < 1e-5, f"perfect pred loss should be ~0: {loss_perfect}"
    print(f"Step 1 ✓ coord_loss={loss_c.item():.4f}, perfect={loss_perfect.item():.6f}")

    # Step 2 测试: 置信度损失
    pred_c = torch.rand(batch, S, S, B).sigmoid()
    tgt_c = torch.zeros(batch, S, S, B)
    tgt_c[obj_mask] = 1.0
    loss_cf = conf_loss(pred_c, tgt_c, obj_mask, lambda_noobj=0.5)
    assert math.isfinite(loss_cf.item()), f"conf_loss not finite: {loss_cf}"
    assert loss_cf.item() > 0, "conf_loss should be positive for random preds"
    print(f"Step 2 ✓ conf_loss={loss_cf.item():.4f}")

    # Step 3 测试: 分类损失
    pred_cl = torch.randn(batch, S, S, C)
    tgt_cl = torch.randint(0, C, (batch, S, S))
    loss_cl = class_loss(pred_cl, tgt_cl, obj_mask)
    assert math.isfinite(loss_cl.item()), f"class_loss not finite: {loss_cl}"
    print(f"Step 3 ✓ class_loss={loss_cl.item():.4f}")

    # Step 4 测试: 完整 YOLO Loss
    criterion = YOLOLoss(S=S, B=B, C=C)
    preds = torch.randn(batch, S, S, B * 5 + C)
    targets = torch.zeros(batch, S, S, B * 5 + C)
    # 设定一个目标物体
    targets[0, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.4, 1.0])
    targets[0, 3, 3, B * 5 + 5] = 1.0  # 类别 5
    targets[1, 4, 2, 5:10] = torch.tensor([0.3, 0.7, 0.2, 0.3, 1.0])
    targets[1, 4, 2, B * 5 + 12] = 1.0  # 类别 12

    loss = criterion(preds, targets)
    assert math.isfinite(loss.item()), f"YOLO loss not finite: {loss}"
    assert loss.item() > 0, "YOLO loss should be positive"
    print(f"Step 4 ✓ YOLO_loss={loss.item():.4f}")
    print("\n所有测试通过 ✓")
