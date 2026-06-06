"""目标检测基础组件 · tracks/vision/object-detection/exercises/anchor_boxes.py · IoU、锚框、匹配编码与 NMS · torch"""

import torch

torch.manual_seed(42)


# ══════════════════════════════════════════════
# Step 1  边界框表示与 IoU
# ══════════════════════════════════════════════

def compute_iou(boxes_a, boxes_b):
    """计算两组边界框之间的 IoU (Intersection over Union)。

    Args:
        boxes_a: (N, 4) 边界框，格式 [x1, y1, x2, y2]
        boxes_b: (M, 4) 边界框，格式 [x1, y1, x2, y2]
    Returns:
        iou: (N, M) IoU 矩阵
    """
    N = boxes_a.size(0)
    M = boxes_b.size(0)

    # 扩展维度以广播: (N,1,4) 和 (1,M,4) -> (N,M,4)
    a = boxes_a[:, None, :].expand(N, M, 4)
    b = boxes_b[None, :, :].expand(N, M, 4)

    # 交集
    inter_x1 = torch.max(a[..., 0], b[..., 0])
    inter_y1 = torch.max(a[..., 1], b[..., 1])
    inter_x2 = torch.min(a[..., 2], b[..., 2])
    inter_y2 = torch.min(a[..., 3], b[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 各自面积
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)

    union = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)


def step1_bbox_and_iou():
    """验证 IoU 计算。"""
    # 完全重合 → IoU = 1.0
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou_same = compute_iou(box, box)
    assert torch.allclose(iou_same, torch.ones(1, 1), atol=1e-5), f"IoU of identical boxes should be 1.0, got {iou_same}"

    # 完全不重叠 → IoU = 0.0
    box_a = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
    box_b = torch.tensor([[10.0, 10.0, 15.0, 15.0]])
    iou_disjoint = compute_iou(box_a, box_b)
    assert iou_disjoint.item() < 1e-5, f"Disjoint boxes should have IoU≈0, got {iou_disjoint}"

    # 部分重叠 → 0 < IoU < 1
    box_c = torch.tensor([[2.0, 2.0, 8.0, 8.0]])
    iou_partial = compute_iou(box, box_c)
    assert 0 < iou_partial.item() < 1, f"Partial overlap IoU should be in (0,1), got {iou_partial}"
    # A=[0,0,10,10] area=100, C=[2,2,8,8] area=36, inter=[2,2,8,8] area=36
    # union = 100+36-36 = 100, IoU = 36/100 = 0.36
    expected = 36.0 / 100.0
    assert abs(iou_partial.item() - expected) < 0.01, f"Expected {expected:.3f}, got {iou_partial.item():.3f}"

    print(f"[Step 1] IoU  same={iou_same.item():.3f}  disjoint={iou_disjoint.item():.4f}  partial={iou_partial.item():.3f}")


# ══════════════════════════════════════════════
# Step 2  锚框 (Anchor) 生成
# ══════════════════════════════════════════════

def generate_anchors(feature_h, feature_w, stride, scales, ratios):
    """在特征图每个位置生成多个锚框。

    Args:
        feature_h, feature_w: 特征图大小
        stride: 特征图相对原图的下采样倍率
        scales: 锚框尺度列表 (如 [0.5, 1.0, 2.0])
        ratios: 宽高比列表 (如 [0.5, 1.0, 2.0])
    Returns:
        anchors: (K, 4) 锚框 [x1, y1, x2, y2]，K = H*W*len(scales)*len(ratios)
    """
    anchors = []
    for y in range(feature_h):
        for x in range(feature_w):
            # 特征图像素中心映射回原图坐标
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for scale in scales:
                for ratio in ratios:
                    # scale 控制面积，ratio 控制宽高比
                    h = stride * scale * (ratio ** 0.5)
                    w = stride * scale * (ratio ** -0.5)
                    anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def step2_anchor_generation():
    """验证锚框生成数量和位置。"""
    H, W, STRIDE = 4, 4, 32
    SCALES = [1.0]
    RATIOS = [1.0]

    anchors = generate_anchors(H, W, STRIDE, SCALES, RATIOS)
    # 每个位置 1 个锚框，共 4*4=16
    assert anchors.shape == (16, 4), f"Expected (16,4), got {anchors.shape}"

    # 多尺度多比例
    anchors_multi = generate_anchors(H, W, STRIDE, [0.5, 1.0], [0.5, 1.0, 2.0])
    # 每个位置 2*3=6 个锚框，共 16*6=96
    assert anchors_multi.shape == (96, 4), f"Expected (96,4), got {anchors_multi.shape}"

    # 验证中心位置: (0,0) 的锚框中心应在 (16, 16)
    first = anchors[0]
    cx = (first[0] + first[2]) / 2
    cy = (first[1] + first[3]) / 2
    assert abs(cx - 16.0) < 1e-5 and abs(cy - 16.0) < 1e-5

    # 所有锚框面积 > 0
    areas = (anchors_multi[:, 2] - anchors_multi[:, 0]) * (anchors_multi[:, 3] - anchors_multi[:, 1])
    assert (areas > 0).all(), "All anchor areas should be positive"

    print(f"[Step 2] anchors  single={anchors.shape}  multi={anchors_multi.shape}  center_ok=True")


# ══════════════════════════════════════════════
# Step 3  锚框匹配与编码/解码
# ══════════════════════════════════════════════

def encode_boxes(gt_boxes, anchor_boxes):
    """将 GT 框编码为相对锚框的偏移量 ( Faster R-CNN 风格 )。

    编码: tx = (gx - ax) / aw,  ty = (gy - ay) / ah,  tw = log(gw/aw),  th = log(gh/ah)
    """
    # 中心 + 宽高
    ax = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2
    ay = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2
    aw = (anchor_boxes[:, 2] - anchor_boxes[:, 0]).clamp(min=1e-6)
    ah = (anchor_boxes[:, 3] - anchor_boxes[:, 1]).clamp(min=1e-6)

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)

    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)

    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(deltas, anchor_boxes):
    """将偏移量解码回绝对坐标。编码的逆操作。"""
    ax = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2
    ay = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2
    aw = (anchor_boxes[:, 2] - anchor_boxes[:, 0]).clamp(min=1e-6)
    ah = (anchor_boxes[:, 3] - anchor_boxes[:, 1]).clamp(min=1e-6)

    gx = deltas[:, 0] * aw + ax
    gy = deltas[:, 1] * ah + ay
    gw = torch.exp(deltas[:, 2]) * aw
    gh = torch.exp(deltas[:, 3]) * ah

    x1 = gx - gw / 2
    y1 = gy - gh / 2
    x2 = gx + gw / 2
    y2 = gy + gh / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def step3_encode_decode():
    """验证 encode-decode 往返一致性。"""
    anchors = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [20.0, 20.0, 80.0, 80.0],
    ])
    gt = torch.tensor([
        [15.0, 15.0, 45.0, 45.0],
        [30.0, 25.0, 70.0, 75.0],
    ])

    # encode → decode 应恢复原始 GT
    deltas = encode_boxes(gt, anchors)
    recovered = decode_boxes(deltas, anchors)

    assert torch.allclose(gt, recovered, atol=1e-4), (
        f"Encode-decode roundtrip failed:\n  GT={gt}\n  recovered={recovered}"
    )

    # delta 形状正确
    assert deltas.shape == (2, 4), f"Expected (2,4), got {deltas.shape}"

    print(f"[Step 3] encode-decode  roundtrip_error={(gt - recovered).abs().max().item():.6f}")


# ══════════════════════════════════════════════
# Step 4  非极大值抑制 (NMS)
# ══════════════════════════════════════════════

def nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制: 去除与更高分框重叠过大的低分框。

    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,) 置信度分数
        iou_threshold: IoU 阈值，超过此值的重叠框被抑制
    Returns:
        keep: 保留框的索引列表
    """
    # 按分数降序排列
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # 计算当前框与其余所有框的 IoU
        ious = compute_iou(boxes[i:i + 1], boxes[order[1:]])  # (1, M)
        ious = ious.squeeze(0)

        # 保留 IoU < 阈值的框
        mask = ious < iou_threshold
        order = order[1:][mask]

    return keep


def step4_nms():
    """验证 NMS 能正确抑制重叠框。"""
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],    # 高分框
        [1.0, 1.0, 11.0, 11.0],    # 与 #0 高度重叠 → 应被抑制
        [20.0, 20.0, 30.0, 30.0],  # 不重叠 → 应保留
        [0.5, 0.5, 10.5, 10.5],    # 与 #0 高度重叠 → 应被抑制
        [50.0, 50.0, 60.0, 60.0],  # 不重叠 → 应保留
    ])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])

    keep = nms(boxes, scores, iou_threshold=0.5)

    # 应保留框 0、2、4
    assert 0 in keep, "Highest scored box should be kept"
    assert 2 in keep, "Non-overlapping box should be kept"
    assert 4 in keep, "Non-overlapping box should be kept"
    # 重叠框 1、3 应被抑制（除非 IoU 恰好低于阈值）
    # 框 0 vs 框 1: 交集 = 9*9=81, 并集 ≈ 100+100-81=119, IoU ≈ 0.68 → 被抑制
    iou_01 = compute_iou(boxes[0:1], boxes[1:2]).item()
    if iou_01 > 0.5:
        assert 1 not in keep, f"Box 1 should be suppressed (IoU={iou_01:.3f})"

    print(f"[Step 4] NMS  kept={keep}  from {len(scores)} boxes  (threshold=0.5)")


# ══════════════════════════════════════════════
# 运行所有步骤
# ══════════════════════════════════════════════
if __name__ == "__main__":
    step1_bbox_and_iou()
    step2_anchor_generation()
    step3_encode_decode()
    step4_nms()
    print("\nAll steps passed!")
