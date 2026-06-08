export interface CurvePoint {
  epoch: number;
  loss: number;
}

const EPOCHS = 200;

/**
 * Plain CNN 训练损失曲线。
 * - 浅模型（depth ≤ 30）：单调指数衰减到 ~0.05
 * - 深模型（depth > 30）：先降后升的 "退化" U 形——越深越糟
 */
export function plainCurve(depth: number, epochs = EPOCHS): CurvePoint[] {
  const isDeep = depth > 30;
  const points: CurvePoint[] = [];

  if (!isDeep) {
    for (let i = 1; i <= epochs; i++) {
      const loss = 0.6 * Math.exp(-i / 35) + 0.05;
      points.push({ epoch: i, loss });
    }
    return points;
  }

  const minEpoch = 50 + (depth - 30) * 1.0;
  const minLoss = 0.1 + (depth - 30) * 0.0015;
  const finalLoss = minLoss + (depth - 30) * 0.0025;

  for (let i = 1; i <= epochs; i++) {
    const downLoss = (0.65 - minLoss) * Math.exp(-i / (minEpoch * 0.4)) + minLoss;
    const rise = Math.max(0, (i - minEpoch) / (epochs - minEpoch));
    const upLoss = minLoss + rise * (finalLoss - minLoss);
    points.push({ epoch: i, loss: Math.max(downLoss, upLoss) });
  }
  return points;
}

/**
 * ResNet 训练损失曲线。
 * - 任何深度都稳定指数衰减到 ~0.04
 * - 深度越深，时间常数稍大（收敛略慢），但最终损失差不多
 */
export function resnetCurve(depth: number, epochs = EPOCHS): CurvePoint[] {
  const finalLoss = 0.04;
  const tau = 30 + Math.sqrt(depth) * 3;
  const points: CurvePoint[] = [];
  for (let i = 1; i <= epochs; i++) {
    const loss = (0.65 - finalLoss) * Math.exp(-i / tau) + finalLoss;
    points.push({ epoch: i, loss });
  }
  return points;
}

/** BasicBlock 参数量（C=64 通道，2 个 3×3 conv） */
export const BASIC_BLOCK_PARAMS = 2 * (3 * 3 * 64 * 64); // 73,728

/** Bottleneck 参数量（1×1↓256→64 + 3×3 64 + 1×1↑64→256） */
export const BOTTLENECK_PARAMS =
  1 * 1 * 256 * 64 + 3 * 3 * 64 * 64 + 1 * 1 * 64 * 256; // 70,144

/** 梯度公路：每经过一个 plain block，梯度幅度衰减系数 */
export const GRADIENT_DECAY_PER_BLOCK = 0.85;
