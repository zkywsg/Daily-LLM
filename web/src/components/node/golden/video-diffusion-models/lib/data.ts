// Video Diffusion Models demo 数据:2D+1D 分解卷积 vs 3D 卷积的
// FLOPs 量级对比 + 图像/视频联合训练混合比例演示 + 自回归滑窗扩展。

/** 给定分辨率 H×W、帧数 T、卷积核大小 k、通道数 C,估算 3D 卷积的 FLOPs(简化量级公式) */
export function flops3D(h: number, w: number, t: number, k: number, c: number): number {
  return h * w * t * k * k * k * c * c;
}

/** 2D 空间卷积(每帧独立)+ 1D 时间卷积的 FLOPs 之和 */
export function flopsFactorized(h: number, w: number, t: number, k: number, c: number): number {
  const spatial = h * w * t * k * k * c * c; // 2D 卷积对每一帧做
  const temporal = h * w * t * k * c * c; // 1D 卷积沿时间轴
  return spatial + temporal;
}

/** 给定图像:视频混合比例(0=纯视频,1=纯图像),模拟训练 loss 曲线的抖动幅度——
 * 纯视频数据量小,loss 抖动大;混入图像数据后由于数据量大大增加,曲线更平滑。
 * 返回 20 个点的 loss 值(确定性,不是真实训练,仅用于示意趋势)。 */
export function simulateLossCurve(imageRatio: number): number[] {
  const points: number[] = [];
  const noiseScale = 0.3 * (1 - imageRatio) + 0.02;
  for (let i = 0; i < 20; i++) {
    const base = 1.0 * Math.exp(-i / 8) + 0.1;
    let h = (i * 977 + Math.round(imageRatio * 1000) * 31) >>> 0;
    h = (h * 2654435761) >>> 0;
    const noise = (((h % 1000) / 1000) - 0.5) * 2 * noiseScale;
    points.push(Math.max(0.05, base + noise));
  }
  return points;
}

/** 自回归滑窗扩展:给定总窗口大小 windowSize,已生成帧数 generatedCount,
 * 返回当前窗口覆盖的帧区间 [start, end) —— 模拟"用后半窗口的已生成帧作为条件,
 * 继续生成下一窗口"这一自回归扩展长度的过程。 */
export function slidingWindow(windowSize: number, generatedCount: number): { start: number; end: number } {
  return { start: Math.max(0, generatedCount - windowSize), end: generatedCount };
}
