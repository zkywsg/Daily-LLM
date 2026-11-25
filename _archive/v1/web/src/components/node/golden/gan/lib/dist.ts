// GAN 数学:2D 玩具分布 + 训练进度对应的 G/D 状态模拟。
// 不真训 GAN —— 用解析公式让 viewer 看到"分布随 iter 收敛 + loss 振荡"。

export interface Point {
  x: number;
  y: number;
}

/** mulberry32 伪随机 */
function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller 高斯采样 */
function gaussian(r: () => number, mean: number, std: number): number {
  const u1 = Math.max(1e-9, r());
  const u2 = r();
  const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return g * std + mean;
}

/** 多模真分布(N 个 GMM 中心) */
export interface RealMode {
  cx: number;
  cy: number;
  std: number;
}

/** 4 模真分布(让 mode collapse 演示有素材) */
export const REAL_MODES: RealMode[] = [
  { cx: -1.5, cy: -1.5, std: 0.3 },
  { cx: 1.5, cy: -1.5, std: 0.3 },
  { cx: -1.5, cy: 1.5, std: 0.3 },
  { cx: 1.5, cy: 1.5, std: 0.3 },
];

/** 从真分布采 n 个点 */
export function sampleReal(n: number, seed = 42): Point[] {
  const r = rng(seed);
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const m = REAL_MODES[Math.floor(r() * REAL_MODES.length)];
    pts.push({ x: gaussian(r, m.cx, m.std), y: gaussian(r, m.cy, m.std) });
  }
  return pts;
}

/**
 * 模拟 G 在 iter t / T 时的"输出分布":
 *   t=0 → 像标准高斯(噪声)
 *   t=T → 接近真分布(4 模 GMM)
 * 用线性插值:G_pts = (1-α) · 高斯 + α · GMM,α = t/T
 */
export function sampleGen(
  n: number,
  iter: number,
  totalIter: number,
  seed = 7,
  modeCollapse = false,
): Point[] {
  const alpha = Math.min(1, iter / Math.max(1, totalIter));
  const r = rng(seed);
  const modes = modeCollapse ? [REAL_MODES[0]] : REAL_MODES; // mode collapse 时 G 只学 1 个模
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const noise = { x: gaussian(r, 0, 1.8), y: gaussian(r, 0, 1.8) };
    const m = modes[Math.floor(r() * modes.length)];
    const real = { x: gaussian(r, m.cx, m.std), y: gaussian(r, m.cy, m.std) };
    pts.push({
      x: (1 - alpha) * noise.x + alpha * real.x,
      y: (1 - alpha) * noise.y + alpha * real.y,
    });
  }
  return pts;
}

/**
 * D 在 iter t 给某点 x 的"判别概率":
 *   t=0 → D 还分不清,所有点都给 ~0.5
 *   t=T → D 几乎完美,真分布附近给 ~1,远离的给 ~0
 * 用距最近真模的距离做 logistic。
 */
export function discriminate(p: Point, iter: number, totalIter: number): number {
  const alpha = Math.min(1, iter / Math.max(1, totalIter));
  // 找最近真模的距离
  let minD = Infinity;
  for (const m of REAL_MODES) {
    const d = Math.hypot(p.x - m.cx, p.y - m.cy);
    if (d < minD) minD = d;
  }
  // D 训得越熟,sharpness 越高;t=0 时 0.5,t=T 时 logistic(steep)
  const steepness = 0.3 + alpha * 4;
  const sigm = 1 / (1 + Math.exp(steepness * (minD - 0.6)));
  // 训练初期返回 ~0.5
  return 0.5 + alpha * (sigm - 0.5);
}

/** 模拟训练 loss 曲线:G/D loss 都在 ln 2 附近振荡,带衰减 */
export function lossCurves(totalIter: number, modeCollapse = false): {
  d: number[];
  g: number[];
} {
  const r = rng(11);
  const d: number[] = [];
  const g: number[] = [];
  for (let i = 0; i < totalIter; i++) {
    const t = i / Math.max(1, totalIter);
    // D 损失:理想点 ln 2 ≈ 0.693
    const dl = 0.693 + 0.3 * Math.exp(-t * 4) * (r() - 0.5);
    // G 损失:也在 ln 2 附近振荡;mode collapse 时 G loss 长期偏低
    const gl = (modeCollapse ? 0.5 : 0.693) + 0.4 * Math.exp(-t * 3) * (r() - 0.5);
    d.push(dl);
    g.push(gl);
  }
  return { d, g };
}

/** 经典 vs non-saturating G 损失曲线对比 */
export function lossComparisonCurve(): {
  pseudoD: number[];
  original: number[];
  nonSat: number[];
} {
  // x = D(G(z)) ∈ [0, 1]:D 给 fake 打分
  const N = 100;
  const pseudoD: number[] = [];
  const original: number[] = []; // log(1 - D)  → D 接近 0 时梯度消失
  const nonSat: number[] = []; // -log(D)        → D 接近 0 时梯度爆炸,适合训练初期
  for (let i = 0; i <= N; i++) {
    const x = (i / N) * 0.98 + 0.01;
    pseudoD.push(x);
    original.push(Math.log(1 - x));
    nonSat.push(-Math.log(x));
  }
  return { pseudoD, original, nonSat };
}
