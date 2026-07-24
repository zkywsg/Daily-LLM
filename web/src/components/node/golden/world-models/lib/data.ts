// World Models demo 数据:8x8 toy "帧"(灰度网格)+ VAE 压缩/重建 +
// MDN(混合高斯)预测下一潜状态的确定性函数。不真跑训练。

export const GRID_SIZE = 8;

/** 一个固定的 toy 帧:8x8 灰度值(0-1),模拟一个简单场景 */
export const TOY_FRAME: number[] = Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, i) => {
  const x = i % GRID_SIZE;
  const y = Math.floor(i / GRID_SIZE);
  const cx = 3.5, cy = 3.5;
  const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
  return Math.max(0, 1 - d / 4);
});

const N = GRID_SIZE * GRID_SIZE; // 64

/** DCT-II 正交基函数第 d 个基在位置 i 的值(标准正交余弦基,d=0..63) */
function dctBasis(d: number, i: number): number {
  const c = d === 0 ? Math.sqrt(1 / N) : Math.sqrt(2 / N);
  return c * Math.cos((Math.PI * (2 * i + 1) * d) / (2 * N));
}

/** 用正交余弦基把 64 维帧投影成 latentDim 维系数(截断 DCT,前 latentDim 个低频分量) */
export function encodeVAE(frame: number[], latentDim: number): number[] {
  const z: number[] = [];
  for (let d = 0; d < latentDim; d++) {
    let coeff = 0;
    for (let i = 0; i < frame.length; i++) {
      coeff += frame[i] * dctBasis(d, i);
    }
    z.push(coeff);
  }
  return z;
}

/** 用截断到 latentDim 个系数的正交基重建帧 —— 由于基是正交归一的,
 * latentDim 越大保留的低频分量越多,重建结果单调收敛到原图(latentDim=64 时完全重建)。 */
export function decodeVAE(z: number[], latentDim: number): number[] {
  const recon: number[] = new Array(N).fill(0);
  for (let d = 0; d < latentDim; d++) {
    for (let i = 0; i < N; i++) {
      recon[i] += z[d] * dctBasis(d, i);
    }
  }
  return recon.map((v) => Math.max(0, Math.min(1, v)));
}

export interface MixtureComponent {
  mean: number;
  std: number;
  weight: number;
}

/** 给定当前潜状态 z 的第 0 维,用 K 个高斯分量模拟 MDN-RNN 预测的下一状态分布 */
export function predictMixture(z0: number, k: number, seed = 0): MixtureComponent[] {
  const comps: MixtureComponent[] = [];
  for (let i = 0; i < k; i++) {
    let h = (seed + i * 977 + Math.round(z0 * 1000) * 31) >>> 0;
    h = (h * 2654435761) >>> 0;
    const mean = z0 + (((h % 1000) / 1000) * 2 - 1) * 0.8;
    const std = 0.1 + ((h >>> 8) % 100) / 1000;
    comps.push({ mean, std, weight: 0 });
  }
  // softmax 权重,确定性
  const rawWeights = comps.map((_, i) => Math.exp(Math.sin((seed + i + 1) * 1.3)));
  const sum = rawWeights.reduce((a, b) => a + b, 0);
  comps.forEach((c, i) => (c.weight = rawWeights[i] / sum));
  return comps;
}

/** 从混合分布采样出下一个 z0(确定性:取加权期望而非随机采样,便于演示复现) */
export function sampleMixtureMean(comps: MixtureComponent[]): number {
  return comps.reduce((s, c) => s + c.mean * c.weight, 0);
}

/** 梦境 rollout:给定初始 z0,自回归调用 predictMixture + sampleMixtureMean 前进 N 步,
 * 全程不接触真实帧/环境 —— 这就是 World Models 论文"完全在梦境里训练 C"的核心机制。 */
export function dreamRollout(z0Start: number, steps: number): number[] {
  const traj = [z0Start];
  let z0 = z0Start;
  for (let t = 0; t < steps; t++) {
    const comps = predictMixture(z0, 5, t);
    z0 = sampleMixtureMean(comps);
    traj.push(z0);
  }
  return traj;
}
