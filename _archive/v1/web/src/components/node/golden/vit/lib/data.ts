// ViT demo 数据:patch 化的玩具 image + scaling 性能对比。

/** 一个 14×14 的"猫脸"玩具 image,值 0-1 表示亮度 */
export const TOY_IMAGE: number[][] = (() => {
  const N = 14;
  const grid: number[][] = Array.from({ length: N }, () => Array(N).fill(0.92));
  // 椭圆脸
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const dx = (c - 6.5) / 5;
      const dy = (r - 7) / 5.5;
      const inside = dx * dx + dy * dy < 1;
      if (inside) grid[r][c] = 0.72;
    }
  }
  // 双眼
  for (const [r, c] of [[5, 4], [5, 9]]) {
    grid[r][c] = 0.18;
    grid[r + 1][c] = 0.18;
  }
  // 鼻子
  grid[8][6] = 0.35; grid[8][7] = 0.35;
  // 嘴
  for (let c = 5; c <= 8; c++) grid[10][c] = 0.25;
  return grid;
})();

/** image 切成 P×P 的 patches。返回 patches 数组,每个是 N/P × N/P 的子矩阵 */
export function splitPatches(image: number[][], patchSize: number): number[][][] {
  const N = image.length;
  const numAxis = N / patchSize;
  const patches: number[][][] = [];
  for (let pr = 0; pr < numAxis; pr++) {
    for (let pc = 0; pc < numAxis; pc++) {
      const patch: number[][] = [];
      for (let r = 0; r < patchSize; r++) {
        const row: number[] = [];
        for (let c = 0; c < patchSize; c++) {
          row.push(image[pr * patchSize + r][pc * patchSize + c]);
        }
        patch.push(row);
      }
      patches.push(patch);
    }
  }
  return patches;
}

/** patch → flatten → "linear projection" 模拟(实际只截前 8 维 + 加噪声给 viewer 看一个 emb 向量样子) */
export function patchToEmb(patch: number[][], dim = 8): number[] {
  const flat = patch.flat();
  const emb: number[] = [];
  for (let i = 0; i < dim; i++) {
    let s = 0;
    for (let j = 0; j < flat.length; j++) {
      // 用 sin 加权当 "linear projection" 占位 — 让不同 patch 出来不一样
      s += flat[j] * Math.sin((i + 1) * (j + 1));
    }
    emb.push(s / flat.length);
  }
  return emb;
}

/** ViT vs ResNet 在不同数据集大小下的精度 */
export interface ScalingPoint {
  dataset: string;
  /** 训练样本数(log10 用) */
  numSamples: number;
  vit: number;
  resnet: number;
}

export const SCALING_DATA: ScalingPoint[] = [
  { dataset: "ImageNet-1k", numSamples: 1.3e6, vit: 77.9, resnet: 80.4 },
  { dataset: "ImageNet-21k", numSamples: 14e6, vit: 84.0, resnet: 83.5 },
  { dataset: "JFT-300M", numSamples: 300e6, vit: 88.0, resnet: 85.3 },
];

/** 一个 14×14 玩具 image 上 CLS 对每个 patch 的"假装注意力强度",用来画 attention map */
export function simulateClsAttention(patchSize: number): number[][] {
  const N = TOY_IMAGE.length;
  const numAxis = N / patchSize;
  const out: number[][] = Array.from({ length: numAxis }, () => Array(numAxis).fill(0));
  // 模仿 ViT 实际行为:CLS 更关注 "object" patches(亮度跟背景差异大的)
  // 中心区域 attention 高,边角低
  for (let r = 0; r < numAxis; r++) {
    for (let c = 0; c < numAxis; c++) {
      // 计算该 patch 平均亮度
      let mean = 0, cnt = 0;
      for (let i = 0; i < patchSize; i++) for (let j = 0; j < patchSize; j++) {
        mean += TOY_IMAGE[r * patchSize + i][c * patchSize + j];
        cnt++;
      }
      mean /= cnt;
      // 偏离背景(0.92) 越多 attention 越大
      const score = Math.abs(mean - 0.92);
      out[r][c] = score;
    }
  }
  // 归一到 [0, 1]
  const flat = out.flat();
  const max = Math.max(...flat);
  if (max > 0) for (let r = 0; r < numAxis; r++) for (let c = 0; c < numAxis; c++) out[r][c] /= max;
  return out;
}
