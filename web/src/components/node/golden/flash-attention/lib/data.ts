// GPU 内存层级(A100)
export interface MemoryTier {
  tier: string;
  capacity: string;
  bandwidthTBs: number;
  latencyNs: number;
}
export const MEMORY_HIERARCHY: MemoryTier[] = [
  { tier: "SRAM", capacity: "20 MB", bandwidthTBs: 19, latencyNs: 10 },
  { tier: "HBM", capacity: "40-80 GB", bandwidthTBs: 1.75, latencyNs: 400 },
];

// 速度/显存 benchmark(论文 A100 fp16)
export interface BenchRow {
  seqLen: number;
  naiveMs: number | null; // null = OOM
  flashMs: number;
}
export const BENCHMARK_TABLE: BenchRow[] = [
  { seqLen: 512, naiveMs: 1.4, flashMs: 0.4 },
  { seqLen: 1024, naiveMs: 5.7, flashMs: 1.0 },
  { seqLen: 2048, naiveMs: 22.8, flashMs: 2.4 },
  { seqLen: 4096, naiveMs: null, flashMs: 7.1 },
  { seqLen: 8192, naiveMs: null, flashMs: 26.8 },
];

// Tiling 网格:Q 切成 Tr 行块,K/V 切成 Tc 列块
export function buildTileGrid(tr: number, tc: number) {
  const grid: { i: number; j: number }[] = [];
  for (let i = 0; i < tr; i++) {
    for (let j = 0; j < tc; j++) {
      grid.push({ i, j });
    }
  }
  return grid;
}

// Online softmax 模拟:给定分块的 raw scores,逐块算出 running max / sum 轨迹
export interface SoftmaxStep {
  blockIdx: number;
  scores: number[];
  mNew: number;
  lNew: number;
}
export function simulateOnlineSoftmax(blocks: number[][]): SoftmaxStep[] {
  let m = -Infinity;
  let l = 0;
  const steps: SoftmaxStep[] = [];
  blocks.forEach((scores, blockIdx) => {
    const blockMax = Math.max(...scores);
    const mNew = Math.max(m, blockMax);
    const rescaleOld = Math.exp(m - mNew);
    const sumNew = scores.reduce((acc, s) => acc + Math.exp(s - mNew), 0);
    const lNew = (m === -Infinity ? 0 : rescaleOld * l) + sumNew;
    m = mNew;
    l = lNew;
    steps.push({ blockIdx, scores, mNew, lNew });
  });
  return steps;
}

// KV cache 对比:MHA / GQA / MQA
export interface KvCacheRow {
  name: string;
  groups: number; // 共享 K/V 的组数
  numHeads: number;
  relativeCacheSize: number; // 相对 MHA 的比例
}
export const KV_CACHE_COMPARE: KvCacheRow[] = [
  { name: "MHA", groups: 64, numHeads: 64, relativeCacheSize: 1.0 },
  { name: "GQA (g=8)", groups: 8, numHeads: 64, relativeCacheSize: 8 / 64 },
  { name: "MQA", groups: 1, numHeads: 64, relativeCacheSize: 1 / 64 },
];
