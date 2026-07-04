// Attention pattern 类型
export type AttnKind = "local" | "global" | "random" | "none";

export function buildAttentionMatrix(
  N: number,
  windowSize: number,
  globalIdx: number[],
  randomPerRow: number
): AttnKind[][] {
  const w = Math.floor(windowSize / 2);
  const matrix: AttnKind[][] = Array.from({ length: N }, () => Array(N).fill("none"));

  for (let i = 0; i < N; i++) {
    const lo = Math.max(0, i - w);
    const hi = Math.min(N - 1, i + w);
    for (let j = lo; j <= hi; j++) {
      matrix[i][j] = "local";
    }
  }

  for (const g of globalIdx) {
    for (let j = 0; j < N; j++) {
      matrix[g][j] = "global";
      matrix[j][g] = "global";
    }
  }

  // deterministic pseudo-random edges
  let seed = 42;
  function rng() {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }
  for (let i = 0; i < N; i++) {
    for (let r = 0; r < randomPerRow; r++) {
      const j = Math.floor(rng() * N);
      if (matrix[i][j] === "none") matrix[i][j] = "random";
    }
  }

  return matrix;
}

// 复杂度对比表
export interface ComplexityRow {
  model: string;
  complexity: string;
  mem4k: number; // GB
  mem8k: number; // GB
}
export const COMPLEXITY_TABLE: ComplexityRow[] = [
  { model: "原版 Transformer(dense)",       complexity: "O(N²)",     mem4k: 16,  mem8k: 64 },
  { model: "Sparse Transformer(strided)",   complexity: "O(N√N)",   mem4k: 2,   mem8k: 5.6 },
  { model: "Reformer(LSH)",                  complexity: "O(N log N)", mem4k: 1,  mem8k: 2 },
  { model: "Longformer / BigBird",           complexity: "O(N)",     mem4k: 1,   mem8k: 2 },
];

// 长文档 benchmark
export interface BenchRow {
  task: string;
  short: number;  // RoBERTa-512
  long: number;   // Longformer-4096
}
export const LONG_DOC_BENCH: BenchRow[] = [
  { task: "SQuAD",     short: 79.8, long: 88.0 },
  { task: "HotpotQA",  short: 68.1, long: 75.6 },
  { task: "IMDb",      short: 92.6, long: 95.7 },
];
