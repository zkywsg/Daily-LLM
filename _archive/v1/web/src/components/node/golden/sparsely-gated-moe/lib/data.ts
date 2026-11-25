// Sparsely-Gated MoE (Shazeer 2017) demo 数据。
// N=2048 experts / top-K=4 routing,用确定性 hash 模拟 gating,
// 不真跑 MoE forward,只是让 viewer 看到 "per-token top-K 稀疏路由" 的行为。

export const NUM_EXPERTS = 2048;
export const TOP_K = 4;

/** 为了可视化,只展示一个 SAMPLE_SIZE 大小的 expert 子集(2048 个画不下) */
export const SAMPLE_SIZE = 64;

/** 给定 token 文本,产生一个 SAMPLE_SIZE 维 router logits 向量(确定性) */
export function routerLogits(token: string, seed = 0): number[] {
  const logits: number[] = [];
  for (let e = 0; e < SAMPLE_SIZE; e++) {
    let h = seed + e * 31;
    for (const c of token) h = (h * 131 + c.charCodeAt(0)) >>> 0;
    logits.push(((h % 1000) / 1000) * 4 - 2);
  }
  return logits;
}

/** softmax */
export function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** 返回 token 在 SAMPLE_SIZE 个 expert 里对应的 top-k expert 索引 + softmax 权重 */
export function topKExperts(token: string, k: number, seed = 0): Array<{ expert: number; weight: number }> {
  const logits = routerLogits(token, seed);
  const indexed = logits.map((l, i) => ({ i, l }));
  indexed.sort((a, b) => b.l - a.l);
  const top = indexed.slice(0, k);
  const topLogits = top.map((t) => t.l);
  const topWeights = softmax(topLogits);
  return top.map((t, idx) => ({ expert: t.i, weight: topWeights[idx] }));
}

export const DEMO_TOKENS: Array<{ label: string; tokens: string[] }> = [
  { label: "中文短句", tokens: ["深度", "学习", "扩", "参数", "锁死", "算力"] },
  { label: "English", tokens: ["gate", "picks", "top", "four", "experts", "sparsely"] },
  { label: "翻译任务", tokens: ["le", "chat", "noir", "mange", "le", "poisson"] },
];

/** 137B 总参数 vs 1.5B 激活参数(Shazeer 2017,MoE-2048-experts 配置) */
export const PARAM_COMPARE = {
  totalParams: 137e9,
  activeParams: 1.5e9,
  /** 对照组:LSTM-Big,参数与算力 1:1 绑死 */
  denseBaseline: {
    params: 1.4e9,
    activeParams: 1.4e9,
  },
};

/** 模拟 expert 负载(64 个采样 expert):无 aux loss 时极不均匀,加 aux loss 后均匀 */
export function expertLoad(balanced: boolean): number[] {
  const n = 16; // 展示用的柱子数(代表 2048 个 expert 里的一组采样)
  if (balanced) {
    return Array.from({ length: n }, (_, i) => 1 + (((i * 17) % 7) - 3) / 30);
  }
  // 不均匀:少数 expert 被反复选中,大多数几乎闲置
  return [4.1, 3.4, 0.1, 0.2, 3.8, 0.1, 0.3, 0.1, 2.9, 0.2, 0.1, 3.2, 0.1, 0.2, 0.1, 0.4];
}

/** LM1B benchmark 性能表(用于 ParamsVsComputeChart 的对比条形) */
export const LM1B_COMPARE = [
  { label: "LSTM-2048\nbaseline", params: 0.2e9, activeParams: 0.2e9, perplexity: 67.5, computeX: 1 },
  { label: "LSTM-Big", params: 1.4e9, activeParams: 1.4e9, perplexity: 39.8, computeX: 7 },
  { label: "MoE-32\nexperts", params: 0.8e9, activeParams: 0.18e9, perplexity: 35.7, computeX: 1 },
  { label: "MoE-512\nexperts", params: 4.4e9, activeParams: 0.42e9, perplexity: 31.3, computeX: 1.4 },
  { label: "MoE-2048\nexperts", params: 137e9, activeParams: 1.5e9, perplexity: 28.0, computeX: 2.4 },
];

/** Expert parallelism:2048 个 expert 分布在 128 块 GPU 上,每 GPU 16 个 */
export const EXPERT_PARALLELISM = {
  numGpus: 128,
  expertsPerGpu: 16,
  totalExperts: 2048,
  /** 可视化时只画一个缩略子集 */
  displayGpus: 8,
};
