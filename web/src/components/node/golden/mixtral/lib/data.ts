// Mixtral demo 数据:8 个 expert + 一句话 token 路由 + 负载分布 + 参数账。
// 不真跑 MoE forward,用确定性 hash 让 viewer 看到\"每个 token 选 top-2\"的行为。

export const NUM_EXPERTS = 8;

/** 给定 token 文本,产生一个 8 维 router logits 向量(确定性) */
export function routerLogits(token: string, seed = 0): number[] {
  const logits: number[] = [];
  for (let e = 0; e < NUM_EXPERTS; e++) {
    let h = seed + e * 31;
    for (const c of token) h = (h * 131 + c.charCodeAt(0)) >>> 0;
    // 映射到 [-2, 2]
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

/** 返回 token 对应的 top-k expert 索引 + softmax 权重 */
export function topKExperts(token: string, k: number, seed = 0): Array<{ expert: number; weight: number }> {
  const logits = routerLogits(token, seed);
  const indexed = logits.map((l, i) => ({ i, l }));
  indexed.sort((a, b) => b.l - a.l);
  const top = indexed.slice(0, k);
  // 只在 top-k 上做 softmax(Mixtral 的实际行为)
  const topLogits = top.map((t) => t.l);
  const topWeights = softmax(topLogits);
  return top.map((t, idx) => ({ expert: t.i, weight: topWeights[idx] }));
}

export const DEMO_SENTENCES: Array<{ label: string; tokens: string[] }> = [
  { label: "中文短句", tokens: ["大", "模型", "稀疏", "激活", "省", "算力"] },
  { label: "English", tokens: ["The", "router", "picks", "top", "2", "experts"] },
  { label: "代码 token", tokens: ["def", "forward", "(", "x", ")", ":"] },
];

/** 8 个 expert 的总参数 / 每 token 激活参数对比(Mixtral 8x7B) */
export const PARAM_COMPARE = {
  totalParams: 46.7e9, // 8x experts × ~5.6B FFN + 共享部分
  activeParams: 12.9e9, // 2 experts (top-2)
  /** 跟 dense 模型对比的容量(类比 Llama-2-70B 之类) */
  denseEquivalent: 70e9,
};

/** 模拟 expert 负载:无 aux loss 时极不均匀,加 aux loss 后均匀 */
export function expertLoad(balanced: boolean): number[] {
  if (balanced) {
    // 均匀附近小扰动
    return Array.from({ length: NUM_EXPERTS }, (_, i) => 1 + (((i * 17) % 7) - 3) / 30);
  }
  // 不均匀:某几个 expert 长期被选 / 某几个永远闲置
  return [3.2, 2.6, 0.4, 0.2, 2.1, 0.3, 1.0, 0.2];
}

/** Top-k 选择的质量 vs 算力权衡 */
export const TOPK_COMPARE = [
  { k: 1, quality: 0.84, computeCostX: 1.0, note: "Switch Transformer · 最便宜但容量浪费一半" },
  { k: 2, quality: 0.91, computeCostX: 2.0, note: "Mixtral 选这个 · 质量↑↑ 算力 2× 还能接受" },
  { k: 4, quality: 0.93, computeCostX: 4.0, note: "GShard 也试过 · 质量收益边际,算力翻倍" },
  { k: 8, quality: 0.945, computeCostX: 8.0, note: "退化成 dense ensemble · 失去稀疏优势" },
];
