// Switch Transformer demo 数据:top-1 vs top-k 路由、负载均衡、capacity factor、
// selective precision、参数规模表。不真跑 MoE forward,用确定性 hash 让 viewer
// 看到 "每 token 只选 1 个 expert" 的行为,并与 Shazeer top-K / Mixtral top-2 对比。

export const NUM_EXPERTS = 8;

/** 给定 token 文本,产生一个 N 维 router logits 向量(确定性) */
export function routerLogits(token: string, seed = 0): number[] {
  const logits: number[] = [];
  for (let e = 0; e < NUM_EXPERTS; e++) {
    let h = seed + e * 31;
    for (const c of token) h = (h * 131 + c.charCodeAt(0)) >>> 0;
    logits.push(((h % 1000) / 1000) * 4 - 2);
  }
  return logits;
}

export function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** 返回 token 对应的 top-k expert 索引 + softmax 权重(k=1 时就是 Switch 的 argmax 路由) */
export function topKExperts(token: string, k: number, seed = 0): Array<{ expert: number; weight: number }> {
  const logits = routerLogits(token, seed);
  const indexed = logits.map((l, i) => ({ i, l }));
  indexed.sort((a, b) => b.l - a.l);
  const top = indexed.slice(0, k);
  const topLogits = top.map((t) => t.l);
  const topWeights = softmax(topLogits);
  return top.map((t, idx) => ({ expert: t.i, weight: topWeights[idx] }));
}

export const DEMO_SENTENCES: Array<{ label: string; tokens: string[] }> = [
  { label: "中文短句", tokens: ["大", "模型", "稀疏", "路由", "省", "算力"] },
  { label: "English", tokens: ["The", "router", "picks", "just", "one", "expert"] },
  { label: "代码 token", tokens: ["def", "forward", "(", "x", ")", ":"] },
];

/** Top-1 (Switch) vs Top-K (Shazeer) 的路由算力/通信对比 */
export const ROUTING_COMPARE = [
  { k: 1, label: "Switch Transformer", computeCostX: 1.0, commCostX: 1.0, note: "每 token 只算 1 个 expert · 路由与 all-to-all 通信最省" },
  { k: 2, label: "Mixtral", computeCostX: 2.0, commCostX: 2.0, note: "质量↑但算力翻倍" },
  { k: 4, label: "Shazeer 2017 (LSTM-MoE)", computeCostX: 4.0, commCostX: 4.0, note: "祖师爷设定 K=4 · 通信量是 Switch 的 4×" },
];

/** 模拟 expert 负载:无 aux loss 时极不均匀,加 aux loss 后均匀 */
export function expertLoad(balanced: boolean): number[] {
  if (balanced) {
    return Array.from({ length: NUM_EXPERTS }, (_, i) => 1 + (((i * 17) % 7) - 3) / 30);
  }
  return [3.4, 2.8, 0.3, 0.2, 2.3, 0.2, 0.9, 0.1];
}

/** capacity factor → 每 expert 能装的 token 数上限(相对单位,1.0 = 均匀理想值) */
export function capacityOverflow(capacityFactor: number): { capacity: number; overflowExperts: number[]; overflowFrac: number } {
  const loads = expertLoad(false);
  const capacity = capacityFactor; // 理想负载=1.0 时,capacity_factor 就是每 expert 的容量上限(相对单位)
  const overflowExperts = loads
    .map((l, i) => ({ i, over: l > capacity }))
    .filter((x) => x.over)
    .map((x) => x.i);
  const totalOverflow = loads.reduce((s, l) => s + Math.max(0, l - capacity), 0);
  const totalLoad = loads.reduce((s, l) => s + l, 0);
  return { capacity, overflowExperts, overflowFrac: totalOverflow / totalLoad };
}

/** Selective precision:router softmax/log 用 fp32,其余用 bf16 —— 对比训练 loss 是否发散 */
export const PRECISION_COMPARE: Array<{ mode: "fp32-all" | "bf16-all" | "selective"; label: string; stable: boolean; memoryX: number; note: string }> = [
  { mode: "fp32-all", label: "全 fp32", stable: true, memoryX: 2.0, note: "稳定但显存翻倍 · 大规模训练太贵" },
  { mode: "bf16-all", label: "全 bf16", stable: false, memoryX: 1.0, note: "省显存但 router softmax/log 精度不够 · 经常 NaN 发散" },
  { mode: "selective", label: "Selective Precision(Switch 方案)", stable: true, memoryX: 1.05, note: "主体 bf16 + router 用 fp32 · 稳定且几乎不多花显存" },
];

/** 模型规模表(来自论文表格) */
export interface ModelRow {
  name: string;
  totalParams: number; // in params (not tokens)
  activeParams: number;
  numExperts: number;
  c4Perplexity: number;
  stepsToT5BaseQuality: number; // relative, 1.0 = T5-Base baseline
}

export const MODEL_SCALE_TABLE: ModelRow[] = [
  { name: "T5-Base (dense)", totalParams: 220e6, activeParams: 220e6, numExperts: 1, c4Perplexity: 5.85, stepsToT5BaseQuality: 1.0 },
  { name: "T5-Large", totalParams: 770e6, activeParams: 770e6, numExperts: 1, c4Perplexity: 5.24, stepsToT5BaseQuality: 2.0 },
  { name: "T5-XXL", totalParams: 11e9, activeParams: 11e9, numExperts: 1, c4Perplexity: 4.65, stepsToT5BaseQuality: 60 },
  { name: "Switch-Base", totalParams: 7e9, activeParams: 0.22e9, numExperts: 128, c4Perplexity: 5.32, stepsToT5BaseQuality: 0.5 },
  { name: "Switch-Large", totalParams: 26e9, activeParams: 0.77e9, numExperts: 128, c4Perplexity: 4.87, stepsToT5BaseQuality: 0.4 },
  { name: "Switch-XXL", totalParams: 395e9, activeParams: 11e9, numExperts: 64, c4Perplexity: 4.41, stepsToT5BaseQuality: 0.25 },
  { name: "Switch-C", totalParams: 1.57e12, activeParams: 11e9, numExperts: 2048, c4Perplexity: 4.05, stepsToT5BaseQuality: 0 },
];

/** 下游 fine-tune 任务分数 */
export interface BenchmarkRow {
  name: string;
  superGlue: number;
  glue: number;
  squadF1: number;
}

export const BENCHMARK_TABLE: BenchmarkRow[] = [
  { name: "T5-Base", superGlue: 76.2, glue: 84.0, squadF1: 83.6 },
  { name: "Switch-Base (7B)", superGlue: 77.5, glue: 85.0, squadF1: 85.4 },
  { name: "T5-Large", superGlue: 82.9, glue: 87.7, squadF1: 87.5 },
  { name: "Switch-Large (26B)", superGlue: 84.7, glue: 88.0, squadF1: 89.2 },
];
