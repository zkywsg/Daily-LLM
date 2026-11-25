// DeepSeek-V3 demo 数据:细粒度专家对比、aux-loss-free 负载均衡模拟、MLA KV cache 压缩、
// 性能/成本对比表。表格数字取自 13-moe-efficient/04-deepseek-v3.md 正文(技术报告 Table 4 等),
// 负载均衡模拟用确定性函数生成 demo 效果(实际 256 expert,这里压缩到 8 个演示)。

export interface GranularityRow {
  model: string;
  expertsPerLayer: string;
  topK: string;
  expertSize: string;
  activeExperts: string;
}

export const GRANULARITY_COMPARE: GranularityRow[] = [
  { model: "Mixtral 8×7B", expertsPerLayer: "8", topK: "2", expertSize: "~5.5B", activeExperts: "~11B" },
  { model: "DeepSeek-V3", expertsPerLayer: "256 routed + 1 shared", topK: "8", expertSize: "~0.5B", activeExperts: "~4B(8 个 small)" },
];

export interface ParamRow {
  model: string;
  totalParamsB: number;
  activeParamsB: number;
}

export const PARAM_COMPARE: ParamRow[] = [
  { model: "LLaMA-3.1-405B", totalParamsB: 405, activeParamsB: 405 },
  { model: "Mixtral 8×7B", totalParamsB: 46.7, activeParamsB: 12.9 },
  { model: "DeepSeek-V3", totalParamsB: 671, activeParamsB: 37 },
];

// KV cache:MLA 把 KV cache 压缩到传统 MHA 的 ~1/4(文档原话,相对单位)
export interface KvCacheRow {
  method: string;
  relativeSize: number;
  note: string;
}

export const KV_CACHE_COMPARE: KvCacheRow[] = [
  { method: "传统 MHA", relativeSize: 4, note: "每 head 独立存完整 K/V" },
  { method: "MLA(V3)", relativeSize: 1, note: "压缩到低秩 latent,按需解压" },
];

// Aux-loss-free 负载均衡模拟(演示压缩到 8 个 expert,实际 256 个)
export const NUM_ROUTED_SAMPLE = 8;

/** progress: 0 = 训练刚开始(bias 还没生效,负载不均),1 = 训练后期(bias 已调好,接近均衡) */
export function expertLoadAuxFree(progress: number): number[] {
  const base = [3.0, 0.3, 2.4, 0.5, 1.8, 0.4, 2.6, 0.6];
  return base.map((v) => v + (1 - v) * progress);
}

/** 传统 aux loss(Switch / Mixtral 风格):靠梯度惩罚强制拉平,从头到尾都接近均衡,但会和主任务 loss 打架 */
export function expertLoadAuxLoss(): number[] {
  return Array.from({ length: NUM_ROUTED_SAMPLE }, (_, i) => 1 + (((i * 13) % 5) - 2) / 20);
}

// 性能数据(技术报告 Table 4 原始数字)
export interface BenchmarkRow {
  model: string;
  mmlu: number;
  mmluPro: number;
  dropF1: number;
  math: number;
  humanEval: number;
  liveCodeBench: number;
  highlight?: boolean;
}

export const BENCHMARK_TABLE: BenchmarkRow[] = [
  { model: "LLaMA-3.1-405B", mmlu: 88.6, mmluPro: 73.3, dropF1: 84.8, math: 73.8, humanEval: 89.0, liveCodeBench: 28.4 },
  { model: "GPT-4o(08-2024)", mmlu: 87.2, mmluPro: 72.6, dropF1: 83.7, math: 76.6, humanEval: 91.0, liveCodeBench: 33.4 },
  { model: "Claude-3.5-Sonnet", mmlu: 88.3, mmluPro: 78.0, dropF1: 88.3, math: 78.3, humanEval: 92.0, liveCodeBench: 36.3 },
  { model: "Qwen-2.5-72B", mmlu: 85.0, mmluPro: 71.6, dropF1: 76.7, math: 80.0, humanEval: 86.6, liveCodeBench: 31.4 },
  { model: "DeepSeek-V3", mmlu: 88.5, mmluPro: 75.9, dropF1: 84.0, math: 90.2, humanEval: 82.6, liveCodeBench: 40.5, highlight: true },
];

// 训练/推理成本对比
export interface CostRow {
  model: string;
  totalParams: string;
  activeParams: string;
  trainingCost: string;
  inferenceCost: string;
}

export const COST_TABLE: CostRow[] = [
  { model: "LLaMA-3.1-405B", totalParams: "405B", activeParams: "405B", trainingCost: "~$60M", inferenceCost: "~$5-10" },
  { model: "GPT-4o", totalParams: "?", activeParams: "?", trainingCost: "~$100M+", inferenceCost: "$2.5-10" },
  { model: "Claude-3.5-Sonnet", totalParams: "?", activeParams: "?", trainingCost: "~$100M+", inferenceCost: "$3-15" },
  { model: "DeepSeek-V3", totalParams: "671B", activeParams: "37B", trainingCost: "$5.6M", inferenceCost: "$0.27-1.1" },
];
