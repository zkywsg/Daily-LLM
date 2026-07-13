// 机制一:Layer-wise Prefix on K/V —— prefix 只加在 K/V,不加在 Q
export const PREFIX_LEN = 50; // m,prefix token 数(示意值,论文中 10-200 均有实验)
export const KV_INJECTION = {
  prefixLen: PREFIX_LEN,
  inputTokens: 6, // 示意:k_1...k_n
  note: "Q 不变,只在 K、V 前面拼 m 个可学习 prefix token",
};

// 机制二:MLP 重参数化 —— P_small → MLP → 展开到所有层 K/V
export const MLP_REPARAM = {
  m: 50, // prefix token 数(源码 forward 示例用 prefix_len,论文参数账例用 m=10)
  dSmall: 512, // P_small 的隐藏维度
  numLayers: 12, // GPT-2(base/medium)
  hiddenSize: 768,
};

// 参数账:GPT-2 large(354M),m=10 → 240K prefix 参数 ≈ 0.07%
export const PARAM_BUDGET_GPT2_LARGE = {
  totalParams: 354_000_000,
  m: 10,
  prefixParams: 240_000,
  prefixParamsPct: 0.07,
};

// HuggingFace peft 库实测:PrefixTuningConfig(num_virtual_tokens=20, prefix_projection=True)
export const PARAM_BUDGET_PEFT_EXAMPLE = {
  trainableParams: 184_320,
  totalParams: 774_030_080,
  trainablePct: 0.024,
};

// 机制三:Full FT vs Prefix Tuning 对比表
export interface CompareRow {
  dimension: string;
  fullFt: string;
  prefixTuning: string;
}
export const FULL_FT_VS_PREFIX: CompareRow[] = [
  { dimension: "更新参数", fullFt: "全部(100%)", prefixTuning: "仅 prefix(0.1%)" },
  { dimension: "推理开销", fullFt: "0", prefixTuning: "seq 长度 +m,attention O((n+m)²)" },
  { dimension: "训练显存", fullFt: "优化器全状态", prefixTuning: "只优化 prefix" },
  { dimension: "多任务部署", fullFt: "每任务一份 model", prefixTuning: "一份 base + N 个 prefix" },
  { dimension: "灾难性遗忘", fullFt: "有", prefixTuning: "无(base 冻结)" },
];

// 性能数据:E2E NLG / WebNLG(GPT-2 medium)
export interface PerformanceRow {
  method: string;
  trainedParamsPct: number;
  e2eBleu: number;
  e2eRougeL: number;
  webNlgBleu: number;
  color: string;
  bg: string;
}
export const PERFORMANCE_COMPARE: PerformanceRow[] = [
  { method: "Full FT", trainedParamsPct: 100, e2eBleu: 68.2, e2eRougeL: 70.6, webNlgBleu: 47.6, color: "#9ca3af", bg: "#f3f4f6" },
  { method: "Adapter(0.1%)", trainedParamsPct: 0.1, e2eBleu: 67.7, e2eRougeL: 69.5, webNlgBleu: 45.2, color: "#3b82f6", bg: "#dbeafe" },
  { method: "Adapter(3%)", trainedParamsPct: 3.0, e2eBleu: 68.4, e2eRougeL: 70.7, webNlgBleu: 48.0, color: "#f59e0b", bg: "#fef3c7" },
  { method: "Prefix Tuning(0.1%)", trainedParamsPct: 0.1, e2eBleu: 70.3, e2eRougeL: 72.1, webNlgBleu: 47.7, color: "#ec4899", bg: "#fce7f3" },
];

// Scaling:GPT-2 medium / large / XL —— Prefix Tuning vs Full FT 的差距
export interface ScalingRow {
  model: string;
  params: string;
  prefixTuning: number;
  fullFt: number;
  gap: number;
}
export const SCALING_TREND: ScalingRow[] = [
  { model: "GPT-2 medium", params: "354M", prefixTuning: 70.3, fullFt: 68.2, gap: 2.1 },
  { model: "GPT-2 large", params: "774M", prefixTuning: 70.4, fullFt: 68.5, gap: 1.9 },
  { model: "GPT-2 XL", params: "1.5B", prefixTuning: 70.6, fullFt: 68.6, gap: 2.0 },
];
