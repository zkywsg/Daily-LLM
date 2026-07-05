// GPT-4 估计规格
export interface StatRow {
  label: string;
  value: string;
}
export const GPT4_STATS: StatRow[] = [
  { label: "参数量", value: "~1.8T(MoE 16 专家,每次激活 ~280B)" },
  { label: "训练 token", value: "~13T" },
  { label: "上下文窗口", value: "8K → 32K → 128K" },
  { label: "训练算力", value: "~2×10²⁵ FLOPs" },
  { label: "训练成本", value: "~1 亿美元" },
  { label: "训练硬件", value: "~25000 A100" },
];

// 上下文窗口扩展时间线
export interface ContextPoint {
  label: string;
  contextK: number;
}
export const CONTEXT_GROWTH: ContextPoint[] = [
  { label: "发布时", contextK: 8 },
  { label: "+4 个月", contextK: 32 },
  { label: "2023 年底", contextK: 128 },
];

// LLaMA-1 四档模型
export interface LlamaModel {
  name: string;
  layers: number;
  dModel: number;
  numHeads: number;
  tokensT: number;
  note: string;
}
export const LLAMA_MODELS: LlamaModel[] = [
  { name: "LLaMA-1 7B", layers: 32, dModel: 4096, numHeads: 32, tokensT: 1.0, note: "推理友好,多数 benchmark 与 GPT-3 175B 相当" },
  { name: "LLaMA-1 13B", layers: 40, dModel: 5120, numHeads: 40, tokensT: 1.0, note: "多个 benchmark 上击败 GPT-3 175B" },
  { name: "LLaMA-1 33B", layers: 60, dModel: 6656, numHeads: 52, tokensT: 1.4, note: "单卡可推理(A100 80GB)" },
  { name: "LLaMA-1 65B", layers: 80, dModel: 8192, numHeads: 64, tokensT: 1.4, note: "旗舰,接近 Chinchilla 70B / PaLM 540B" },
];

// over-train 数据/参数比对比
export interface OverTrainRow {
  model: string;
  paramsB: number;
  tokensB: number;
  ratio: number;
}
export const OVER_TRAIN_COMPARE: OverTrainRow[] = [
  { model: "Chinchilla optimal", paramsB: 70, tokensB: 1400, ratio: 20 },
  { model: "LLaMA-1 7B", paramsB: 7, tokensB: 1000, ratio: 143 },
  { model: "LLaMA-2 7B", paramsB: 7, tokensB: 2000, ratio: 286 },
  { model: "LLaMA-3 8B", paramsB: 8, tokensB: 15000, ratio: 1875 },
];

// 现代 LLM 配方 6 件套演化
export interface RecipeRow {
  component: string;
  original2017: string;
  gpt3: string;
  llama: string;
}
export const RECIPE_TABLE: RecipeRow[] = [
  { component: "Normalization 位置", original2017: "Post-LN", gpt3: "Pre-LN", llama: "Pre-LN" },
  { component: "Normalization 类型", original2017: "LayerNorm", gpt3: "LayerNorm", llama: "RMSNorm" },
  { component: "位置编码", original2017: "正余弦 PE", gpt3: "learned PE", llama: "RoPE" },
  { component: "FFN 激活", original2017: "ReLU", gpt3: "GELU", llama: "SwiGLU" },
  { component: "Attention 类型", original2017: "Multi-Head", gpt3: "Multi-Head + Sparse", llama: "MHA / GQA" },
  { component: "Tokenizer", original2017: "BPE", gpt3: "BPE", llama: "SentencePiece BPE" },
];

// 行业采用现代配方的模型列表(演示用)
export const RECIPE_ADOPTERS = ["LLaMA-1/2/3", "Mistral", "Mixtral", "Qwen", "Yi", "DeepSeek-V3"];

// 闭源-开源差距缩短时间线
export interface GapRow {
  era: string;
  months: number;
}
export const GAP_NARROWING: GapRow[] = [
  { era: "GPT-3 时代(2020-2022)", months: 18 },
  { era: "GPT-4/LLaMA 之后(2023+)", months: 6 },
];
