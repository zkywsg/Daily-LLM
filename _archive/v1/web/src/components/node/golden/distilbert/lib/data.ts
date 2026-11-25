// DistilBERT vs BERT-base 架构与规模对比
export interface ArchRow {
  dim: string;
  bert: string;
  distilbert: string;
}
export const ARCH_COMPARE: ArchRow[] = [
  { dim: "层数", bert: "12", distilbert: "6" },
  { dim: "d_model", bert: "768", distilbert: "768(同 BERT)" },
  { dim: "attention heads", bert: "12", distilbert: "12(同)" },
  { dim: "d_ff", bert: "3072", distilbert: "3072(同)" },
  { dim: "参数量", bert: "110M", distilbert: "66M" },
  { dim: "推理速度(V100)", bert: "1x", distilbert: "1.6x" },
  { dim: "GLUE 平均", bert: "79.5", distilbert: "77.0" },
];

// 隔层初始化:BERT 12 层 -> DistilBERT 6 层的映射
export const TEACHER_LAYERS = 12;
export const STUDENT_LAYERS = 6;
// 用 BERT 第 1,3,5,7,9,11 层(0-indexed: 1,3,5,7,9,11)初始化 DistilBERT
export const INIT_LAYER_INDICES: number[] = [1, 3, 5, 7, 9, 11];

// 温度缩放对 softmax 分布的影响(教师在情感三分类上的示例 logits)
export const TEACHER_LOGITS: { label: string; logit: number }[] = [
  { label: "正面", logit: 3.2 },
  { label: "负面", logit: 1.0 },
  { label: "中性", logit: -0.4 },
];

export function softmaxWithTemperature(logits: number[], temperature: number): number[] {
  const scaled = logits.map((z) => z / temperature);
  const maxZ = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - maxZ));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// 三损失的相对权重(论文超参)
export interface LossWeight {
  name: string;
  symbol: string;
  weight: number;
  desc: string;
}
export const LOSS_WEIGHTS: LossWeight[] = [
  { name: "Distillation(KL)", symbol: "α", weight: 0.5, desc: "模仿 teacher 软分布" },
  { name: "MLM(硬标签)", symbol: "β", weight: 0.2, desc: "原始 masked LM 任务" },
  { name: "Cosine(隐状态对齐)", symbol: "γ", weight: 0.1, desc: "中间表征方向对齐" },
];

// GLUE 9 任务成绩(论文 Table 1,节选)
export interface GlueRow {
  task: string;
  bert: number;
  distilbert: number;
}
export const GLUE_TABLE: GlueRow[] = [
  { task: "MNLI", bert: 86.7, distilbert: 82.2 },
  { task: "QNLI", bert: 91.8, distilbert: 89.2 },
  { task: "QQP", bert: 89.6, distilbert: 88.5 },
  { task: "SST-2", bert: 93.5, distilbert: 91.3 },
  { task: "CoLA", bert: 56.3, distilbert: 51.3 },
  { task: "MRPC", bert: 88.6, distilbert: 87.5 },
  { task: "GLUE 平均", bert: 79.5, distilbert: 77.0 },
];

// 速度对比(V100,batch 1,序列 128)
export interface SpeedRow {
  model: string;
  paramsM: number;
  latencyMs: number;
  tps: number;
}
export const SPEED_TABLE: SpeedRow[] = [
  { model: "BERT-large", paramsM: 340, latencyMs: 60, tps: 15 },
  { model: "BERT-base", paramsM: 110, latencyMs: 15, tps: 60 },
  { model: "DistilBERT", paramsM: 66, latencyMs: 9, tps: 100 },
];

// 参数量 vs GLUE 保留率 vs 推理速度(工业部署三维对比:DistilBERT vs BERT vs ALBERT)
export interface TradeoffRow {
  model: string;
  paramsM: number;
  glueRetainPct: number;
  speedupX: number;
}
export const TRADEOFF_TABLE: TradeoffRow[] = [
  { model: "BERT-base", paramsM: 110, glueRetainPct: 100, speedupX: 1 },
  { model: "ALBERT-base", paramsM: 12, glueRetainPct: 97, speedupX: 1 },
  { model: "DistilBERT", paramsM: 66, glueRetainPct: 97, speedupX: 1.6 },
];
