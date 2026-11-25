// Constitution 原则示例(摘录,原版为英文,这里给中文释义用于图示)
export interface ConstitutionPrinciple {
  id: number;
  label: string;
  principle: string;
}

export const CONSTITUTION_PRINCIPLES: ConstitutionPrinciple[] = [
  { id: 4, label: "#4 合法性", principle: "回答不应包含非法、危险或有害内容" },
  { id: 7, label: "#7 友善度", principle: "回答应尽量礼貌、体贴,并给出有益的替代方案" },
  { id: 12, label: "#12 非西方视角", principle: "回答应尽量不被非西方文化背景的读者视为冒犯" },
  { id: 15, label: "#15 医疗谨慎", principle: "回答不应给人以医疗权威或专业建议的印象" },
];

// SL-CAI 单个 prompt 的 critique-revise 流程(演示用 4 步)
export interface CritiqueRevisionStep {
  stage: "harmful" | "critique" | "revise" | "final";
  label: string;
  text: string;
}

export function buildCritiqueRevisionFlow(principle: ConstitutionPrinciple): CritiqueRevisionStep[] {
  return [
    { stage: "harmful", label: "有害 prompt 的初始回答", text: "Here's how to make explosives at home..." },
    {
      stage: "critique",
      label: `AI 按 ${principle.label} 批评`,
      text: `This response violates principle: "${principle.principle}"`,
    },
    { stage: "revise", label: "AI 重写回答", text: "I cannot provide instructions for making explosives, as they could cause serious harm." },
    { stage: "final", label: "无害替代回答", text: "I can't help with that, but if you're interested in chemistry, I can suggest some safe experiments." },
  ];
}

// RLAIF vs RLHF:标注流程对比(标注员规模 / 周期 / 成本 — 定性对比用于图示比例)
export interface LabelPipelineRow {
  method: string;
  laborMonths: number; // 人月(RLAIF 记为 AI 推理无人月)
  costUSD: number; // 估计成本(美元)
  pairs: number; // 偏好对数量(千)
}

export const LABEL_PIPELINE_COMPARE: LabelPipelineRow[] = [
  { method: "InstructGPT(人工 RLHF)", laborMonths: 240, costUSD: 3_500_000, pairs: 33 },
  { method: "Constitutional AI(RLAIF)", laborMonths: 0, costUSD: 350_000, pairs: 30 },
];

// Table 1(论文):helpfulness / harmlessness 胜率对比(人工评分,vs SFT-only 基线)
export interface HelpfulHarmlessRow {
  method: string;
  helpfulness: number; // 胜率 %,vs SFT-only
  harmlessness: number; // 胜率 %,vs 基准(正值更无害)
}

export const HELPFUL_HARMLESS_COMPARE: HelpfulHarmlessRow[] = [
  { method: "Helpful-only RLHF baseline", helpfulness: 51, harmlessness: -23 },
  { method: "Standard RLHF(人类反馈)", helpfulness: 50, harmlessness: 0 },
  { method: "Constitutional AI(RLAIF)", helpfulness: 51, harmlessness: 9 },
];

// 训练细节表
export interface TrainingDetailRow {
  dimension: string;
  value: string;
}

export const TRAINING_DETAILS: TrainingDetailRow[] = [
  { dimension: "Backbone", value: "Anthropic 内部 LM(约 52B,细节未公开)" },
  { dimension: "Constitution", value: "16 条原则,约 1500 字" },
  { dimension: "有害 prompt 数据", value: "~16K 条 red team prompts" },
  { dimension: "SL-CAI 迭代轮数", value: "通常 4 轮 critique-revise" },
  { dimension: "RLAIF 偏好数据", value: "~30K AI 偏好对(InstructGPT 为 33K 人类偏好)" },
  { dimension: "总成本", value: "主要是算力 — 估计 InstructGPT 的 1/10 到 1/5" },
];
