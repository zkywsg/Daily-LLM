/**
 * 把 12 个阶段字符串归并成 6 个叙事家族
 * —— 让 15 个年份节点也能一眼看出「在哪条主线」。
 * 颜色 token 在 global.css 中定义为 --phase-{family}-{ink|soft|line}
 */
export type PhaseFamily =
  | "foundation" // 理论起点
  | "vision" //     视觉线 / 决策智能
  | "language" //   语言线 / 生成与序列
  | "scale" //      预训练 / 规模化
  | "multimodal" // 多模态 / 开源与多模态
  | "alignment"; // 对齐 / 系统生产 / 推理模型

const PHASE_TO_FAMILY: Record<string, PhaseFamily> = {
  理论起点: "foundation",
  视觉线: "vision",
  决策智能: "vision",
  语言线: "language",
  生成与序列: "language",
  预训练: "scale",
  规模化: "scale",
  多模态: "multimodal",
  开源与多模态: "multimodal",
  对齐: "alignment",
  系统生产: "alignment",
  推理模型: "alignment",
};

export function phaseFamilyOf(phase: string): PhaseFamily {
  return PHASE_TO_FAMILY[phase] ?? "foundation";
}

export const PHASE_FAMILY_ORDER: PhaseFamily[] = [
  "foundation",
  "vision",
  "language",
  "scale",
  "multimodal",
  "alignment",
];

export const PHASE_FAMILY_LABEL: Record<PhaseFamily, string> = {
  foundation: "理论起点",
  vision: "视觉智能",
  language: "语言序列",
  scale: "规模与预训练",
  multimodal: "多模态汇流",
  alignment: "对齐与系统",
};
