// ICL 涌现:LLM 规模 vs few-shot 提升幅度
export interface LlmScaleRow {
  name: string;
  params: number; // B
  zeroShot: number;
  fourShot: number;
  hasICL: boolean;
}
export const LLM_SCALE_ICL: LlmScaleRow[] = [
  { name: "OPT-1.3B",      params: 1.3,  zeroShot: 42, fourShot: 43, hasICL: false },
  { name: "Flan-T5 11B",   params: 11,   zeroShot: 45, fourShot: 46, hasICL: false },
  { name: "Chinchilla 70B",params: 70,   zeroShot: 49.2, fourShot: 56.3, hasICL: true },
];

// VQAv2 shot 曲线
export interface ShotRow {
  shots: number;
  vqav2: number;
  okvqa: number;
  textvqa: number;
}
export const SHOT_CURVE: ShotRow[] = [
  { shots: 0,  vqav2: 49.2, okvqa: 41.2, textvqa: 30.1 },
  { shots: 4,  vqav2: 56.3, okvqa: 47.4, textvqa: 32.7 },
  { shots: 32, vqav2: 60.0, okvqa: 50.6, textvqa: 36.0 },
];

// Gated cross-attention 训练动态:α 随 step 变化
export function gateAlpha(step: number): number {
  // sigmoid-ish ramp: 0 at step 0, ramps 1K-10K, stable after
  const x = (step - 3000) / 2000;
  return Math.tanh(Math.max(0, 1 / (1 + Math.exp(-x))) * 1.2);
}

// 三个 VLM 架构对比
export interface VlmCompareRow {
  name: string;
  llm: string;
  llmSize: string;
  frozen: boolean;
  bridge: string;
  icl: string;
  vqav2: string;
}
export const VLM_COMPARE: VlmCompareRow[] = [
  { name: "Flamingo", llm: "Chinchilla", llmSize: "70B", frozen: true,  bridge: "Perceiver Resampler + 间隔 gated cross-attn", icl: "强(4-shot 56.3)", vqav2: "4-shot 56.3" },
  { name: "BLIP-2",   llm: "Flan-T5",    llmSize: "11B", frozen: true,  bridge: "Q-Former",                                    icl: "弱",              vqav2: "0-shot 65.2" },
  { name: "LLaVA",    llm: "LLaMA",      llmSize: "7B",  frozen: false, bridge: "Linear projection",                           icl: "弱(强对话)",     vqav2: "instruction tuned" },
];
