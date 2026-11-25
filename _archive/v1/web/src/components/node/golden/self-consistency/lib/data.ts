// 一次 Self-Consistency 采样示例:农场羊奶问题,5 条路径投票
export interface SamplePath {
  id: number;
  reasoning: string;
  answer: string;
  correct: boolean;
}
export const SHEEP_MILK_PATHS: SamplePath[] = [
  { id: 1, reasoning: "每只羊每天 4 升,7 只共 28 升/天 × 7 天", answer: "196", correct: true },
  { id: 2, reasoning: "7 只羊 × 4 升 = 28 升每天,一周 7 × 28", answer: "196", correct: true },
  { id: 3, reasoning: "一只羊一周 4 × 7 = 28 升,7 只共 28 × 7", answer: "196", correct: true },
  { id: 4, reasoning: "总共 7 × 4 × 7", answer: "196", correct: true },
  { id: 5, reasoning: "每只羊每天 4 升,一周 28 升,7 只共 7 × 28(算错成 168)", answer: "168", correct: false },
];

// 贪婪解码 vs 温度采样一次(单条路径,无投票)
export interface SingleSampleRow {
  method: string;
  gsm8k: number;
}
export const SINGLE_SAMPLE_COMPARE: SingleSampleRow[] = [
  { method: "单次贪婪解码(T=0)", gsm8k: 56.5 },
  { method: "温度采样一次(T=0.7)", gsm8k: 55.4 },
];

// 采样次数 N vs GSM8K 准确率(PaLM 540B,对数线性增长、收益递减)
export interface AccuracyVsN {
  n: number;
  gsm8k: number;
  cost: string;
}
export const ACCURACY_VS_N: AccuracyVsN[] = [
  { n: 1, gsm8k: 56.5, cost: "1×" },
  { n: 5, gsm8k: 65.4, cost: "5×" },
  { n: 10, gsm8k: 70.0, cost: "10×" },
  { n: 20, gsm8k: 72.5, cost: "20×" },
  { n: 40, gsm8k: 74.4, cost: "40×" },
];

// CoT 单次 vs Self-Consistency(N=40)在多个 benchmark 上的对比(PaLM 540B)
export interface BenchmarkRow {
  benchmark: string;
  cot: number;
  selfConsistency: number;
  delta: number;
}
export const BENCHMARK_COMPARE: BenchmarkRow[] = [
  { benchmark: "GSM8K", cot: 56.5, selfConsistency: 74.4, delta: 17.9 },
  { benchmark: "AQuA-RAT", cot: 35.8, selfConsistency: 48.3, delta: 12.5 },
  { benchmark: "MultiArith", cot: 92.4, selfConsistency: 99.3, delta: 6.9 },
  { benchmark: "StrategyQA", cot: 75.3, selfConsistency: 81.6, delta: 6.3 },
];

// 答案归一化示例:不同表达 → 同一答案
export interface NormalizationExample {
  raw: string;
  normalized: string;
}
export const NORMALIZATION_EXAMPLES: NormalizationExample[] = [
  { raw: "196", normalized: "196" },
  { raw: "196 升", normalized: "196" },
  { raw: "the answer is 196", normalized: "196" },
  { raw: "196 liters", normalized: "196" },
];
