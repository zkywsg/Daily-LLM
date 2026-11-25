// Toolformer 集成的 5 个工具
export interface ToolInfo {
  name: string;
  backend: string;
}
export const TOOLS: ToolInfo[] = [
  { name: "QA", backend: "Atlas(Meta 的 retrieval QA 模型)" },
  { name: "Calculator", backend: "数学计算" },
  { name: "Wikipedia Search", backend: "BM25 检索" },
  { name: "Machine Translation", backend: "NLLB 翻译模型" },
  { name: "Calendar", backend: "返回当前日期" },
];

// 候选位置采样示例
export interface CandidatePosition {
  sentence: string;
  insertAt: number; // 字符位置(演示用)
  call: string;
  probAboveThreshold: boolean;
}
export const CANDIDATE_EXAMPLES: CandidatePosition[] = [
  { sentence: "1969 年人类首次登月。", insertAt: 0, call: 'QA("首次登月是哪一年?")', probAboveThreshold: true },
  { sentence: "法国总统是马克龙。", insertAt: 5, call: 'QA("法国总统是谁?")', probAboveThreshold: true },
  { sentence: "今天天气很好。", insertAt: 2, call: 'QA("今天天气?")', probAboveThreshold: false },
];

// Perplexity 过滤示例:带/不带 API 调用的 loss 对比
export interface FilterExample {
  call: string;
  lossMinus: number; // 不带调用
  lossPlus: number; // 带调用
  kept: boolean;
}
export const FILTER_EXAMPLES: FilterExample[] = [
  { call: 'QA("首次登月是哪一年?") → 1969', lossMinus: 4.8, lossPlus: 1.2, kept: true },
  { call: 'Calculator("132*4") → 528', lossMinus: 6.1, lossPlus: 0.9, kept: true },
  { call: 'QA("今天天气?") → 晴', lossMinus: 3.1, lossPlus: 2.9, kept: false },
  { call: 'Calendar() → 2023-02-14', lossMinus: 5.5, lossPlus: 1.8, kept: true },
];

// benchmark 对比(论文数据)
export interface BenchRow {
  task: string;
  gptJ: number;
  gptJCC: number;
  gpt3175B: number;
  opt66B: number;
  toolformer: number;
}
export const BENCHMARK_TABLE: BenchRow[] = [
  { task: "LAMA(事实问答)", gptJ: 17.8, gptJCC: 19.5, gpt3175B: 31.3, opt66B: 22.5, toolformer: 29.7 },
  { task: "ASDiv(数学)", gptJ: 7.5, gptJCC: 9.6, gpt3175B: 14.0, opt66B: 6.0, toolformer: 40.4 },
  { task: "MathQA", gptJ: 4.3, gptJCC: 3.5, gpt3175B: 12.5, opt66B: 4.3, toolformer: 20.6 },
  { task: "SVAMP(数学应用题)", gptJ: 5.2, gptJCC: 5.4, gpt3175B: 10.0, opt66B: 6.2, toolformer: 29.4 },
  { task: "MLQA(跨语言QA)", gptJ: 8.4, gptJCC: 7.4, gpt3175B: 25.6, opt66B: 14.4, toolformer: 20.6 },
  { task: "TempLAMA(时序事实)", gptJ: 13.7, gptJCC: 12.3, gpt3175B: 23.6, opt66B: 14.6, toolformer: 16.3 },
];
