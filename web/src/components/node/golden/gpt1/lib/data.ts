// 旧范式 vs GPT-1 新范式对比数据
export interface ParadigmTask {
  task: string;
  oldModel: string;
  dataSize: string;
}
export const OLD_PARADIGM_TASKS: ParadigmTask[] = [
  { task: "情感分类", oldModel: "BiLSTM",     dataSize: "~10K 条" },
  { task: "阅读理解", oldModel: "BiDAF",       dataSize: "~100K 条" },
  { task: "NER",      oldModel: "BiLSTM-CRF", dataSize: "~20K 条" },
  { task: "NLI",      oldModel: "ESIM",       dataSize: "~500K 条" },
];

// 4 种任务输入格式
export interface TaskFormat {
  name: string;
  format: string;
  example: string;
  forwardCount: string;
}
export const TASK_FORMATS: TaskFormat[] = [
  { name: "分类(单句)",     format: "<s> 文本 </s>",               example: "<s> This movie is great </s>",                    forwardCount: "1 次" },
  { name: "NLI(句对)",       format: "<s> 前提 <$> 假设 </s>",       example: "<s> A man is running <$> Someone is moving </s>", forwardCount: "1 次" },
  { name: "相似度(对称)",    format: "<s> A <$> B </s> + <s> B <$> A </s>", example: "两次 forward 求和",                          forwardCount: "2 次" },
  { name: "多选(QA)",        format: "<s> 文档+问题 <$> c_i </s>",   example: "N 个候选各一次,softmax 选最优",                    forwardCount: "N 次" },
];

// 12 个 benchmark 成绩
export interface BenchmarkRow {
  task: string;
  prevSota: number;
  gpt1: number;
  gain: number;
  category: "nli" | "reading" | "similarity" | "sentiment" | "grammar";
}
export const BENCHMARK_RESULTS: BenchmarkRow[] = [
  { task: "MNLI",     prevSota: 80.6, gpt1: 82.1, gain: 1.5,  category: "nli" },
  { task: "SNLI",     prevSota: 89.3, gpt1: 89.9, gain: 0.6,  category: "nli" },
  { task: "QNLI",     prevSota: 82.3, gpt1: 88.1, gain: 5.8,  category: "nli" },
  { task: "RTE",      prevSota: 61.7, gpt1: 56.0, gain: -5.7, category: "nli" },
  { task: "SciTail",  prevSota: 83.3, gpt1: 88.3, gain: 5.0,  category: "nli" },
  { task: "RACE-h",   prevSota: 53.3, gpt1: 59.0, gain: 5.7,  category: "reading" },
  { task: "RACE-m",   prevSota: 55.7, gpt1: 62.9, gain: 7.2,  category: "reading" },
  { task: "StoryCloze", prevSota: 77.6, gpt1: 86.5, gain: 8.9, category: "reading" },
  { task: "MRPC",     prevSota: 86.0, gpt1: 82.3, gain: -3.7, category: "similarity" },
  { task: "QQP",      prevSota: 70.3, gpt1: 70.3, gain: 0.0,  category: "similarity" },
  { task: "SST-2",    prevSota: 91.6, gpt1: 91.3, gain: -0.3, category: "sentiment" },
  { task: "CoLA",     prevSota: 35.0, gpt1: 45.4, gain: 10.4, category: "grammar" },
];

// 2018-2024 GPT 路线规模演化(复用于 footer 图)
export interface RouteEvolution {
  name: string;
  year: number;
  params: string;
  keyFeature: string;
}
export const ROUTE_EVOLUTION: RouteEvolution[] = [
  { name: "GPT-1",  year: 2018, params: "117M",  keyFeature: "预训练 + 微调" },
  { name: "GPT-2",  year: 2019, params: "1.5B",  keyFeature: "zero-shot 涌现" },
  { name: "GPT-3",  year: 2020, params: "175B",  keyFeature: "in-context learning" },
  { name: "LLaMA/GPT-4", year: 2023, params: "~1T?", keyFeature: "decoder-only 全面主导" },
];
