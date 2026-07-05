// ============ 机制一:跨层参数共享 ============
// BERT vs ALBERT 的层参数组织方式(BERT-large 24 层为例)
export interface LayerShareRow {
  model: string;
  numLayers: number;
  perLayerParamsM: number; // 每层参数量(百万)
  shared: boolean;
}
export const LAYER_SHARE_COMPARE: LayerShareRow[] = [
  { model: "BERT-large", numLayers: 24, perLayerParamsM: 12, shared: false },
  { model: "ALBERT-large", numLayers: 24, perLayerParamsM: 12, shared: true },
];

// Table 7:不同共享策略的效果(论文)
export interface SharingStrategyRow {
  strategy: string;
  squadF1: number;
  paramsM: number;
}
export const SHARING_STRATEGY_TABLE: SharingStrategyRow[] = [
  { strategy: "全部独立(BERT-style)", squadF1: 90.4, paramsM: 89 },
  { strategy: "只共享 attention", squadF1: 89.9, paramsM: 64 },
  { strategy: "只共享 FFN", squadF1: 90.4, paramsM: 38 },
  { strategy: "全部共享(默认)", squadF1: 90.0, paramsM: 12 },
];

// ============ 机制二:Embedding 因式分解 ============
// V×H(原版) vs V×E + E×H(因式分解)的参数量对比
export interface EmbeddingFactorRow {
  model: string;
  vocabSize: number; // V
  hiddenSize: number; // H
}
export const EMBEDDING_MODELS: EmbeddingFactorRow[] = [
  { model: "BERT-base", vocabSize: 30000, hiddenSize: 768 },
  { model: "BERT-large", vocabSize: 30000, hiddenSize: 1024 },
  { model: "ALBERT-xxlarge", vocabSize: 30000, hiddenSize: 4096 },
];

export function rawEmbeddingParams(vocabSize: number, hiddenSize: number): number {
  return vocabSize * hiddenSize;
}

export function factorizedEmbeddingParams(vocabSize: number, embedSize: number, hiddenSize: number): number {
  return vocabSize * embedSize + embedSize * hiddenSize;
}

// ============ 机制三:NSP → SOP ============
// Table 5:预训练任务消融
export interface SopAblationRow {
  task: string;
  squadF1: number;
  race: number;
}
export const SOP_ABLATION_TABLE: SopAblationRow[] = [
  { task: "仅 MLM(无 NSP, 无 SOP)", squadF1: 81.0, race: 64.0 },
  { task: "MLM + NSP", squadF1: 81.5, race: 64.5 },
  { task: "MLM + SOP", squadF1: 82.1, race: 65.5 },
];

// ============ 性能与权衡(footer) ============
// Table 2:ALBERT 各版本 vs BERT 全量对比
export interface AlbertBertRow {
  model: string;
  paramsM: number;
  squadF1: number;
  mnli: number;
  race: number;
  trainHours: number;
}
export const ALBERT_BERT_TABLE: AlbertBertRow[] = [
  { model: "BERT-base", paramsM: 108, squadF1: 90.4, mnli: 84.6, race: 64.3, trainHours: 4.7 },
  { model: "BERT-large", paramsM: 334, squadF1: 92.2, mnli: 86.6, race: 70.4, trainHours: 11.1 },
  { model: "ALBERT-base", paramsM: 12, squadF1: 89.3, mnli: 81.6, race: 63.5, trainHours: 5.6 },
  { model: "ALBERT-large", paramsM: 18, squadF1: 90.9, mnli: 83.9, race: 66.0, trainHours: 17.7 },
  { model: "ALBERT-xlarge", paramsM: 60, squadF1: 93.0, mnli: 86.4, race: 73.9, trainHours: 41.8 },
  { model: "ALBERT-xxlarge", paramsM: 235, squadF1: 94.1, mnli: 88.1, race: 82.3, trainHours: 77.4 },
];
