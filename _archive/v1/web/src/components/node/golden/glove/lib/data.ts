// 共现概率比值表 (论文 Table 1: ice / steam)
export interface ProbeRow {
  probe: string;
  pIce: number;
  pSteam: number;
  ratio: number;
  interp: string;
}
export const PROBE_TABLE: ProbeRow[] = [
  { probe: "solid",   pIce: 1.9e-4, pSteam: 2.2e-5, ratio: 8.9,   interp: "强区分 · 偏 ice" },
  { probe: "gas",     pIce: 6.6e-5, pSteam: 7.8e-4, ratio: 0.085, interp: "强区分 · 偏 steam" },
  { probe: "water",   pIce: 3.0e-3, pSteam: 2.2e-3, ratio: 1.36,  interp: "都相关" },
  { probe: "fashion", pIce: 1.7e-5, pSteam: 1.8e-5, ratio: 0.96,  interp: "都不相关" },
];

// 加权函数 f(x)
export function weightFn(x: number, xMax: number = 100, alpha: number = 0.75): number {
  if (x >= xMax) return 1.0;
  return Math.pow(x / xMax, alpha);
}

// GloVe vs Word2Vec similarity 任务对比
export interface SimRow {
  task: string;
  word2vec: number;
  glove: number;
}
export const SIMILARITY_COMPARE: SimRow[] = [
  { task: "WS353", word2vec: 65.6, glove: 75.9 },
  { task: "MC",    word2vec: 75.4, glove: 83.6 },
  { task: "RG",    word2vec: 72.4, glove: 82.9 },
  { task: "SCWS",  word2vec: 60.7, glove: 62.9 },
  { task: "RW",    word2vec: 38.5, glove: 47.8 },
];

// Analogy 任务对比(不同语料规模)
export interface AnalogyRow {
  model: string;
  corpus: string;
  semantic: number;
  syntactic: number;
  total: number;
}
export const ANALOGY_COMPARE: AnalogyRow[] = [
  { model: "ivLBL",         corpus: "1.5B", semantic: 60.0, syntactic: 50.1, total: 53.2 },
  { model: "Word2Vec SG",   corpus: "6B",   semantic: 73.0, syntactic: 66.0, total: 69.1 },
  { model: "GloVe (6B)",    corpus: "6B",   semantic: 77.4, syntactic: 67.0, total: 71.7 },
  { model: "GloVe (42B)",   corpus: "42B",  semantic: 81.9, syntactic: 69.3, total: 75.0 },
];

// 预训练版本
export interface PretrainedVersion {
  name: string;
  corpus: string;
  tokens: string;
  vocab: string;
}
export const PRETRAINED_VERSIONS: PretrainedVersion[] = [
  { name: "glove.6B",         corpus: "Wikipedia + Gigaword", tokens: "6B",   vocab: "400K" },
  { name: "glove.42B.300d",   corpus: "Common Crawl",         tokens: "42B",  vocab: "1.9M" },
  { name: "glove.840B.300d",  corpus: "Common Crawl",         tokens: "840B", vocab: "2.2M" },
  { name: "glove.twitter.27B",corpus: "Twitter",               tokens: "27B",  vocab: "1.2M" },
];
