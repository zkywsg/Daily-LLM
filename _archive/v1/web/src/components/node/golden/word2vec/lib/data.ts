// 词频统计:模拟英文语料中部分词的频率分布(rank-frequency,Zipf 风格)
// 单位:每 1e6 tokens 出现次数
export interface WordFreq {
  word: string;
  count: number;          // per million
  category: "stopword" | "common" | "rare";
}

export const WORD_FREQS: WordFreq[] = [
  { word: "the", count: 56000, category: "stopword" },
  { word: "of",  count: 28000, category: "stopword" },
  { word: "and", count: 26000, category: "stopword" },
  { word: "a",   count: 22000, category: "stopword" },
  { word: "to",  count: 21000, category: "stopword" },
  { word: "is",  count: 12000, category: "stopword" },
  { word: "in",  count: 11000, category: "stopword" },
  { word: "it",  count: 9000,  category: "stopword" },
  { word: "fox",   count: 600, category: "common" },
  { word: "king",  count: 540, category: "common" },
  { word: "queen", count: 320, category: "common" },
  { word: "man",   count: 800, category: "common" },
  { word: "woman", count: 700, category: "common" },
  { word: "jumps", count: 180, category: "common" },
  { word: "brown", count: 220, category: "common" },
  { word: "quick", count: 150, category: "common" },
  { word: "monarch", count: 40, category: "rare" },
  { word: "regent",  count: 12, category: "rare" },
  { word: "throne",  count: 60, category: "rare" },
];

// 总词数(per million)用作分母
export const TOTAL_TOKENS_PER_M = WORD_FREQS.reduce((s, w) => s + w.count, 0);

export function frequency(word: string): number {
  const w = WORD_FREQS.find((x) => x.word === word);
  if (!w) return 0;
  return w.count / TOTAL_TOKENS_PER_M;
}

// Subsampling 公式:P_discard(w) = 1 - sqrt(t / f(w))
// 当 f(w) < t 时返回 0(全保留)
export function discardProb(freq: number, t: number = 1e-5): number {
  if (freq <= t) return 0;
  return Math.max(0, 1 - Math.sqrt(t / freq));
}

// Negative sampling 分布:P_n(w) ∝ count(w)^alpha
export function negSamplingDist(alpha: number = 0.75): Array<{ word: string; prob: number }> {
  const raw = WORD_FREQS.map((w) => ({ word: w.word, weight: Math.pow(w.count, alpha) }));
  const Z = raw.reduce((s, x) => s + x.weight, 0);
  return raw.map((x) => ({ word: x.word, prob: x.weight / Z }));
}

// 2D 词向量投影:king-man+woman ≈ queen 的可视化坐标
// 手工设计的坐标体现两个方向:gender (y) + royalty (x)
export interface WordPoint {
  word: string;
  x: number;  // royalty axis
  y: number;  // gender axis (negative = male, positive = female)
  group: "royalty" | "gender" | "country";
}

export const WORD_POINTS_2D: WordPoint[] = [
  { word: "king",   x: 0.78, y: -0.42, group: "royalty" },
  { word: "queen",  x: 0.80, y:  0.42, group: "royalty" },
  { word: "man",    x: 0.10, y: -0.40, group: "gender" },
  { word: "woman",  x: 0.12, y:  0.40, group: "gender" },
  { word: "prince", x: 0.62, y: -0.30, group: "royalty" },
  { word: "uncle",  x: 0.18, y: -0.32, group: "gender" },
  { word: "aunt",   x: 0.20, y:  0.32, group: "gender" },
  { word: "Paris",  x: 0.40, y:  0.70, group: "country" },
  { word: "France", x: 0.22, y:  0.78, group: "country" },
  { word: "Rome",   x: 0.66, y:  0.74, group: "country" },
  { word: "Italy",  x: 0.48, y:  0.82, group: "country" },
];

// 演示句子:用来展示 subsampling 把高频词丢掉
export const DEMO_SENTENCE = [
  "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
  "king", "of", "the", "woods", "and", "a", "small", "fox", "follows",
];
