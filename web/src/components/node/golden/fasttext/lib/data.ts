// Subword n-gram 分解:词 → {<..边界符 n-gram, 整词 token}
export function getSubwords(word: string, nMin: number = 3, nMax: number = 6): string[] {
  const bounded = `<${word}>`;
  const grams: string[] = [];
  for (let n = nMin; n <= nMax; n++) {
    if (n > bounded.length) break;
    for (let i = 0; i <= bounded.length - n; i++) {
      grams.push(bounded.slice(i, i + n));
    }
  }
  grams.push(bounded); // 整词 token
  return grams;
}

// 简单字符串 hash(用于 bucket 演示,非生产实现)
export function hashSubword(subword: string, buckets: number): number {
  let h = 0;
  for (let i = 0; i < subword.length; i++) {
    h = (h * 31 + subword.charCodeAt(i)) >>> 0;
  }
  return h % buckets;
}

export interface OovExample {
  word: string;
  label: string;
  knownFraction: number; // 该词的 subword 中"训练时见过"的比例(演示用)
  note: string;
}
export const OOV_EXAMPLES: OovExample[] = [
  { word: "apple", label: "常见词", knownFraction: 1.0, note: "训练时见过整词 + 所有 subword" },
  { word: "applle", label: "typo(未见过)", knownFraction: 0.82, note: "9/11 个 subword 与 apple 共享,向量近似" },
  { word: "antidisestablishmentarianism", label: "罕见词", knownFraction: 0.65, note: "anti / establish / ment 等常见 subword 在别的词里训练充分" },
];

// Word Similarity 任务对比(Bojanowski 2016 Table)
export interface SimRow {
  task: string;
  word2vec: number;
  fasttext: number;
}
export const RARE_WORD_COMPARE: SimRow[] = [
  { task: "WS353", word2vec: 70.0, fasttext: 74.0 },
  { task: "RG", word2vec: 70.0, fasttext: 77.0 },
  { task: "RW(rare)", word2vec: 50.0, fasttext: 57.0 },
  { task: "DE-Gur65", word2vec: 73.0, fasttext: 79.0 },
  { task: "DE-RW", word2vec: 44.0, fasttext: 53.0 },
];

// 形态丰富语言提升(Bojanowski 2016)
export interface MorphRow {
  lang: string;
  word2vec: number;
  fasttext: number;
  delta: number;
}
export const MORPHOLOGY_COMPARE: MorphRow[] = [
  { lang: "German", word2vec: 38.9, fasttext: 44.3, delta: 5.4 },
  { lang: "Czech", word2vec: 32.7, fasttext: 44.4, delta: 11.7 },
  { lang: "Russian", word2vec: 38.6, fasttext: 49.1, delta: 10.5 },
  { lang: "Arabic", word2vec: 24.1, fasttext: 45.2, delta: 21.1 },
  { lang: "Turkish", word2vec: 21.8, fasttext: 42.8, delta: 21.0 },
];

export const DEMO_WORDS = ["where", "apple", "applle", "evler"];
