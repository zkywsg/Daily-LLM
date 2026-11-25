// 英法翻译对齐示例(论文 Figure 3 风格)
export const ALIGNMENT_DEMO = {
  src: ["The", "agreement", "on", "the", "European", "Economic", "Area", "was", "signed", "in", "August", "1992"],
  tgt: ["L'", "accord", "sur", "la", "zone", "économique", "européenne", "a", "été", "signé", "en", "août", "1992"],
};

// 构造合理的对齐矩阵(对角线 + 局部交换体现 "European Economic Area" ↔ "zone économique européenne")
export function buildAlignmentMatrix(): number[][] {
  const T = ALIGNMENT_DEMO.tgt.length;
  const S = ALIGNMENT_DEMO.src.length;
  const M: number[][] = Array.from({ length: T }, () => Array(S).fill(0));

  // 默认沿对角线
  const pairs: Array<[number, number]> = [
    [0, 0],  // L' ↔ The
    [1, 1],  // accord ↔ agreement
    [2, 2],  // sur ↔ on
    [3, 3],  // la ↔ the
    // European Economic Area ↔ zone économique européenne (反序!)
    [4, 6],  // zone ↔ Area
    [5, 5],  // économique ↔ Economic
    [6, 4],  // européenne ↔ European
    [7, 7],  // a ↔ was
    [8, 8],  // été ↔ ...
    [9, 8],  // signé ↔ signed
    [10, 9], // en ↔ in
    [11, 10],// août ↔ August
    [12, 11],// 1992 ↔ 1992
  ];
  for (const [t, s] of pairs) {
    M[t][s] = 0.78;
    if (s > 0) M[t][s - 1] = 0.10;
    if (s < S - 1) M[t][s + 1] = 0.08;
  }
  // 归一化每行
  for (let t = 0; t < T; t++) {
    const sum = M[t].reduce((a, b) => a + b, 0);
    if (sum > 0) for (let s = 0; s < S; s++) M[t][s] /= sum;
  }
  return M;
}

// 长句 BLEU 对比 (论文 Table)
export interface LengthBleu {
  bucket: string;
  fixedC: number;
  attention: number;
}
export const LENGTH_BLEU: LengthBleu[] = [
  { bucket: "≤ 20",    fixedC: 25, attention: 28 },
  { bucket: "20–40",   fixedC: 22, attention: 27 },
  { bucket: "40–60",   fixedC: 18, attention: 26 },
  { bucket: "> 60",    fixedC: 12, attention: 24 },
];

// Bahdanau additive vs Luong dot vs Luong general 对比
export interface ScoreType {
  name: string;
  formula: string;
  cost: string;
  color: string;
  note: string;
}
export const SCORE_TYPES: ScoreType[] = [
  { name: "Bahdanau additive", formula: "vᵀ tanh(W_s s + W_h h)", cost: "MLP 一层",   color: "#ec4899", note: "需要 W_s/W_h/v 三个矩阵 · 较慢" },
  { name: "Luong dot",          formula: "sᵀ h",                   cost: "1 次点积",   color: "#f59e0b", note: "无参数 · 要求 s/h 同维度" },
  { name: "Luong general",      formula: "sᵀ W h",                  cost: "1 次矩阵乘", color: "#10b981", note: "增加表达力 · 不需同维度" },
  { name: "Transformer scaled", formula: "QKᵀ / √d_k",             cost: "全并行矩阵", color: "#3b82f6", note: "+ multi-head · 2017 最终形态" },
];
