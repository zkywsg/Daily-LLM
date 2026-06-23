// 把 Transformer 三大机制涉及的数学全塞这里 —— widget 只负责画。

/** softmax(行) */
export function softmaxRows(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const m = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  });
}

/** 矩阵转置 */
export function transpose(m: number[][]): number[][] {
  const r = m.length;
  const c = m[0]?.length ?? 0;
  const out: number[][] = Array.from({ length: c }, () => Array(r).fill(0));
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) out[j][i] = m[i][j];
  return out;
}

/** 矩阵乘 a (m×k) · b (k×n) = (m×n) */
export function matmul(a: number[][], b: number[][]): number[][] {
  const m = a.length;
  const k = a[0]?.length ?? 0;
  const n = b[0]?.length ?? 0;
  const out: number[][] = Array.from({ length: m }, () => Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += a[i][t] * b[t][j];
      out[i][j] = s;
    }
  return out;
}

/** 标量 scale */
export function scale(m: number[][], k: number): number[][] {
  return m.map((row) => row.map((v) => v * k));
}

/**
 * Scaled Dot-Product Attention 全流程:
 *   scores = Q · Kᵀ          (n × n,raw 相似度)
 *   if scaled: scores /= √dk
 *   weights = softmax(scores) (n × n,每行和为 1)
 *   out = weights · V         (n × dv)
 */
export function scaledDotProduct(
  Q: number[][],
  K: number[][],
  V: number[][],
  scaled = true,
): {
  scores: number[][];
  scaledScores: number[][];
  weights: number[][];
  out: number[][];
} {
  const dk = Q[0]?.length ?? 1;
  const scores = matmul(Q, transpose(K));
  const scaledScores = scaled ? scale(scores, 1 / Math.sqrt(dk)) : scores;
  const weights = softmaxRows(scaledScores);
  const out = matmul(weights, V);
  return { scores, scaledScores, weights, out };
}

/**
 * 原版 Transformer 的 sin/cos 位置编码:
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 * 返回 (n_pos × d_model) 矩阵。
 */
export function positionalEncoding(
  nPos: number,
  dModel: number,
): number[][] {
  const out: number[][] = Array.from({ length: nPos }, () => Array(dModel).fill(0));
  for (let pos = 0; pos < nPos; pos++) {
    for (let i = 0; i < dModel; i++) {
      const dimPair = Math.floor(i / 2);
      const angle = pos / Math.pow(10000, (2 * dimPair) / dModel);
      out[pos][i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }
  }
  return out;
}

/** 两向量余弦相似度 */
export function cosineSim(a: number[], b: number[]): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * 生成一个"看起来像真实 attention pattern"的 demo Q, K, V。
 * 用 token 文字驱动,固定种子(基于字符 code)让每次 render 一样。
 */
export function demoQKV(tokens: string[], dModel: number): {
  Q: number[][];
  K: number[][];
  V: number[][];
} {
  const seed = (s: string, salt: number) => {
    let h = salt;
    for (const c of s) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return ((h % 1000) / 1000) * 2 - 1; // [-1, 1)
  };
  const mat = (salt: number) =>
    tokens.map((t, _i) =>
      Array.from({ length: dModel }, (_, j) => seed(t, salt + j * 17)),
    );
  return { Q: mat(1), K: mat(7), V: mat(13) };
}

/**
 * Multi-Head:把 d_model 切成 h × d_k,每个 head 独立做 attention。
 * 返回每个 head 的 attention weights(n × n)。
 */
export function multiHeadWeights(
  Q: number[][],
  K: number[][],
  numHeads: number,
  scaled = true,
): number[][][] {
  const n = Q.length;
  const dModel = Q[0]?.length ?? 0;
  const dk = Math.floor(dModel / numHeads);
  const heads: number[][][] = [];
  for (let h = 0; h < numHeads; h++) {
    const Qh = Q.map((row) => row.slice(h * dk, (h + 1) * dk));
    const Kh = K.map((row) => row.slice(h * dk, (h + 1) * dk));
    const scores = matmul(Qh, transpose(Kh));
    const adj = scaled ? scale(scores, 1 / Math.sqrt(dk)) : scores;
    heads.push(softmaxRows(adj));
  }
  return heads;
}
