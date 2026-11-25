// GAT demo 数据:复用 GCN 同款 6 节点 toy 图(方便 viewer 认出同一张图在
// 演示不同机制),额外加确定性的"注意力 logits"计算函数。

export const NODES = [0, 1, 2, 3, 4, 5];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 1, b: 2 },
  { a: 1, b: 3 }, { a: 3, b: 4 }, { a: 3, b: 5 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [120, 200], 1: [260, 110], 2: [260, 290], 3: [420, 200], 4: [560, 110], 5: [560, 290],
};

export function rawNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}

/** 用简单 hash 模拟"共享前馈网络"算出的原始 attention logit(未 softmax) */
function baseLogit(i: number, j: number, headSeed: number): number {
  let h = headSeed;
  h = (h * 131 + (i + 1) * 977 + (j + 1) * 331) >>> 0;
  return ((h % 1000) / 1000) * 3 - 1; // 映射到 [-1, 2]
}

/** temperature 越大分布越尖锐(除以 temperature 再 softmax 的常见写法反过来:
 * 这里 temperature 越大表示"越敏感/越锐化",故直接乘 temperature */
export function attentionLogit(i: number, j: number, temperature: number, headSeed = 0): number {
  return baseLogit(i, j, headSeed) * temperature;
}

export function softmax(xs: number[]): number[] {
  if (xs.length === 0) return [];
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** 4 个头的 headSeed,保证每个头结果不同但确定性可复现 */
export const HEAD_SEEDS = [0, 17, 42, 99];
