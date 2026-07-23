// GraphSAGE demo 数据:8 节点图(比 GCN 的 toy 图大,便于演示"采样子集"),
// 每个节点有一个 2 维特征向量(纯 demo 数值,不代表真实语义)。

export const NODES = [0, 1, 2, 3, 4, 5, 6, 7];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 0, b: 3 }, { a: 0, b: 4 },
  { a: 0, b: 5 }, { a: 0, b: 6 }, { a: 1, b: 2 }, { a: 3, b: 7 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [340, 200], 1: [180, 80], 2: [180, 320], 3: [500, 80], 4: [500, 320],
  5: [220, 200], 6: [460, 200], 7: [640, 80],
};

export const NODE_FEATURES: Record<number, [number, number]> = {
  0: [0.5, 0.5], 1: [0.9, 0.1], 2: [0.1, 0.9], 3: [0.8, 0.8],
  4: [0.2, 0.3], 5: [0.6, 0.2], 6: [0.3, 0.7], 7: [0.95, 0.9],
};

export function fullNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}

/** 确定性"随机"采样:用简单 hash 决定选哪 k 个邻居,同一 seed 结果稳定 */
export function sampleNeighbors(node: number, k: number, seed: number): number[] {
  const all = fullNeighbors(node);
  if (all.length <= k) return all;
  const scored = all.map((n) => {
    let h = seed;
    h = (h * 131 + n * 977) >>> 0;
    return { n, score: h % 1000 };
  });
  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, k).map((s) => s.n).sort((a, b) => a - b);
}

export function meanAggregate(vectors: Array<[number, number]>): [number, number] {
  if (vectors.length === 0) return [0, 0];
  const sx = vectors.reduce((s, v) => s + v[0], 0);
  const sy = vectors.reduce((s, v) => s + v[1], 0);
  return [sx / vectors.length, sy / vectors.length];
}

export function maxPoolAggregate(vectors: Array<[number, number]>): [number, number] {
  if (vectors.length === 0) return [0, 0];
  return [Math.max(...vectors.map((v) => v[0])), Math.max(...vectors.map((v) => v[1]))];
}

/** 简化版"顺序敏感"聚合(代表 LSTM 聚合器的关键性质):
 * 越靠前的邻居权重越高,所以打乱输入顺序会改变结果 —— 这正是 mean/max 不具备的性质。 */
export function orderSensitiveAggregate(vectors: Array<[number, number]>): [number, number] {
  if (vectors.length === 0) return [0, 0];
  let wsum = 0;
  let sx = 0;
  let sy = 0;
  vectors.forEach((v, idx) => {
    const w = 1 / (idx + 1);
    sx += v[0] * w;
    sy += v[1] * w;
    wsum += w;
  });
  return [sx / wsum, sy / wsum];
}

/** 图 B:一个训练时不存在的新节点(用于演示归纳式泛化) */
export const GRAPH_B_NEW_NODE = 100;
export const GRAPH_B_NEW_NODE_FEATURE: [number, number] = [0.4, 0.6];
export const GRAPH_B_NEW_NODE_NEIGHBORS = [201, 202, 203];
export const GRAPH_B_NEIGHBOR_FEATURES: Record<number, [number, number]> = {
  201: [0.7, 0.3], 202: [0.5, 0.5], 203: [0.2, 0.8],
};
