// GIN demo 数据:非单射反例(mean/max 相同但 sum 不同的两个多重集)+
// ε 可调的自身/邻居加权演示 + WL 染色用的 6 节点 toy 图。

/** 经典反例:两个多重集在 mean/max 下无法区分,但 sum 下不同。
 * X = {1, 1}(两个特征为 1 的邻居),Y = {1, 1, 1, 1}(四个特征为 1 的邻居)。
 * mean(X) = mean(Y) = 1,max(X) = max(Y) = 1,但 sum(X) = 2 ≠ sum(Y) = 4。 */
export const MULTISET_X = [1, 1];
export const MULTISET_Y = [1, 1, 1, 1];

export function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}
export function max(xs: number[]): number {
  return xs.length === 0 ? 0 : Math.max(...xs);
}
export function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

/** GIN 更新(简化到 MLP 前的标量组合):(1+ε)·h_self + Σ neighbors */
export function ginPreMlp(selfFeature: number, neighborSum: number, epsilon: number): number {
  return (1 + epsilon) * selfFeature + neighborSum;
}

// WL 染色用的 6 节点 toy 图(与 GCN/GAT 同款,方便认出同一张图)
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

/** 一轮 WL 颜色迭代:新颜色 = hash(自己的颜色, 排序后的邻居颜色多重集) */
export function wlRefine(colors: number[]): number[] {
  const signatures = colors.map((c, node) => {
    const neighborColors = rawNeighbors(node).map((n) => colors[n]).sort((a, b) => a - b);
    return `${c}|${neighborColors.join(",")}`;
  });
  const uniqueSigs = Array.from(new Set(signatures)).sort();
  return signatures.map((sig) => uniqueSigs.indexOf(sig));
}
