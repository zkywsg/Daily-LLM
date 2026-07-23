// Graphormer demo 数据:复用同款 6 节点 toy 图,加最短路径 BFS +
// 中心性/空间/边编码的确定性查表函数。

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

export function degree(node: number): number {
  return rawNeighbors(node).length;
}

/** 中心性编码:按度数查表得到一个 embedding 标量(demo 用单维简化) */
const CENTRALITY_TABLE = [0, 0.2, 0.5, 0.9, 1.3, 1.6];
export function centralityEmbedding(node: number): number {
  return CENTRALITY_TABLE[Math.min(degree(node), CENTRALITY_TABLE.length - 1)];
}

/** BFS 最短路径距离,返回路径上经过的边列表(用于边编码) */
export function shortestPath(i: number, j: number): { distance: number; path: number[] } {
  if (i === j) return { distance: 0, path: [i] };
  const visited = new Set([i]);
  const queue: number[][] = [[i]];
  while (queue.length > 0) {
    const path = queue.shift()!;
    const last = path[path.length - 1];
    for (const nb of rawNeighbors(last)) {
      if (nb === j) return { distance: path.length, path: [...path, nb] };
      if (!visited.has(nb)) {
        visited.add(nb);
        queue.push([...path, nb]);
      }
    }
  }
  return { distance: Infinity, path: [] };
}

/** 空间编码 bias:距离越远,bias 越负(衰减邻居之外节点的注意力) */
const SPATIAL_BIAS_TABLE = [0, -0.1, -0.3, -0.6, -1.0];
export function spatialBias(distance: number): number {
  if (!Number.isFinite(distance)) return -2;
  return SPATIAL_BIAS_TABLE[Math.min(distance, SPATIAL_BIAS_TABLE.length - 1)];
}

/** 边编码:路径上每条边有一个固定的小 bias,累加到 spatial bias 之上 */
function edgeWeight(a: number, b: number): number {
  let h = (a + 1) * 131 + (b + 1) * 977;
  h = h >>> 0;
  return ((h % 100) / 100) * 0.3; // [0, 0.3)
}
export function edgeBias(path: number[]): number {
  let total = 0;
  for (let k = 0; k < path.length - 1; k++) total += edgeWeight(path[k], path[k + 1]);
  return total;
}

/** 基础 QK attention score(不含任何结构 bias),确定性 hash 模拟 */
export function baseScore(i: number, j: number): number {
  let h = (i + 1) * 313 + (j + 1) * 71;
  h = h >>> 0;
  return ((h % 1000) / 1000) * 2 - 1; // [-1, 1]
}
