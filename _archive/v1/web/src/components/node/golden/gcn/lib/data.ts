// GCN demo 数据:6 节点 toy 图 + 自环/归一化相关的纯函数。
// 不真跑训练,所有数值都是确定性计算。

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

/** 度数,可选是否计入自环(Ã = A + I 会让每个节点度数 +1) */
export function degree(node: number, withSelfLoop: boolean): number {
  return rawNeighbors(node).length + (withSelfLoop ? 1 : 0);
}

/** 邻接矩阵一格的值(1 = 有边,withSelfLoop 时对角线也是 1) */
export function adjacencyCell(i: number, j: number, withSelfLoop: boolean): number {
  if (i === j) return withSelfLoop ? 1 : 0;
  return rawNeighbors(i).includes(j) ? 1 : 0;
}

/** 未归一化的求和聚合权重:边存在就是 1 */
export function rawWeight(i: number, j: number): number {
  return rawNeighbors(i).includes(j) ? 1 : 0;
}

/** D̃^(-1/2) Ã D̃^(-1/2) 对称归一化权重 */
export function normWeight(i: number, j: number, withSelfLoop: boolean): number {
  if (adjacencyCell(i, j, withSelfLoop) === 0) return 0;
  const di = degree(i, withSelfLoop);
  const dj = degree(j, withSelfLoop);
  return 1 / Math.sqrt(di * dj);
}

/** k-hop 内可达的节点集合(含自身),用于展示感受野随层数扩大 */
export function reachableWithinHops(center: number, hops: number): Set<number> {
  let frontier = new Set([center]);
  const visited = new Set([center]);
  for (let h = 0; h < hops; h++) {
    const next = new Set<number>();
    for (const n of frontier) {
      for (const nb of rawNeighbors(n)) {
        if (!visited.has(nb)) { next.add(nb); visited.add(nb); }
      }
    }
    frontier = next;
  }
  return visited;
}
