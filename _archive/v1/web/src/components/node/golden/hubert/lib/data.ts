// HuBERT demo 数据:toy 特征点 k-means 聚类 + 掩码分类置信度分布 +
// 迭代式重新聚类(Lloyd's 算法一步更新)。全部确定性构造。

export const NUM_POINTS = 8;
export const TOY_POINTS: Array<[number, number]> = Array.from({ length: NUM_POINTS }, (_, i) => {
  const angle = (i / NUM_POINTS) * 2 * Math.PI;
  const r = 0.5 + (i % 3) * 0.25;
  return [Math.cos(angle) * r, Math.sin(angle) * r];
});

export const NUM_CLUSTERS = 3;

export function initialCenters(): Array<[number, number]> {
  return Array.from({ length: NUM_CLUSTERS }, (_, k) => {
    const angle = (k / NUM_CLUSTERS) * 2 * Math.PI;
    return [Math.cos(angle) * 0.3, Math.sin(angle) * 0.3];
  });
}

function dist2(a: [number, number], b: [number, number]): number {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
}

export function assignClusters(points: Array<[number, number]>, centers: Array<[number, number]>): number[] {
  return points.map((p) => {
    let best = 0, bestD = Infinity;
    centers.forEach((c, k) => {
      const d = dist2(p, c);
      if (d < bestD) { bestD = d; best = k; }
    });
    return best;
  });
}

/** 一步 Lloyd's 算法更新:用当前分配重新计算聚类中心(各聚类内点的均值) */
export function updateCenters(points: Array<[number, number]>, assignments: number[]): Array<[number, number]> {
  const sums: Array<[number, number, number]> = Array.from({ length: NUM_CLUSTERS }, () => [0, 0, 0]);
  points.forEach((p, i) => {
    const k = assignments[i];
    sums[k][0] += p[0];
    sums[k][1] += p[1];
    sums[k][2] += 1;
  });
  const fallback = initialCenters();
  return sums.map(([sx, sy, n], k) => (n > 0 ? [sx / n, sy / n] as [number, number] : fallback[k]));
}

/** 给定"真实类别"和置信度(sharpness),softmax 出一个类别分布 —— sharpness 越大分布越尖锐,
 * 但 trueClass 的 logit 恒定比其余类别高,所以任意 sharpness 下 trueClass 概率都是最高的。 */
export function classDistribution(trueClass: number, sharpness: number): number[] {
  const logits = Array.from({ length: NUM_CLUSTERS }, (_, k) => (k === trueClass ? sharpness : -sharpness / 2));
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
