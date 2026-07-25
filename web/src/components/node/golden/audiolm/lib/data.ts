// AudioLM demo 数据:语义 token 提取(帧→离散 token 哈希)+ 残差向量量化(RVQ)+
// 三阶段级联生成状态机。全部确定性构造。

export const SEMANTIC_GRID = 6;

/** 用简单哈希把一个 toy "帧"(用位置参数化)映射成语义 token id(0-31) */
export function extractSemanticToken(framePos: number): number {
  let h = Math.round(framePos * 1000);
  h = (h * 2654435761) >>> 0;
  return h % 32;
}

export const RVQ_LEVELS = 4;
export const CODES_PER_LEVEL = 9;

/** 第 level 层码本的第 idx 个条目取值:范围随层数指数缩小(逐层捕捉更细的残差)。
 * CODES_PER_LEVEL 取奇数(9),使 idx=(CODES_PER_LEVEL-1)/2=4 处的条目恒为 0,
 * 即每一层码本都包含 0 这个选项 —— 这保证最近邻量化后残差的绝对值不会超过量化前
 * (因为 0 总是候选之一,最近邻不会选一个比 0 更远的点),从而重建误差逐层单调不增。 */
function levelCodebookEntry(level: number, idx: number): number {
  const range = 1 / Math.pow(2, level);
  return (idx / (CODES_PER_LEVEL - 1) - 0.5) * 2 * range;
}

/** 残差向量量化:逐层贪心找当前残差最接近的码本条目,累加进重建值,残差递减。
 * 层的取值范围按 2 的幂缩小,足以覆盖上一层最坏情况下的残差(设计上保证收敛)。 */
export function residualQuantize(x: number): { codes: number[]; reconstruction: number } {
  let residual = x;
  const codes: number[] = [];
  let recon = 0;
  for (let level = 0; level < RVQ_LEVELS; level++) {
    let bestIdx = 0, bestDist = Infinity;
    for (let idx = 0; idx < CODES_PER_LEVEL; idx++) {
      const d = Math.abs(residual - levelCodebookEntry(level, idx));
      if (d < bestDist) { bestDist = d; bestIdx = idx; }
    }
    codes.push(bestIdx);
    const chosen = levelCodebookEntry(level, bestIdx);
    recon += chosen;
    residual -= chosen;
  }
  return { codes, reconstruction: recon };
}

/** 只用前 numLevels 层码本做部分重建(codes 数组的前 numLevels 项在完整量化下就已确定,
 * 与后续层无关,所以可以直接截断使用) */
export function partialReconstruction(codes: number[], numLevels: number): number {
  let recon = 0;
  for (let level = 0; level < numLevels; level++) recon += levelCodebookEntry(level, codes[level]);
  return recon;
}

export const TOY_SEMANTIC_VALUE = 0.37;

export const CASCADE_STAGES: Array<{ label: string; detail: string }> = [
  { label: "阶段一:生成语义 token", detail: "自回归生成语义 token 序列,决定内容和说话人是谁" },
  { label: "阶段二:生成粗声学 token", detail: "以语义 token 为条件,生成 RVQ 前几层粗粒度声学 token" },
  { label: "阶段三:生成细声学 token", detail: "以前两阶段为条件,生成 RVQ 剩余层的精细声学 token" },
];
