// Wav2Vec 2.0 demo 数据:toy 波形下采样 + 量化码本 + 对比学习分数。
// 全部确定性构造,不真跑模型。

export const WAVEFORM_LEN = 64;
export const TOY_WAVEFORM: number[] = Array.from({ length: WAVEFORM_LEN }, (_, i) =>
  Math.sin(i * 0.4) * 0.5 + Math.sin(i * 0.09) * 0.3
);

/** 简化下采样:每层做 stride=2 的相邻平均池化,模拟 CNN 特征编码器逐层压缩帧率 */
export function downsample(waveform: number[], numLayers: number): number[] {
  let cur = waveform;
  for (let l = 0; l < numLayers; l++) {
    const next: number[] = [];
    for (let i = 0; i + 1 < cur.length; i += 2) next.push((cur[i] + cur[i + 1]) / 2);
    cur = next;
  }
  return cur;
}

export const CODEBOOK_SIZE = 6;

/** 确定性码本向量(单位圆上均匀分布,2 维便于画在平面上) */
export function codebookVector(idx: number): [number, number] {
  const angle = (idx / CODEBOOK_SIZE) * 2 * Math.PI;
  return [Math.cos(angle), Math.sin(angle)];
}

/** 给定"真实"码本索引,生成一个带小扰动的连续特征 z(扰动幅度远小于码本间距,
 * 保证对比学习任务里真实目标始终是相似度最高的那个,这是本 demo 的设计前提)。 */
export function frameToContinuousFeature(trueCodeIdx: number): [number, number] {
  const [cx, cy] = codebookVector(trueCodeIdx);
  const jitter = (((trueCodeIdx * 977) % 100) / 1000) - 0.05; // [-0.05, 0.05)
  return [cx + jitter, cy - jitter];
}

/** 对比学习:z 与全部码本向量的余弦相似度,softmax(带温度)得到"选中概率" */
export function contrastiveScores(z: [number, number]): number[] {
  const sims = Array.from({ length: CODEBOOK_SIZE }, (_, i) => {
    const [cx, cy] = codebookVector(i);
    const dot = z[0] * cx + z[1] * cy;
    const normZ = Math.sqrt(z[0] ** 2 + z[1] ** 2) || 1e-6;
    const normC = Math.sqrt(cx ** 2 + cy ** 2) || 1e-6;
    return dot / (normZ * normC);
  });
  const m = Math.max(...sims);
  const exps = sims.map((s) => Math.exp((s - m) * 5)); // 温度缩放让分布更尖锐,便于可视化
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
