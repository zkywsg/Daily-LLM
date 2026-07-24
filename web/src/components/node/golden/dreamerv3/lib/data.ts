// DreamerV3 demo 数据:离散类别隐变量(RSSM)采样可视化 +
// symlog 归一化跨量级 reward 演示 + 纯想象轨迹 rollout。

/** 用确定性 hash 给 numCategoricals 个类别分布(每个 numClasses 类)生成概率, 模拟 RSSM 离散隐状态 */
export function categoricalLatent(numCategoricals: number, numClasses: number, seed = 0): number[][] {
  const dists: number[][] = [];
  for (let c = 0; c < numCategoricals; c++) {
    const logits: number[] = [];
    for (let k = 0; k < numClasses; k++) {
      let h = (seed + c * 977 + k * 31) >>> 0;
      h = (h * 2654435761) >>> 0;
      logits.push(((h % 1000) / 1000) * 4 - 2);
    }
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const s = exps.reduce((a, b) => a + b, 0);
    dists.push(exps.map((e) => e / s));
  }
  return dists;
}

/** symlog(x) = sign(x) * log(1 + |x|) —— 把跨越多个数量级的数值压缩到可比范围 */
export function symlog(x: number): number {
  return Math.sign(x) * Math.log(1 + Math.abs(x));
}

/** symlog 的反函数,用于从压缩空间还原 */
export function symexp(x: number): number {
  return Math.sign(x) * (Math.exp(Math.abs(x)) - 1);
}

export const DOMAIN_REWARDS: Array<{ domain: string; raw: number }> = [
  { domain: "Atari(单步得分)", raw: 1 },
  { domain: "DMC(连续控制)", raw: 12 },
  { domain: "Minecraft(采集里程碑)", raw: 500 },
  { domain: "稀疏大额奖励", raw: 8000 },
];

/** 纯想象(imagination)轨迹:actor-critic 只在这条轨迹上训练,不接触真实环境 */
export function imaginationRollout(steps: number): Array<{ x: number; y: number }> {
  const traj: Array<{ x: number; y: number }> = [{ x: 0, y: 0 }];
  for (let t = 1; t <= steps; t++) {
    let h = (t * 2654435761) >>> 0;
    const angle = ((h % 1000) / 1000) * Math.PI * 2;
    const prev = traj[traj.length - 1];
    traj.push({ x: prev.x + Math.cos(angle) * 0.6, y: prev.y + Math.sin(angle) * 0.6 });
  }
  return traj;
}
