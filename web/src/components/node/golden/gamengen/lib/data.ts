// GameNGen demo 数据:RL agent 自动生成的轨迹 + 条件 diffusion 预测下一帧 +
// 噪声增强 vs 不增强时的长程自回归漂移对比。全部确定性构造。

export interface TrajectoryStep {
  x: number;
  y: number;
  action: string;
}

/** RL agent 自我博弈生成的一段训练轨迹(确定性,模拟"自动生成而非人类录屏") */
export function generateRlTrajectory(steps: number): TrajectoryStep[] {
  const actions = ["前进", "左转", "右转", "开火"];
  const traj: TrajectoryStep[] = [];
  let x = 0, y = 0;
  for (let t = 0; t < steps; t++) {
    let h = (t * 2654435761) >>> 0;
    const actionIdx = h % actions.length;
    const angle = ((h >>> 8) % 360) * (Math.PI / 180);
    x += Math.cos(angle) * 0.5;
    y += Math.sin(angle) * 0.5;
    traj.push({ x, y, action: actions[actionIdx] });
  }
  return traj;
}

/** 给定历史帧质量分(0-1)和是否加噪声增强,模拟自回归 N 步后的画面质量衰减曲线。
 * 不加噪声增强:训练时从未见过"自己生成的略有瑕疵的帧"作为条件,推理时误差逐步放大(漂移);
 * 加噪声增强:训练时人为在条件帧上加噪声,模型学会了"即使条件帧不完美也能修正",漂移显著变慢。 */
export function driftCurve(withNoiseAugmentation: boolean, steps: number): number[] {
  const curve: number[] = [];
  for (let t = 0; t <= steps; t++) {
    const decayRate = withNoiseAugmentation ? 0.008 : 0.035;
    curve.push(Math.max(0.1, 1 - decayRate * t - (withNoiseAugmentation ? 0 : 0.0006 * t * t)));
  }
  return curve;
}
