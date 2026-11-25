// 直线 vs 曲线路径上的采样点(2D toy 演示,noise x1 → data x0)
export const NOISE_POINT = { x: 560, y: 60, label: "x₁(噪声)" };
export const DATA_POINT = { x: 80, y: 260, label: "x₀(数据)" };

// DDPM 曲线路径的中间控制点(仅用于可视化"弯曲/带随机扰动"的采样轨迹)
export const CURVED_WAYPOINTS: Array<{ x: number; y: number }> = [
  { x: 560, y: 60 },
  { x: 470, y: 140 },
  { x: 500, y: 200 },
  { x: 380, y: 150 },
  { x: 330, y: 230 },
  { x: 230, y: 170 },
  { x: 190, y: 260 },
  { x: 80, y: 260 },
];

// 采样步数对比 —— 源文档:DDPM 反向 SDE 需要 50-100 步,Flow Matching ODE Euler 10-20 步即可
export interface StepsCompareRow {
  method: string;
  steps: number;
  note: string;
}
export const STEPS_COMPARE: StepsCompareRow[] = [
  { method: "DDPM(SDE 反向)", steps: 50, note: "曲线路径,需 50-100 步才稳定" },
  { method: "Flow Matching(ODE, 默认)", steps: 50, note: "也可用 50 步,质量更稳" },
  { method: "Flow Matching(ODE, 少步)", steps: 20, note: "直线路径,20 步已接近 DDPM 50 步质量" },
  { method: "Flow Matching(ODE, 极限少步)", steps: 10, note: "10 步仍可用,画质轻微下降" },
];

// CFG scale 对比 —— 源文档"训练细节"表:SD3(Flow Matching)CFG scale 4-5,比 SD 1.5(DDPM)的 7-10 低
export interface CfgScaleRow {
  model: string;
  objective: string;
  scaleMin: number;
  scaleMax: number;
}
export const CFG_SCALE_COMPARE: CfgScaleRow[] = [
  { model: "SD 1.5", objective: "DDPM(ε-prediction)", scaleMin: 7, scaleMax: 10 },
  { model: "SD3 Medium", objective: "Flow Matching(velocity-prediction)", scaleMin: 4, scaleMax: 5 },
];

// 训练目标对比表(源文档机制一表格,供 stickyPanel / 速度场 widget 参考)
export interface ObjectiveCompareRow {
  dim: string;
  ddpm: string;
  flowMatching: string;
}
export const OBJECTIVE_COMPARE: ObjectiveCompareRow[] = [
  { dim: "网络预测", ddpm: "噪声 ε", flowMatching: "速度 v = x₁ - x₀" },
  { dim: "训练目标", ddpm: "‖ε - ε_θ(x_t, t)‖²", flowMatching: "‖v - v_θ(x_t, t)‖²" },
  { dim: "x_t 构造", ddpm: "√āₜ·x₀ + √(1-āₜ)·ε(曲线)", flowMatching: "(1-t)·x₀ + t·x₁(直线)" },
  { dim: "Noise schedule", ddpm: "需要(β_t / cosine 等)", flowMatching: "不需要(t 均匀采)" },
  { dim: "采样路径", ddpm: "SDE 反向(随机)", flowMatching: "ODE 反向(确定)" },
];

// SD3 Medium 参考实现关键配置(源文档"训练细节"表)
export interface Sd3ConfigRow {
  dim: string;
  value: string;
}
export const SD3_CONFIG: Sd3ConfigRow[] = [
  { dim: "Backbone", value: "MM-DiT, 2B 参数" },
  { dim: "文本编码器", value: "T5-XXL + CLIP-L + CLIP-G" },
  { dim: "训练目标", value: "Rectified Flow(等价 Flow Matching)" },
  { dim: "Noise schedule", value: "Logit-normal(t 偏好中间值)" },
  { dim: "采样器", value: "Euler ODE(50 steps 默认,20 steps 也 work)" },
  { dim: "CFG", value: "scale 4-5(比 SD 1.5 的 7-10 低)" },
];
