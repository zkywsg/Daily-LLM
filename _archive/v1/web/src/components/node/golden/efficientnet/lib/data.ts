// 单轴缩放 vs 复合缩放:示意曲线(还原论文 Figure 5 的定性形状 ——
// 只加深 / 只加宽 / 只加分辨率均较快出现边际收益递减,三轴联合缩放的帕累托线明显更高)。
// 数据点为示意性还原源文档描述的"饱和"形状,非论文原始逐点数值。
export interface ScalingCurvePoint {
  flops: number; // 相对 FLOPs(以 B0=1 为基准)
  top1: number; // Top-1 准确率(%)
}

export interface ScalingCurve {
  key: "depth" | "width" | "resolution" | "compound";
  label: string;
  color: string;
  bg: string;
  points: ScalingCurvePoint[];
}

export const SINGLE_VS_COMPOUND_SCALING: ScalingCurve[] = [
  {
    key: "depth",
    label: "只加深(depth only)",
    color: "#3b82f6",
    bg: "#dbeafe",
    points: [
      { flops: 1, top1: 77.1 },
      { flops: 2, top1: 78.6 },
      { flops: 4, top1: 79.6 },
      { flops: 8, top1: 80.1 },
      { flops: 16, top1: 80.3 },
    ],
  },
  {
    key: "width",
    label: "只加宽(width only)",
    color: "#9ca3af",
    bg: "#f3f4f6",
    points: [
      { flops: 1, top1: 77.1 },
      { flops: 2, top1: 78.3 },
      { flops: 4, top1: 79.1 },
      { flops: 8, top1: 79.5 },
      { flops: 16, top1: 79.6 },
    ],
  },
  {
    key: "resolution",
    label: "只加分辨率(resolution only)",
    color: "#f59e0b",
    bg: "#fef3c7",
    points: [
      { flops: 1, top1: 77.1 },
      { flops: 2, top1: 78.0 },
      { flops: 4, top1: 78.8 },
      { flops: 8, top1: 79.2 },
      { flops: 16, top1: 79.4 },
    ],
  },
  {
    key: "compound",
    label: "三轴等比联合缩放(compound)",
    color: "#ec4899",
    bg: "#fce7f3",
    points: [
      { flops: 1, top1: 77.1 },
      { flops: 2, top1: 79.1 },
      { flops: 4, top1: 81.1 },
      { flops: 8, top1: 82.6 },
      { flops: 16, top1: 83.6 },
    ],
  },
];

// Compound Coefficient 公式常数(B0 上 grid search 得到)
export const COMPOUND_CONSTANTS = {
  alpha: 1.2, // depth
  beta: 1.1, // width
  gamma: 1.15, // resolution
  constraint: "α·β²·γ² ≈ 2",
};

// φ = 0..7 时三轴相对 B0 的放大倍数(depth=α^φ, width=β^φ, resolution=γ^φ)
export interface PhiStep {
  phi: number;
  variant: string;
  depthMul: number;
  widthMul: number;
  resMul: number;
}

export const PHI_STEPS: PhiStep[] = Array.from({ length: 8 }, (_, phi) => ({
  phi,
  variant: `B${phi}`,
  depthMul: Number((COMPOUND_CONSTANTS.alpha ** phi).toFixed(2)),
  widthMul: Number((COMPOUND_CONSTANTS.beta ** phi).toFixed(2)),
  resMul: Number((COMPOUND_CONSTANTS.gamma ** phi).toFixed(2)),
}));

// EfficientNet-B0 的 NAS 搜索结果:7 个 MBConv stage
export interface MBConvStageRow {
  stage: number;
  block: string;
  kernel: string;
  channels: number;
  numBlocks: number;
  stride: number;
}

export const B0_STAGES: MBConvStageRow[] = [
  { stage: 1, block: "MBConv1", kernel: "3×3", channels: 16, numBlocks: 1, stride: 1 },
  { stage: 2, block: "MBConv6", kernel: "3×3", channels: 24, numBlocks: 2, stride: 2 },
  { stage: 3, block: "MBConv6", kernel: "5×5", channels: 40, numBlocks: 2, stride: 2 },
  { stage: 4, block: "MBConv6", kernel: "3×3", channels: 80, numBlocks: 3, stride: 2 },
  { stage: 5, block: "MBConv6", kernel: "5×5", channels: 112, numBlocks: 3, stride: 1 },
  { stage: 6, block: "MBConv6", kernel: "5×5", channels: 192, numBlocks: 4, stride: 2 },
  { stage: 7, block: "MBConv6", kernel: "3×3", channels: 320, numBlocks: 1, stride: 1 },
];

// MBConv block 内部 5 个步骤(用于结构图)
export interface MBConvStep {
  label: string;
  detail: string;
}

export const MBCONV_STEPS: MBConvStep[] = [
  { label: "1×1 升维 expand×6", detail: "Conv1×1 → BN → Swish,通道 ×6(inverted bottleneck)" },
  { label: "Depthwise k×k", detail: "groups=mid_c 的深度可分离卷积 → BN → Swish" },
  { label: "SE 门控", detail: "全局池化 → 压到 1/4 → 升回 → sigmoid,通道注意力" },
  { label: "1×1 降维 project", detail: "Conv1×1 → BN,线性瓶颈(project 后无激活)" },
  { label: "+ shortcut", detail: "仅同 shape(stride=1 且 in_c=out_c)时相加" },
];

// B0-B7 模型族(源文档表格中给出的真实数据点:B0/B3/B5/B7)+ 对比基线
export interface ModelFamilyRow {
  model: string;
  params: number; // M
  flops: number; // B (GFLOPs)
  top1: number;
  dropout: number;
  stochasticDepth: number;
  isEfficientNet: boolean;
  highlight?: boolean;
}

export const MODEL_FAMILY: ModelFamilyRow[] = [
  { model: "ResNet-50", params: 25.6, flops: 4.1, top1: 76.0, dropout: 0, stochasticDepth: 0, isEfficientNet: false },
  { model: "ResNet-152", params: 60.2, flops: 11.5, top1: 78.3, dropout: 0, stochasticDepth: 0, isEfficientNet: false },
  { model: "ResNeXt-101 (64×4d)", params: 84, flops: 31.5, top1: 80.9, dropout: 0, stochasticDepth: 0, isEfficientNet: false },
  { model: "GPipe", params: 557, flops: 0, top1: 84.3, dropout: 0, stochasticDepth: 0, isEfficientNet: false },
  { model: "EfficientNet-B0", params: 5.3, flops: 0.39, top1: 77.1, dropout: 0.2, stochasticDepth: 0.0, isEfficientNet: true },
  { model: "EfficientNet-B3", params: 12, flops: 1.8, top1: 81.6, dropout: 0.3, stochasticDepth: 0.1, isEfficientNet: true },
  { model: "EfficientNet-B5", params: 30, flops: 9.9, top1: 83.6, dropout: 0.4, stochasticDepth: 0.2, isEfficientNet: true },
  { model: "EfficientNet-B7", params: 66, flops: 37, top1: 84.3, dropout: 0.5, stochasticDepth: 0.2, isEfficientNet: true, highlight: true },
];

// EfficientNet B0→B7 的正则强度 schedule(源文档明确给出 B0/B4/B7 三个锚点,
// 中间按"随规模线性涨"线性插值,标注为 interpolated)
export interface RegularizationStep {
  variant: string;
  dropout: number;
  stochasticDepth: number;
  anchor: boolean; // 是否为源文档明确给出的锚点(B0/B4/B7),其余为线性插值
}

export const REGULARIZATION_SCHEDULE: RegularizationStep[] = [
  { variant: "B0", dropout: 0.2, stochasticDepth: 0.0, anchor: true },
  { variant: "B1", dropout: 0.24, stochasticDepth: 0.05, anchor: false },
  { variant: "B2", dropout: 0.28, stochasticDepth: 0.1, anchor: false },
  { variant: "B3", dropout: 0.33, stochasticDepth: 0.15, anchor: false },
  { variant: "B4", dropout: 0.37, stochasticDepth: 0.2, anchor: true },
  { variant: "B5", dropout: 0.41, stochasticDepth: 0.2, anchor: false },
  { variant: "B6", dropout: 0.46, stochasticDepth: 0.2, anchor: false },
  { variant: "B7", dropout: 0.5, stochasticDepth: 0.2, anchor: true },
];
