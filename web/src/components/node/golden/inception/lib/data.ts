// Inception block 的 4 条并行分支 —— 输入并行走 1×1 / 1×1→3×3 / 1×1→5×5 / pool→1×1,
// 最后在通道维 concat。
export interface InceptionBranch {
  id: string;
  label: string;
  steps: string[]; // 分支内部按顺序执行的算子
  note: string;
  kind: "conv1x1" | "conv3x3" | "conv5x5" | "pool";
}

export const INCEPTION_BRANCHES: InceptionBranch[] = [
  {
    id: "b1",
    label: "分支 1 · 纯 1×1",
    steps: ["1×1 conv"],
    note: "看像素本身的通道组合",
    kind: "conv1x1",
  },
  {
    id: "b2",
    label: "分支 2 · 1×1 → 3×3",
    steps: ["1×1 conv(降维)", "3×3 conv"],
    note: "1×1 先把通道砍下再算 3×3,看小邻域",
    kind: "conv3x3",
  },
  {
    id: "b3",
    label: "分支 3 · 1×1 → 5×5",
    steps: ["1×1 conv(降维)", "5×5 conv"],
    note: "1×1 先把通道砍下再算 5×5,看更大邻域",
    kind: "conv5x5",
  },
  {
    id: "b4",
    label: "分支 4 · pool → 1×1",
    steps: ["3×3 MaxPool", "1×1 conv(降维)"],
    note: "MaxPool 不改通道,1×1 在 pool 之后压通道,提供位置不变性",
    kind: "pool",
  },
];

// 1×1 卷积瓶颈的参数对比 —— 直接算 5×5 vs 先用 1×1 压到 64 再算 5×5。
// 数字来自源 markdown 机制二:输入 256 通道、每分支输出 128 通道场景下的 5×5 分支示例。
export interface BottleneckCompareRow {
  label: string;
  params: number; // 参数量(个)
  highlight?: boolean;
  formula: string;
}

export const BOTTLENECK_COMPARE: BottleneckCompareRow[] = [
  {
    label: "直接 5×5(512 → 128 通道)",
    params: 1_600_000,
    formula: "5×5×512×128",
  },
  {
    label: "1×1 先压到 64 再算 5×5",
    params: 200_000,
    highlight: true,
    formula: "5×5×64×128",
  },
];

// GoogLeNet vs AlexNet / VGG-16 的整网参数量对比,以及对应 ImageNet Top-5 错误率。
export interface ModelParamRow {
  label: string;
  year: number;
  params: number; // 参数量(个)
  top5Error: number; // Top-5 错误率(%)
  highlight?: boolean;
}

export const MODEL_PARAM_COMPARE: ModelParamRow[] = [
  { label: "AlexNet", year: 2012, params: 60_000_000, top5Error: 15.3 },
  { label: "ZFNet", year: 2013, params: 60_000_000, top5Error: 14.8 },
  { label: "VGG-16", year: 2014, params: 138_000_000, top5Error: 7.3 },
  { label: "GoogLeNet", year: 2014, params: 5_000_000, top5Error: 7.89, highlight: true },
];

// VGG fc6 单层 vs GoogLeNet GAP+FC 的参数对比 —— GAP 替代大 FC 干掉参数尾巴的核心证据。
export interface HeadParamRow {
  label: string;
  params: number;
  shareOfTotal: string; // 占整网参数比例的文字描述
  highlight?: boolean;
}

export const HEAD_PARAM_COMPARE: HeadParamRow[] = [
  { label: "VGG-16 · fc6 单层(4096 维 FC)", params: 102_000_000, shareOfTotal: "占整网 138M 的 74%" },
  { label: "GoogLeNet · GAP + 单层 FC", params: 1_000_000, shareOfTotal: "占整网 5M 的 ~20%", highlight: true },
];

// GoogLeNet 整网结构:stem + 9 个 Inception block + GAP + FC,2 个 aux head 挂在 4a/4d。
export interface GoogLeNetStageItem {
  label: string;
  kind: "stem" | "inception" | "aux" | "pool" | "output";
  note?: string;
}

export const GOOGLENET_STAGES: GoogLeNetStageItem[] = [
  { label: "stem", kind: "stem", note: "conv + maxpool ×2" },
  { label: "3a", kind: "inception" },
  { label: "3b", kind: "inception" },
  { label: "maxpool", kind: "pool" },
  { label: "4a", kind: "inception", note: "aux head 1 挂在此处" },
  { label: "4b", kind: "inception" },
  { label: "4c", kind: "inception" },
  { label: "4d", kind: "inception", note: "aux head 2 挂在此处" },
  { label: "4e", kind: "inception" },
  { label: "maxpool", kind: "pool" },
  { label: "5a", kind: "inception" },
  { label: "5b", kind: "inception" },
  { label: "GAP + FC", kind: "output" },
];

export const IMAGENET_RESULTS = {
  googleNetSingleTop5: 7.89,
  googleNetEnsembleTop5: 6.67,
  vgg16Top5: 7.3,
  paramReductionVsVgg: 28, // GoogLeNet 比 VGG-16 少 28 倍参数
  paramReductionVsAlexNet: 12, // GoogLeNet 比 AlexNet 少 12 倍参数
};
