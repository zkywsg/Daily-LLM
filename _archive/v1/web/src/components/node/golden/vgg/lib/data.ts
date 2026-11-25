// 小卷积堆叠 vs 大卷积:等价感受野下的参数量对比(设输入输出通道数相同,记作 C)
// 数值取自 01-cnn/03-vgg.md 机制一小节的表格:5×5 感受野降 28%,7×7 感受野降 45%。
export interface KernelCompareRow {
  receptiveField: string; // 等价感受野,如 "5×5"
  approach: string; // 实现方式
  paramsFormula: string; // 参数量公式(以 C^2 为单位)
  paramsCoeff: number; // C^2 前的系数,用于画图
  isStacked: boolean;
  savingsLabel?: string; // 相对大卷积的参数节省百分比
}

export const KERNEL_COMPARISON: KernelCompareRow[] = [
  { receptiveField: "5×5", approach: "1 层 5×5 conv", paramsFormula: "25C²", paramsCoeff: 25, isStacked: false },
  {
    receptiveField: "5×5",
    approach: "2 层 3×3 conv",
    paramsFormula: "18C²",
    paramsCoeff: 18,
    isStacked: true,
    savingsLabel: "−28%",
  },
  { receptiveField: "7×7", approach: "1 层 7×7 conv", paramsFormula: "49C²", paramsCoeff: 49, isStacked: false },
  {
    receptiveField: "7×7",
    approach: "3 层 3×3 conv",
    paramsFormula: "27C²",
    paramsCoeff: 27,
    isStacked: true,
    savingsLabel: "−45%",
  },
];

// VGG-16 的 5 个 conv block 配置(通道数 + block 内卷积重复次数),取自关键代码 VGG16_CFG
export interface VggBlock {
  block: number;
  channels: number;
  convRepeats: number;
  spatialAfterPool: string; // 经过该 block 末尾 MaxPool/2 后的空间尺寸(输入 224×224)
}

export const VGG16_BLOCKS: VggBlock[] = [
  { block: 1, channels: 64, convRepeats: 2, spatialAfterPool: "112×112" },
  { block: 2, channels: 128, convRepeats: 2, spatialAfterPool: "56×56" },
  { block: 3, channels: 256, convRepeats: 3, spatialAfterPool: "28×28" },
  { block: 4, channels: 512, convRepeats: 3, spatialAfterPool: "14×14" },
  { block: 5, channels: 512, convRepeats: 3, spatialAfterPool: "7×7" },
];

// VGG-11 → VGG-19 深度演化:conv 层数由 block 内重复次数累加而来,
// 均为 3 层 fc(fc6/fc7/fc8),总层数 = conv 层数 + 3。命名本身即由此而来。
export interface VggDepthStep {
  name: string;
  convLayers: number;
  fcLayers: number;
  totalLayers: number;
}

export const VGG_DEPTH_PROGRESSION: VggDepthStep[] = [
  { name: "VGG-11", convLayers: 8, fcLayers: 3, totalLayers: 11 },
  { name: "VGG-13", convLayers: 10, fcLayers: 3, totalLayers: 13 },
  { name: "VGG-16", convLayers: 13, fcLayers: 3, totalLayers: 16 },
  { name: "VGG-19", convLayers: 16, fcLayers: 3, totalLayers: 19 },
];

// ImageNet Top-5 错误率演化,取自 01-cnn/03-vgg.md「训练细节」小节的表格
export interface ErrorRateRow {
  year: number;
  method: string;
  top5Error: number; // 百分比
  highlight?: boolean;
}

export const IMAGENET_ERROR_HISTORY: ErrorRateRow[] = [
  { year: 2012, method: "AlexNet", top5Error: 15.3 },
  { year: 2013, method: "ZFNet", top5Error: 14.8 },
  { year: 2014, method: "VGG-16 single model", top5Error: 8.1, highlight: true },
  { year: 2014, method: "VGG ensemble", top5Error: 7.3, highlight: true },
  { year: 2014, method: "GoogLeNet(冠军)", top5Error: 6.7 },
];

// 138M 参数分布,取自 图2 说明:fc6 占 74%(102M),fc7/fc8 占 15%,所有 conv 合计仅 11%(14.7M)
export interface ParamShareRow {
  label: string;
  params: number; // 百万参数
  share: number; // 占比 %
  highlight?: boolean;
}

export const VGG16_PARAM_DISTRIBUTION: ParamShareRow[] = [
  { label: "fc6(25088→4096)", params: 102, share: 74, highlight: true },
  { label: "fc7 + fc8", params: 21, share: 15 },
  { label: "所有 conv 层(13 层合计)", params: 14.7, share: 11 },
];

export const VGG16_TOTAL_PARAMS_M = 138;
