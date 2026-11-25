/** DenseNet 结构化数据 —— 供 widgets 渲染用,数值均来自 01-cnn/06-densenet.md 正文 */

/** Dense block 内单层的连接元数据:第 ℓ 层接收前面 0..ℓ-1 所有层的 concat */
export interface DenseLayerNode {
  index: number; // 层号 ℓ(0 = block 输入 x0)
  label: string;
  channels: number; // 该层自身贡献的通道数(growth rate k),block 输入为 k0
}

export const GROWTH_RATE_K = 32;
export const BLOCK_INPUT_K0 = 64;
export const DENSE_BLOCK_LAYERS = 5; // 示意用的小型 dense block 层数(图 1 用 5 层)

/** 5 层示意 dense block 的连接列表,index 0 为 block 输入 x0 */
export const DENSE_BLOCK_NODES: DenseLayerNode[] = [
  { index: 0, label: "x₀ (输入)", channels: BLOCK_INPUT_K0 },
  { index: 1, label: "H₁", channels: GROWTH_RATE_K },
  { index: 2, label: "H₂", channels: GROWTH_RATE_K },
  { index: 3, label: "H₃", channels: GROWTH_RATE_K },
  { index: 4, label: "H₄", channels: GROWTH_RATE_K },
  { index: 5, label: "H₅", channels: GROWTH_RATE_K },
];

/** 累积输入通道数:第 ℓ 层输入 = k0 + (ℓ-1)*k */
export function cumulativeChannels(layerIndex: number): number {
  if (layerIndex <= 0) return BLOCK_INPUT_K0;
  return BLOCK_INPUT_K0 + (layerIndex - 1) * GROWTH_RATE_K;
}

/** DenseNet-121 四个 Dense block 的层数配置 */
export const DENSENET121_BLOCKS = [
  { name: "Dense Block 1", layers: 6 },
  { name: "Dense Block 2", layers: 12 },
  { name: "Dense Block 3", layers: 24 },
  { name: "Dense Block 4", layers: 16 },
];

/** DenseNet vs ResNet 参数量 / Top-5 错误率对比(ImageNet,来自训练细节章节表格) */
export interface ModelComparisonPoint {
  model: string;
  paramsM: number; // 百万参数
  top5Error: number; // %
  family: "resnet" | "densenet";
}

export const MODEL_COMPARISON: ModelComparisonPoint[] = [
  { model: "ResNet-50", paramsM: 25.6, top5Error: 6.7, family: "resnet" },
  { model: "ResNet-152", paramsM: 60.2, top5Error: 5.6, family: "resnet" },
  { model: "DenseNet-121", paramsM: 7.0, top5Error: 6.1, family: "densenet" },
  { model: "DenseNet-169", paramsM: 14.1, top5Error: 5.5, family: "densenet" },
  { model: "DenseNet-201", paramsM: 20.0, top5Error: 5.2, family: "densenet" },
  { model: "DenseNet-264", paramsM: 33.3, top5Error: 5.0, family: "densenet" },
];

/** Bottleneck + Compression 前后的参数量对比(文中提到的关键数字) */
export const BC_PARAM_SAVINGS = {
  withoutBC: 20, // 若无 BC,DenseNet-121 会从 7M 涨到的量级(单位 M)
  withBC: 7.0,
};

/** Transition layer 压缩因子 */
export const TRANSITION_COMPRESSION_THETA = 0.5;

/** Bottleneck 中间层扩张倍数(1×1 conv 把输入压到 bn_size * k) */
export const BOTTLENECK_BN_SIZE = 4;
