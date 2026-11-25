// DCGAN Generator 各层的空间尺寸变化(z -> 64x64x3)
export interface ConvLayer {
  label: string;
  size: string;
  channels: number;
}

export const GENERATOR_LAYERS_DCGAN: ConvLayer[] = [
  { label: "z (noise)", size: "1×1", channels: 100 },
  { label: "deconv1", size: "4×4", channels: 1024 },
  { label: "deconv2", size: "8×8", channels: 512 },
  { label: "deconv3", size: "16×16", channels: 256 },
  { label: "deconv4", size: "32×32", channels: 128 },
  { label: "output (tanh)", size: "64×64", channels: 3 },
];

export const DISCRIMINATOR_LAYERS_DCGAN: ConvLayer[] = [
  { label: "input", size: "64×64", channels: 3 },
  { label: "conv1", size: "32×32", channels: 128 },
  { label: "conv2", size: "16×16", channels: 256 },
  { label: "conv3", size: "8×8", channels: 512 },
  { label: "conv4", size: "4×4", channels: 1024 },
  { label: "output (sigmoid)", size: "1×1", channels: 1 },
];

// 原版 GAN(MLP)的 G/D 结构 — 用于和 DCGAN 全卷积对比
export interface MlpLayer {
  label: string;
  units: number;
}

export const GENERATOR_LAYERS_MLP: MlpLayer[] = [
  { label: "z (noise)", units: 100 },
  { label: "fc1 + ReLU", units: 256 },
  { label: "fc2 + ReLU", units: 512 },
  { label: "fc3 + ReLU", units: 1024 },
  { label: "fc4 (tanh)", units: 4096 },
];

export const DISCRIMINATOR_LAYERS_MLP: MlpLayer[] = [
  { label: "input (flatten)", units: 4096 },
  { label: "fc1 + LeakyReLU", units: 1024 },
  { label: "fc2 + LeakyReLU", units: 512 },
  { label: "fc3 + LeakyReLU", units: 256 },
  { label: "fc4 (sigmoid)", units: 1 },
];

// DCGAN 之前 / 之后 GAN 训练复现成功率(论文强调的工程影响)
export interface ReproRow {
  label: string;
  successRate: number; // 0-100
}
export const REPRO_RATE: ReproRow[] = [
  { label: "DCGAN 之前(仅 MLP-GAN)", successRate: 28 },
  { label: "DCGAN 之后(CNN + 三件套)", successRate: 92 },
];

// CIFAR-10 监督学习对比(用 D 当特征提取器 + L2-SVM)
export interface Cifar10Row {
  model: string;
  acc: number;
  highlight?: boolean;
}
export const CIFAR10_COMPARE: Cifar10Row[] = [
  { model: "K-means(无监督 baseline)", acc: 80.6 },
  { model: "DCGAN + L2-SVM", acc: 82.8, highlight: true },
  { model: "Supervised CNN", acc: 84.8 },
];

// 架构指南 6 条
export const ARCHITECTURE_GUIDELINES: string[] = [
  "去 fc — G/D 全程在 conv 上操作",
  "加 BN — G 和 D 几乎所有层都用,除了 G output 和 D input",
  "G 用 ReLU 内部 + tanh 输出 — 输出 normalize 到 [-1, 1]",
  "D 用 LeakyReLU(0.2) — 避免 dying ReLU",
  "strided conv 替代 pool — G 用 transposed conv 上采样,D 用 strided conv 下采样",
  "不用 max-pool — 让网络学采样方式而不是用固定算子",
];

// 训练稳定剂三件套:每件的作用
export interface StabilizerRow {
  name: string;
  appliesTo: string;
  effect: string;
}
export const STABILIZERS: StabilizerRow[] = [
  { name: "BatchNorm", appliesTo: "G/D 几乎所有层(除 G 输出层、D 输入层)", effect: "稳定训练,显著减少 mode collapse" },
  { name: "LeakyReLU(0.2)", appliesTo: "D 全部层", effect: "避免 dying ReLU,难样本仍有梯度流回" },
  { name: "Adam(β₁=0.5)", appliesTo: "G/D 优化器", effect: "降低动量,避免 G/D 被推向极端" },
];

// Latent space 插值 / 算术的锚点示例(用于插值滑块 demo)
export interface LatentAnchor {
  label: string;
  z: number[]; // 简化的低维投影,仅用于可视化位置
}
export const LATENT_ANCHORS: LatentAnchor[] = [
  { label: "neutral woman", z: [-0.6, 0.4] },
  { label: "smiling woman", z: [0.6, 0.7] },
];

// 向量算术示例:smiling woman − neutral woman + neutral man ≈ smiling man
export const VECTOR_ARITHMETIC_LABELS = {
  a: "smiling woman",
  b: "neutral woman",
  c: "neutral man",
  result: "smiling man",
};
