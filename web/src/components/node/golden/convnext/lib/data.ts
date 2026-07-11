// ResNet-50 → ConvNeXt-T 现代化路径:每一步的累积 Top-1 精度
export interface WaterfallStep {
  label: string;
  delta: number; // 相对上一步的精度变化(百分点),起点为 null 语义上用 0
  cumulative: number; // 累积精度
  source: string; // 该改造借鉴自哪个工作
}

export const MODERNIZATION_WATERFALL: WaterfallStep[] = [
  { label: "ResNet-50 起点", delta: 0, cumulative: 76.1, source: "ResNet (2015)" },
  { label: "① 训练 recipe 现代化", delta: 2.7, cumulative: 78.8, source: "DeiT / ViT" },
  { label: "② stage 比例调整", delta: 0.6, cumulative: 79.4, source: "Swin" },
  { label: "③ patchify stem", delta: 0.1, cumulative: 79.5, source: "ViT / Swin" },
  { label: "④ depthwise conv", delta: 1.0, cumulative: 80.5, source: "MobileNet / Swin" },
  { label: "⑤ inverted bottleneck", delta: 0.1, cumulative: 80.6, source: "MobileNet v2" },
  { label: "⑥ 7×7 大 kernel", delta: 0.7, cumulative: 81.3, source: "Swin window size" },
  { label: "⑦ 减少 norm/act + GELU/LN", delta: 0.7, cumulative: 82.0, source: "Transformer" },
  { label: "⑧ 独立 downsample", delta: 0.5, cumulative: 82.0, source: "Swin patch merging" },
];

// 训练 recipe:ResNet-50 原版(2015) vs ConvNeXt(2022)
export interface RecipeRow {
  dim: string;
  oldVal: string;
  newVal: string;
}

export const TRAINING_RECIPE_COMPARE: RecipeRow[] = [
  { dim: "优化器", oldVal: "SGD + Momentum 0.9", newVal: "AdamW (β₁=0.9, β₂=0.999)" },
  { dim: "学习率 schedule", oldVal: "阶梯 decay, lr=0.1", newVal: "cosine decay, lr=4×10⁻³" },
  { dim: "学习率 warmup", oldVal: "无", newVal: "20 epoch linear warmup" },
  { dim: "权重衰减", oldVal: "1×10⁻⁴", newVal: "0.05" },
  { dim: "Batch size", oldVal: "256", newVal: "4096" },
  { dim: "Epochs", oldVal: "90–120", newVal: "300" },
  { dim: "激活函数", oldVal: "ReLU", newVal: "GELU" },
  { dim: "归一化", oldVal: "BatchNorm", newVal: "LayerNorm" },
  { dim: "增强", oldVal: "flip + crop", newVal: "Mixup + CutMix + RandAugment + RandomErasing" },
  { dim: "正则", oldVal: "weight decay 1e-4", newVal: "stochastic depth 0.1–0.5 + label smoothing + EMA" },
];

// ConvNeXt-T vs Swin-T 关键对比(同算力档位)
export interface ModelCompareRow {
  model: string;
  params: string;
  flops: string;
  top1: number;
  throughput?: number; // img/s
  highlight?: boolean;
}

export const CONVNEXT_VS_SWIN: ModelCompareRow[] = [
  { model: "ResNet-50(2015 recipe)", params: "25.6M", flops: "4.1B", top1: 76.1 },
  { model: "ResNet-50(现代 recipe)", params: "25.6M", flops: "4.1B", top1: 78.8 },
  { model: "Swin-T", params: "28M", flops: "4.5B", top1: 81.3, throughput: 645 },
  { model: "ConvNeXt-T", params: "29M", flops: "4.5B", top1: 82.1, throughput: 774, highlight: true },
];

// ImageNet-22K 预训练 + 1K 微调(384×384)
export const CONVNEXT_VS_SWIN_LARGE: ModelCompareRow[] = [
  { model: "Swin-L", params: "197M", flops: "-", top1: 87.3 },
  { model: "ConvNeXt-L", params: "198M", flops: "-", top1: 87.5, highlight: true },
  { model: "Swin-XL(CLIP/SwinV2)", params: "350M", flops: "-", top1: 87.6 },
  { model: "ConvNeXt-XL", params: "350M", flops: "-", top1: 87.8, highlight: true },
];

// ConvNeXt Block 内部结构(用于 Block 结构图)
export interface BlockStep {
  label: string;
  detail: string;
  role: string; // 类比 Transformer 里的角色
}

export const CONVNEXT_BLOCK_STEPS: BlockStep[] = [
  { label: "DWConv 7×7", detail: "depthwise, groups=C", role: "局部 token mixer(≈ self-attention)" },
  { label: "LayerNorm", detail: "channel-last [B,H,W,C]", role: "归一化(≈ Transformer 前置 LN)" },
  { label: "PWConv 1×1 ↑", detail: "升维 4× hidden", role: "FFN 第一层" },
  { label: "GELU", detail: "非线性激活", role: "FFN 中间激活" },
  { label: "PWConv 1×1 ↓", detail: "降回原维度", role: "FFN 第二层" },
  { label: "+ shortcut", detail: "DropPath + 残差", role: "继承自 ResNet" },
];

// Norm/Act 数量对比:ResNet Bottleneck vs ConvNeXt/Transformer Block
export interface NormActCount {
  name: string;
  normCount: number;
  actCount: number;
  normType: string;
  actType: string;
}

export const NORM_ACT_COUNTS: NormActCount[] = [
  { name: "ResNet Bottleneck", normCount: 3, actCount: 3, normType: "BatchNorm", actType: "ReLU" },
  { name: "ConvNeXt Block", normCount: 1, actCount: 1, normType: "LayerNorm", actType: "GELU" },
];
