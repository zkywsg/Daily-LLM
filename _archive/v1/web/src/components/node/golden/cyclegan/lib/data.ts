// CycleGAN 双向 G/D 结构:两个 domain X / Y,两个 generator,两个 discriminator
export interface DomainMapping {
  label: string;
  from: "X" | "Y";
  to: "X" | "Y";
  generator: string;
  discriminator: string;
  example: string;
}

export const DOMAIN_MAPPINGS: DomainMapping[] = [
  {
    label: "G: X → Y",
    from: "X",
    to: "Y",
    generator: "G",
    discriminator: "D_Y",
    example: "马 → 斑马",
  },
  {
    label: "F: Y → X",
    from: "Y",
    to: "X",
    generator: "F",
    discriminator: "D_X",
    example: "斑马 → 马",
  },
];

// Cycle consistency 训练步数 -> 重建误差(示意曲线,非真实训练日志)
export interface CycleStepPoint {
  step: number;
  reconError: number; // ||F(G(x)) - x||_1,已归一化到 0-1
  contentPreserved: boolean;
}

export const CYCLE_TRAINING_CURVE: CycleStepPoint[] = [
  { step: 0, reconError: 0.95, contentPreserved: false },
  { step: 20, reconError: 0.7, contentPreserved: false },
  { step: 50, reconError: 0.45, contentPreserved: true },
  { step: 100, reconError: 0.25, contentPreserved: true },
  { step: 150, reconError: 0.12, contentPreserved: true },
  { step: 200, reconError: 0.06, contentPreserved: true },
];

// 无 cycle loss 时的模式塌缩对比
export const NO_CYCLE_RESULT = {
  label: "无 cycle loss",
  description: "所有马都映射到同一张“标准斑马”,G(x) 与输入 x 内容无关,模式塌缩",
};

export const WITH_CYCLE_RESULT = {
  label: "有 cycle loss",
  description: "F(G(x)) 能恢复原图姿态 / 背景 / 光照,G 保留了 x 的内容信息",
};

// pix2pix(配对) vs CycleGAN(无配对)对比
export interface PairedComparisonRow {
  method: string;
  dataRequirement: string;
  needsPaired: boolean;
}

export const PAIRED_VS_UNPAIRED: PairedComparisonRow[] = [
  { method: "pix2pix", dataRequirement: "{(x_i, y_i)} 严格配对", needsPaired: true },
  { method: "CycleGAN", dataRequirement: "X 域集合 + Y 域集合,无需一一对应", needsPaired: false },
];

// 无配对任务示例(论文标志性 demo)
export interface UnpairedTask {
  domainX: string;
  domainY: string;
  note: string;
}

export const UNPAIRED_TASKS: UnpairedTask[] = [
  { domainX: "马", domainY: "斑马", note: "ImageNet 各 ~1000 张,无配对,保留姿态/背景/光照" },
  { domainX: "风景照", domainY: "莫奈 / 梵高 / 塞尚画风", note: "莫奈画 1074 张 + 风景照 6287 张" },
  { domainX: "夏季照片", domainY: "冬季照片", note: "优胜美地同地点不同季节,无配对" },
  { domainX: "苹果", domainY: "橙子", note: "外观互转,几何形状变化时会失败" },
];

// Cityscapes labels ↔ photos 定量对比
export interface QuantRow {
  method: string;
  perPixelAcc: number;
  perClassAcc: number;
  classIoU: number;
  paired: boolean;
  highlight?: boolean;
}

export const CITYSCAPES_COMPARE: QuantRow[] = [
  { method: "pix2pix(配对监督)", perPixelAcc: 0.71, perClassAcc: 0.25, classIoU: 0.18, paired: true },
  { method: "CoGAN(无配对)", perPixelAcc: 0.4, perClassAcc: 0.1, classIoU: 0.06, paired: false },
  { method: "CycleGAN(无配对)", perPixelAcc: 0.52, perClassAcc: 0.17, classIoU: 0.11, paired: false, highlight: true },
];

// AMT human perceptual study:被骗比例
export interface AmtRow {
  task: string;
  foolRate: number; // 百分比,随机猜是 50%
}

export const AMT_STUDY: AmtRow[] = [
  { task: "maps → aerial", foolRate: 26 },
  { task: "街景 → labels", foolRate: 23 },
];

// 三件套稳定剂(机制三)
export interface StabilizerRow {
  name: string;
  role: string;
}

export const ENGINEERING_STABILIZERS: StabilizerRow[] = [
  { name: "PatchGAN Discriminator", role: "70×70 局部判别,减少 D 参数、加快训练、提升纹理细节" },
  { name: "LSGAN loss(MSE)", role: "替代 BCE,避免梯度饱和,让 D 强 G 弱时梯度仍有意义" },
  { name: "ResNet Generator", role: "9 个 residual block,学习“在输入上叠加修改”而非从零重建,适合局部翻译任务" },
];

// loss 组成(用于 sticky panel 摘要)
export const LOSS_TERMS = [
  { name: "L_GAN", formula: "L_GAN(G,D_Y,X,Y) + L_GAN(F,D_X,Y,X)", role: "让 G(x) 像 Y 域、F(y) 像 X 域" },
  { name: "L_cyc", formula: "||F(G(x))-x||₁ + ||G(F(y))-y||₁", role: "强迫内容保留,λ=10" },
  { name: "L_idt(可选)", formula: "||G(y)-y||₁ + ||F(x)-x||₁", role: "输入已在目标域时输出原样,保留风格/颜色" },
];
