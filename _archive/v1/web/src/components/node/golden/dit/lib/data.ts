// DiT 模型规模 (论文 Table 1)
export interface DiTSize {
  name: string;
  layers: number;
  dModel: number;
  heads: number;
  paramsM: number;
  gflops: number;
  fid: number;        // FID-50K cfg=1.5 at 400K iter for scaling demo
}

export const DIT_SIZES: DiTSize[] = [
  { name: "DiT-S/2", layers: 12, dModel: 384,  heads: 6,  paramsM: 33,  gflops: 6,   fid: 68.40 },
  { name: "DiT-B/2", layers: 12, dModel: 768,  heads: 12, paramsM: 130, gflops: 23,  fid: 43.47 },
  { name: "DiT-L/2", layers: 24, dModel: 1024, heads: 16, paramsM: 458, gflops: 80,  fid: 23.33 },
  { name: "DiT-XL/2",layers: 28, dModel: 1152, heads: 16, paramsM: 675, gflops: 119, fid: 19.47 },
];

// DiT vs U-Net 在 ImageNet 256 SOTA
export interface SotaRow {
  model: string;
  params: number;
  gflops: number;
  fid: number;
  isDit: boolean;
}
export const SOTA_COMPARE: SotaRow[] = [
  { model: "ADM (U-Net)",     params: 554, gflops: 119, fid: 3.94, isDit: false },
  { model: "LDM (U-Net+lat)", params: 400, gflops: 104, fid: 3.60, isDit: false },
  { model: "DiT-XL/2",        params: 675, gflops: 119, fid: 2.27, isDit: true  },
];

// DiT family timeline
export interface DiTVariant {
  name: string;
  year: number;
  month: number;
  org: string;
  oneLiner: string;
  color: string;
  isVideo?: boolean;
}
export const DIT_FAMILY: DiTVariant[] = [
  { name: "DiT",        year: 2022, month: 12, org: "UC Berkeley",     oneLiner: "原始 · class-conditional ImageNet",       color: "#ec4899" },
  { name: "PixArt-α",   year: 2023, month: 9,  org: "Huawei",          oneLiner: "文生图 · T5 cross-attention · 10% 训练成本", color: "#f59e0b" },
  { name: "SD 3",       year: 2024, month: 3,  org: "Stability",       oneLiner: "MM-DiT · 文图 joint self-attention",      color: "#10b981" },
  { name: "Sora",       year: 2024, month: 2,  org: "OpenAI",          oneLiner: "时空 patches · 视频 DiT",                  color: "#a855f7", isVideo: true },
  { name: "FLUX",       year: 2024, month: 8,  org: "Black Forest",    oneLiner: "12B 开源 · 压过 Midjourney v6",            color: "#3b82f6" },
];

// adaLN-Zero 调节参数:t 影响曲线
// 模拟训练初期 (α≈0) 到训练后期 (α≈0.5-1) 的演化
export function adaLNParam(epoch: number, target: number): number {
  // sigmoid-ish curve
  const r = 1 / (1 + Math.exp(-(epoch - 5) * 0.6));
  return target * r;
}
