// FFHQ FID 对比
export interface FidRow {
  model: string;
  fid: number;
  isStyleGan: boolean;
}
export const FFHQ_FID: FidRow[] = [
  { model: "Progressive GAN", fid: 8.04, isStyleGan: false },
  { model: "StyleGAN",        fid: 4.40, isStyleGan: true },
  { model: "StyleGAN2",       fid: 2.84, isStyleGan: true },
];

// PPL 对比
export const PPL_COMPARE = [
  { space: "Z (传统)",   full: 412.0, end: 415.3, color: "#9ca3af" },
  { space: "W (StyleGAN)", full: 228.9, end: 200.5, color: "#ec4899" },
];

// LSUN bench
export interface LsunRow {
  cls: string;
  prog: number;
  stylegan: number;
}
export const LSUN: LsunRow[] = [
  { cls: "Bedroom", prog: 8.34,  stylegan: 2.65 },
  { cls: "Car",     prog: 12.99, stylegan: 5.07 },
  { cls: "Cat",     prog: 37.52, stylegan: 8.53 },
];

// 各 layer 分辨率 + 控制粒度
export interface LayerSpec {
  res: number;          // 4 .. 1024
  granularity: "粗" | "中" | "细";
  controls: string;
  color: string;
}
export const LAYER_SPECS: LayerSpec[] = [
  { res: 4,    granularity: "粗", controls: "整体姿态", color: "#ec4899" },
  { res: 8,    granularity: "粗", controls: "脸型轮廓",  color: "#ec4899" },
  { res: 16,   granularity: "中", controls: "发型走向", color: "#f59e0b" },
  { res: 32,   granularity: "中", controls: "眼神嘴型", color: "#f59e0b" },
  { res: 64,   granularity: "细", controls: "肤色基调", color: "#10b981" },
  { res: 128,  granularity: "细", controls: "皮肤质地", color: "#10b981" },
  { res: 256,  granularity: "细", controls: "雀斑毛发", color: "#10b981" },
  { res: 512,  granularity: "细", controls: "细微纹理", color: "#10b981" },
  { res: 1024, granularity: "细", controls: "光线反射", color: "#10b981" },
];

// Style mixing demo:不同 split point 对应的 "持有人 vs 提供人"
export interface MixConfig {
  label: string;
  splitLayer: number;     // 0..9, 在哪一层切换 w_A → w_B
  desc: string;
}
export const MIX_CONFIGS: MixConfig[] = [
  { label: "全 A",                  splitLayer: 9, desc: "完全使用人 A" },
  { label: "A 粗 + B 中细 (第 2 层)", splitLayer: 2, desc: "A 姿态/脸型 + B 发型/肤色" },
  { label: "A 粗中 + B 细 (第 4 层)", splitLayer: 4, desc: "A 姿态发型 + B 肤色纹理" },
  { label: "A 粗中细 + B 极细 (第 7 层)", splitLayer: 7, desc: "A 主体 + B 仅光线/反射" },
  { label: "全 B",                  splitLayer: 0, desc: "完全使用人 B" },
];
