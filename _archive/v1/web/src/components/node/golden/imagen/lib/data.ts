// Text encoder scaling: FID vs 参数量
export interface EncoderRow {
  name: string;
  params: number; // B
  fid: number;
}
export const ENCODER_SCALING: EncoderRow[] = [
  { name: "CLIP ViT-L/14", params: 0.4,  fid: 12.1 },
  { name: "T5-Small",       params: 0.06, fid: 13.5 },
  { name: "T5-Base",        params: 0.22, fid: 11.4 },
  { name: "T5-Large",       params: 0.74, fid: 10.2 },
  { name: "T5-XL",          params: 3,    fid: 8.6 },
  { name: "T5-XXL",         params: 11,   fid: 7.27 },
];

// CFG guidance scale demo:模拟 fidelity / creativity / distortion 三条曲线
export function cfgFidelity(w: number): number {
  // 越高越接近 prompt,但过高会饱和
  return Math.min(100, 20 + w * 9);
}
export function cfgCreativity(w: number): number {
  // 越低 w 越有创造性(远离 prompt 但多样)
  return Math.max(5, 90 - w * 6);
}
export function cfgDistortion(w: number): number {
  // w 越大失真越明显,阈值后陡增
  if (w <= 10) return Math.max(0, (w - 5) * 2);
  return 10 + (w - 10) * 8;
}

// SOTA benchmark
export interface SotaRow {
  model: string;
  fid: number;
  org: string;
  isImagen: boolean;
}
export const SOTA_COMPARE: SotaRow[] = [
  { model: "DALL-E 2",         fid: 10.39, org: "OpenAI",     isImagen: false },
  { model: "GLIDE",             fid: 12.24, org: "OpenAI",     isImagen: false },
  { model: "Make-A-Scene",      fid: 11.84, org: "Meta",       isImagen: false },
  { model: "Stable Diffusion",  fid: 12.6,  org: "Stability",  isImagen: false },
  { model: "Imagen",            fid: 7.27,  org: "Google",     isImagen: true  },
];

// Cascade 分辨率阶段
export interface CascadeStage {
  name: string;
  resolution: number;
  params: string;
  role: string;
}
export const CASCADE_STAGES: CascadeStage[] = [
  { name: "Stage 1", resolution: 64,   params: "~2B",   role: "text → 语义映射" },
  { name: "Stage 2", resolution: 256,  params: "~600M", role: "super-res 高频细节" },
  { name: "Stage 3", resolution: 1024, params: "~400M", role: "super-res 高频细节" },
];
