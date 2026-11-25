// Sora demo 数据:spacetime patch 切分可视化 + 模型规模 vs 质量曲线 +
// 原生分辨率/时长预设的 patch 数量演示。

export interface VideoConfig {
  label: string;
  frames: number;
  height: number;
  width: number;
}

export const PRESET_CONFIGS: VideoConfig[] = [
  { label: "方形短片", frames: 8, height: 4, width: 4 },
  { label: "竖屏(9:16)", frames: 12, height: 6, width: 3 },
  { label: "宽屏(16:9)", frames: 6, height: 3, width: 6 },
  { label: "长视频", frames: 20, height: 4, width: 4 },
];

/** 给定视频体和 patch 大小,计算沿三个维度的 patch 数量 */
export function patchCounts(cfg: VideoConfig, patchSize: number): { pt: number; ph: number; pw: number; total: number } {
  const pt = Math.ceil(cfg.frames / patchSize);
  const ph = Math.ceil(cfg.height / patchSize);
  const pw = Math.ceil(cfg.width / patchSize);
  return { pt, ph, pw, total: pt * ph * pw };
}

/** 模型规模(参数量档位,单位任意)vs 生成质量的示意曲线:边际收益递减但不封顶 */
export function scaleToQuality(scale: number): number {
  return 1 - Math.exp(-scale / 3);
}

export const SCALE_PRESETS = [0.5, 1, 2, 4, 8, 16];
