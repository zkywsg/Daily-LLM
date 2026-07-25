// Whisper demo 数据:toy 波形 → log-mel 频谱可视化 + 弱监督数据质量过滤 +
// 多任务前缀-输出示例。全部确定性构造。

export const N_FRAMES = 10;
export const N_MELS = 6;

export const TOY_WAVEFORM: number[] = Array.from({ length: 80 }, (_, i) =>
  Math.sin(i * 0.5) * 0.4 + Math.sin(i * 0.13) * 0.3 + Math.sin(i * 0.05) * 0.2
);

/** 简化 log-mel 频谱:把波形分帧,每帧与几个固定频率模板做余弦相关,取 log(1+|.|) 作为能量,
 * 最后按整体最大值归一化到 [0,1]。不是真实 FFT,但作为示意频谱图数值上是合理有界的。 */
export function logMelSpectrogram(waveform: number[]): number[][] {
  const frameLen = Math.floor(waveform.length / N_FRAMES);
  const spec: number[][] = [];
  for (let t = 0; t < N_FRAMES; t++) {
    const frame = waveform.slice(t * frameLen, (t + 1) * frameLen);
    const row: number[] = [];
    for (let m = 0; m < N_MELS; m++) {
      const freq = (m + 1) * 0.3;
      let energy = 0;
      frame.forEach((v, i) => { energy += v * Math.cos(freq * i); });
      row.push(Math.log(1 + Math.abs(energy)));
    }
    spec.push(row);
  }
  const maxV = Math.max(...spec.flat(), 1e-6);
  return spec.map((row) => row.map((v) => v / maxV));
}

/** 68 万小时弱监督数据里,一小批样本的质量分示例(0-1,越高越像高质量人工转写) */
export const RAW_QUALITY_SCORES: number[] = [0.9, 0.85, 0.2, 0.75, 0.1, 0.6, 0.95, 0.15, 0.8, 0.3, 0.7, 0.05];

export function filterLowQuality(scores: number[], threshold = 0.5): number[] {
  return scores.filter((s) => s >= threshold);
}

export const TASK_PREFIXES: Array<{ id: string; label: string; prefix: string; outputExample: string }> = [
  { id: "transcribe", label: "转写", prefix: "<|transcribe|>", outputExample: "今天天气不错。" },
  { id: "translate", label: "翻译", prefix: "<|translate|>", outputExample: "The weather is nice today." },
  { id: "langid", label: "语言识别", prefix: "<|langid|>", outputExample: "zh(中文)" },
  { id: "timestamp", label: "时间戳", prefix: "<|timestamps|>", outputExample: "[00:00–00:03] 今天天气不错。" },
];
