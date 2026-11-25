import { TOY_WAVEFORM, N_FRAMES, N_MELS, logMelSpectrogram } from "../lib/data";

const CELL = 30;

export function SpectrogramWidget() {
  const spec = logMelSpectrogram(TOY_WAVEFORM);

  return (
    <div>
      <svg viewBox={`0 0 ${N_FRAMES * CELL + 60} ${N_MELS * CELL + 40}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="log-mel 频谱图可视化">
        <text x={(N_FRAMES * CELL + 60) / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          log-mel 频谱图(横轴时间,纵轴 mel 频率)
        </text>
        {spec.map((row, t) =>
          row.map((v, m) => {
            const lightness = 90 - v * 55;
            return (
              <rect
                key={`${t}-${m}`}
                x={40 + t * CELL}
                y={30 + (N_MELS - 1 - m) * CELL}
                width={CELL - 1}
                height={CELL - 1}
                fill={`hsl(350, 70%, ${lightness}%)`}
              />
            );
          })
        )}
        <text x={20} y={30 + (N_MELS * CELL) / 2} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" transform={`rotate(-90, 20, ${30 + (N_MELS * CELL) / 2})`}>
          mel 频率
        </text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        颜色越深表示该时间-频率位置能量越高。Whisper 用这种频谱图(而非原始波形)作为 Transformer encoder 的输入。
      </p>
    </div>
  );
}
