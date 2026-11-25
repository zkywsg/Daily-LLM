import { NUM_LEVELS, NUM_TIMESTEPS, encodecCode } from "../lib/data";

const CELL = 36;

export function EncodecRvqWidget() {
  return (
    <div>
      <svg viewBox={`0 0 ${NUM_TIMESTEPS * CELL + 80} ${NUM_LEVELS * CELL + 40}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="EnCodec 多层并行残差量化码本可视化">
        <text x={(NUM_TIMESTEPS * CELL + 80) / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          每个时间步 {NUM_LEVELS} 个并行码本(粗→细)
        </text>
        {Array.from({ length: NUM_LEVELS }, (_, level) => (
          <text key={level} x={20} y={40 + level * CELL + CELL / 2 + 4} fontSize={10} fill="var(--ink-muted)">层{level}</text>
        ))}
        {Array.from({ length: NUM_TIMESTEPS }, (_, t) =>
          Array.from({ length: NUM_LEVELS }, (_, level) => {
            const code = encodecCode(t, level);
            const lightness = 90 - level * 12;
            return (
              <g key={`${t}-${level}`}>
                <rect x={50 + t * CELL} y={30 + level * CELL} width={CELL - 2} height={CELL - 2} fill={`hsl(350, 70%, ${lightness}%)`} />
                <text x={50 + t * CELL + (CELL - 2) / 2} y={30 + level * CELL + (CELL - 2) / 2 + 4} textAnchor="middle" fontSize={10} fill="#fff">{code}</text>
              </g>
            );
          })
        )}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        每一帧时间步上,{NUM_LEVELS} 个码本并行各自贡献一个离散 token——层 0 捕捉最粗粒度的信息,层数越高补充的细节越精细。单阶段模型需要同时处理所有层,而不是像 AudioLM 那样分阶段生成。
      </p>
    </div>
  );
}
