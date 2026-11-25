import type { MiniArchProps } from "./types";

export function MiniFlamingo({
  width = 160,
  height = 70,
  ariaLabel = "Flamingo 架构缩图",
}: MiniArchProps) {
  // 冻结大 LLM stack + 间隔插入的 cross-attention 层(高亮)+ Perceiver 视觉模块
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:视觉编码器 + Perceiver Resampler */}
        <rect x="4" y="20" width="16" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="12" y="29" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.65">img</text>
        <rect x="4" y="38" width="16" height="20" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="12" y="48" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.75">Perc</text>
        <text x="12" y="55" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">64q</text>
        {/* 中央:冻结 LLM stack — 每隔几层有一个高亮的 cross-attention */}
        {Array.from({ length: 12 }, (_, i) => {
          const isXAttn = (i + 1) % 4 === 0;  // 每 4 层一个 cross-attn
          return (
            <rect key={i} x="50" y={6 + i * 4.4} width="60"
              height={isXAttn ? 3.4 : 2.4} rx="0.5"
              className={isXAttn
                ? "illustration__proj illustration__proj--v"
                : "illustration__proj illustration__proj--ffn"}
              opacity={isXAttn ? 1 : 0.55} />
          );
        })}
        {/* Perceiver → 每个 cross-attn 层的虚线箭头 */}
        {[24, 41, 58].map((y, i) => (
          <line key={i} x1="20" y1="48" x2="50" y2={y}
            className="illustration__residual" />
        ))}
        {/* 右侧输出 */}
        <rect x="128" y="28" width="28" height="14" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="142" y="37" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">text</text>
        <line x1="110" y1="35" x2="128" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="65" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">
          冻结 70B LLM · gated cross-attn 间隔注入
        </text>
      </g>
    </svg>
  );
}
