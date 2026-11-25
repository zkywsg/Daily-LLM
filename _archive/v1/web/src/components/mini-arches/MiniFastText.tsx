import type { MiniArchProps } from "./types";

export function MiniFastText({
  width = 160,
  height = 70,
  ariaLabel = "FastText 架构缩图",
}: MiniArchProps) {
  // word → subword n-grams → sum → word vector
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* word "apple" */}
        <rect x="2" y="28" width="24" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="14" y="38" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">apple</text>
        {/* 拆分箭头 */}
        <text x="32" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.75">→</text>
        {/* 6 个 subword n-gram */}
        {[
          { y: 4, text: "<ap" },
          { y: 14, text: "app" },
          { y: 24, text: "ppl" },
          { y: 34, text: "ple" },
          { y: 44, text: "le>" },
          { y: 54, text: "<apple>" },
        ].map((sg, i) => (
          <g key={i}>
            <rect x="40" y={sg.y} width="32" height="8" rx="1.5"
              className="illustration__featuremap illustration__featuremap--ctx" />
            <text x="56" y={sg.y + 6} textAnchor="middle" fontSize="4"
              fill="currentColor" opacity="0.85">{sg.text}</text>
          </g>
        ))}
        <text x="56" y="-1" textAnchor="middle" fontSize="4" fontWeight="600"
          fill="currentColor" opacity="0.7">subwords (n=3-6)</text>
        {/* hash → bucket vectors */}
        <text x="80" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.75">→</text>
        {/* Σ 块 */}
        <rect x="90" y="22" width="20" height="26" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="100" y="34" textAnchor="middle" fontSize="9" fontWeight="600"
          fill="currentColor" opacity="0.85">Σ</text>
        <text x="100" y="42" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">z_g</text>
        {/* word vector */}
        <rect x="118" y="28" width="20" height="14" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="128" y="38" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">v_apple</text>
        <line x1="110" y1="35" x2="118" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* OOV 处理标签 */}
        <rect x="142" y="14" width="16" height="14" rx="2"
          fill="none"
          strokeDasharray="2 2"
          className="illustration__block" />
        <text x="150" y="22" textAnchor="middle" fontSize="3.8" fontWeight="600"
          fill="currentColor" opacity="0.75">OOV</text>
        <text x="150" y="26" textAnchor="middle" fontSize="3"
          fill="currentColor" opacity="0.6">applle</text>
        {/* 底部 */}
        <text x="80" y="68" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          subword 组合 · 处理 OOV / 形态丰富语言
        </text>
      </g>
    </svg>
  );
}
