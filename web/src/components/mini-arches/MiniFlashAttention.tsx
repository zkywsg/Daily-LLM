import type { MiniArchProps } from "./types";

export function MiniFlashAttention({
  width = 160,
  height = 70,
  ariaLabel = "FlashAttention 架构缩图",
}: MiniArchProps) {
  // 两层内存层级 + 分块流动:HBM(下,大) 和 SRAM(上,小),
  // 中间小方块在 SRAM 里高亮 — 体现"把 attention 搬到 SRAM 里分块算"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* HBM 层(下,横向长条) */}
        <rect
          x="6"
          y="44"
          width="148"
          height="18"
          rx="2"
          className="illustration__layer illustration__layer--input"
        />
        <text
          x="80"
          y="56"
          textAnchor="middle"
          fontSize="8"
          fill="currentColor"
          opacity="0.55"
        >
          HBM
        </text>
        {/* SRAM 层(上,小窗) */}
        <rect
          x="58"
          y="8"
          width="44"
          height="18"
          rx="2"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <text
          x="80"
          y="20"
          textAnchor="middle"
          fontSize="8"
          fill="currentColor"
          opacity="0.6"
        >
          SRAM
        </text>
        {/* 分块上下流动 — 3 个 block 上传到 SRAM */}
        {[20, 80, 140].map((x, i) => (
          <g key={i}>
            <line
              x1={x}
              y1="44"
              x2={x < 60 ? 64 : x > 100 ? 96 : 80}
              y2="26"
              className="illustration__branch illustration__branch--q"
            />
          </g>
        ))}
        {/* SRAM 里的 active block */}
        <rect
          x="74"
          y="12"
          width="12"
          height="10"
          rx="1"
          className="illustration__proj illustration__proj--act"
        />
      </g>
    </svg>
  );
}
