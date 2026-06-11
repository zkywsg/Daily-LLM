import type { MiniArchProps } from "./types";

export function MiniSeq2Seq({
  width = 160,
  height = 70,
  ariaLabel = "Seq2Seq 架构缩图",
}: MiniArchProps) {
  // 左半 encoder 三个块 → 中间一个圆形上下文向量 c → 右半 decoder 三个块
  const enc = [4, 22, 40];
  const dec = [104, 122, 140];
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Encoder 三块 — 输入色调 */}
        {enc.map((x, i) => (
          <rect
            key={`e-${i}`}
            x={x}
            y="32"
            width="14"
            height="14"
            rx="2"
            className="illustration__layer illustration__layer--input"
          />
        ))}
        {/* 上下文向量 c — 中央圆 */}
        <circle cx="78" cy="39" r="10" className="illustration__addnorm" />
        <text x="78" y="42" textAnchor="middle" fontSize="9">
          c
        </text>
        {/* Decoder 三块 — 输出色调 */}
        {dec.map((x, i) => (
          <rect
            key={`d-${i}`}
            x={x}
            y="32"
            width="14"
            height="14"
            rx="2"
            className="illustration__proj illustration__proj--act"
          />
        ))}
        {/* enc → c → dec 三条主线 */}
        <line
          x1="58"
          y1="39"
          x2="68"
          y2="39"
          className="illustration__branch illustration__branch--q"
        />
        <line
          x1="88"
          y1="39"
          x2="100"
          y2="39"
          className="illustration__branch illustration__branch--q"
        />
      </g>
    </svg>
  );
}
