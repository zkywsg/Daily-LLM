import type { MiniArchProps } from "./types";

export function MiniDistilBERT({
  width = 160,
  height = 70,
  ariaLabel = "DistilBERT 架构缩图",
}: MiniArchProps) {
  // Teacher (大,左) → Student (小,右),中间一条蒸馏箭头 + KD loss
  // 突出"小但更密"的 student
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Teacher BERT-base 12 层(左侧大方块) */}
        <rect
          x="4"
          y="10"
          width="46"
          height="48"
          rx="3"
          className="illustration__proj illustration__proj--ffn"
          opacity="0.7"
        />
        {Array.from({ length: 12 }, (_, i) => (
          <line
            key={i}
            x1="8"
            y1={14 + i * 3.5}
            x2="46"
            y2={14 + i * 3.5}
            stroke="currentColor"
            strokeWidth="0.4"
            opacity="0.4"
          />
        ))}
        <text
          x="27"
          y="66"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.7"
        >
          Teacher (12)
        </text>
        {/* 蒸馏箭头 + loss */}
        <line
          x1="50"
          y1="34"
          x2="90"
          y2="34"
          className="illustration__branch illustration__branch--v"
        />
        <text
          x="70"
          y="28"
          textAnchor="middle"
          fontSize="6"
          fontWeight="600"
          fill="currentColor"
          opacity="0.75"
        >
          KD
        </text>
        <text
          x="70"
          y="44"
          textAnchor="middle"
          fontSize="5"
          fill="currentColor"
          opacity="0.6"
        >
          τ=2
        </text>
        {/* Student DistilBERT 6 层(右侧小方块) */}
        <rect
          x="94"
          y="20"
          width="46"
          height="28"
          rx="3"
          className="illustration__proj illustration__proj--v"
        />
        {Array.from({ length: 6 }, (_, i) => (
          <line
            key={i}
            x1="98"
            y1={23 + i * 4}
            x2="136"
            y2={23 + i * 4}
            stroke="currentColor"
            strokeWidth="0.5"
            opacity="0.6"
          />
        ))}
        <text
          x="117"
          y="66"
          textAnchor="middle"
          fontSize="6"
          fontWeight="600"
          fill="currentColor"
          opacity="0.8"
        >
          Student (6)
        </text>
        {/* 顶部三损失标签 */}
        <text
          x="80"
          y="8"
          textAnchor="middle"
          fontSize="5"
          fill="currentColor"
          opacity="0.5"
        >
          L_distill + L_MLM + L_cos
        </text>
      </g>
    </svg>
  );
}
