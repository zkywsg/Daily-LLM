import type { MiniArchProps } from "./types";

export function MiniGAN({
  width = 160,
  height = 70,
  ariaLabel = "GAN 架构缩图",
}: MiniArchProps) {
  // z → G → fake → D ← real;突出对抗博弈
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* z(噪声) */}
        <rect x="2" y="30" width="14" height="10" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="9" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">z</text>
        {/* G */}
        <rect x="22" y="22" width="26" height="26" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="35" y="32" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.85">G</text>
        <text x="35" y="40" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.7">造假者</text>
        {/* fake(G(z)) */}
        <rect x="54" y="14" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="65" y="22" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.85">G(z)</text>
        <text x="65" y="27" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">fake</text>
        {/* real(数据) */}
        <rect x="54" y="42" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="65" y="50" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.85">x</text>
        <text x="65" y="55" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">real</text>
        {/* D */}
        <rect x="84" y="22" width="26" height="26" rx="3"
          className="illustration__proj illustration__proj--ffn" />
        <text x="97" y="32" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.85">D</text>
        <text x="97" y="40" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.7">鉴别者</text>
        {/* 输出 real/fake */}
        <rect x="116" y="30" width="20" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="126" y="38" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">real?</text>
        {/* 箭头 */}
        <line x1="16" y1="35" x2="22" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="48" y1="29" x2="54" y2="21"
          className="illustration__branch illustration__branch--q" />
        <line x1="76" y1="21" x2="84" y2="29"
          className="illustration__branch illustration__branch--q" />
        <line x1="76" y1="49" x2="84" y2="41"
          className="illustration__branch illustration__branch--q" />
        <line x1="110" y1="35" x2="116" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 反馈回路:对抗梯度(虚线箭头 D → G) */}
        <path d="M 97 48 Q 60 64 35 48"
          fill="none"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--v" opacity="0.6" />
        <text x="65" y="68" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.65">minimax 对抗</text>
        {/* 标签:min max */}
        <text x="80" y="6" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.7">min_G  max_D</text>
      </g>
    </svg>
  );
}
