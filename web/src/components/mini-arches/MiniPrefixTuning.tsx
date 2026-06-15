import type { MiniArchProps } from "./types";

export function MiniPrefixTuning({
  width = 160,
  height = 70,
  ariaLabel = "Prefix Tuning 架构缩图",
}: MiniArchProps) {
  // 在 K/V 序列前拼 m 个可训练 prefix token
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 顶部:Q(原 input) */}
        <text x="4" y="11" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.7">Q (input)</text>
        {[0, 1, 2, 3].map((i) => (
          <rect key={i} x={48 + i * 9} y="6" width="7" height="8" rx="1"
            className="illustration__featuremap illustration__featuremap--ctx" />
        ))}
        {/* 中间:K(prefix + input) */}
        <text x="4" y="35" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.7">K</text>
        {/* prefix(高亮,可训练) */}
        {[0, 1, 2].map((i) => (
          <rect key={"pk" + i} x={20 + i * 7} y="30" width="6" height="8" rx="1"
            className="illustration__proj illustration__proj--v" />
        ))}
        <text x="29" y="46" textAnchor="middle" fontSize="3.8" fontWeight="600"
          fill="currentColor" opacity="0.85">P_K</text>
        {/* input(灰,冻结) */}
        {[0, 1, 2, 3].map((i) => (
          <rect key={"ik" + i} x={48 + i * 9} y="30" width="7" height="8" rx="1"
            className="illustration__featuremap illustration__featuremap--ctx" opacity="0.6" />
        ))}
        {/* 底部:V */}
        <text x="4" y="62" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.7">V</text>
        {[0, 1, 2].map((i) => (
          <rect key={"pv" + i} x={20 + i * 7} y="57" width="6" height="8" rx="1"
            className="illustration__proj illustration__proj--v" />
        ))}
        <text x="29" y="54" textAnchor="middle" fontSize="3.8" fontWeight="600"
          fill="currentColor" opacity="0.85">P_V</text>
        {[0, 1, 2, 3].map((i) => (
          <rect key={"iv" + i} x={48 + i * 9} y="57" width="7" height="8" rx="1"
            className="illustration__featuremap illustration__featuremap--ctx" opacity="0.6" />
        ))}
        {/* 右侧:soft prefix MLP 重参数化 */}
        <rect x="100" y="14" width="32" height="42" rx="3"
          className="illustration__proj illustration__proj--ffn" />
        <text x="116" y="24" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">MLP</text>
        <text x="116" y="32" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">soft</text>
        <text x="116" y="38" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">prefix</text>
        <text x="116" y="46" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">trainable</text>
        <text x="116" y="52" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.65">0.1%</text>
        {/* MLP 到 prefix 的箭头(虚线表示参数化) */}
        <line x1="100" y1="34" x2="42" y2="34"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--v" />
        <line x1="100" y1="61" x2="42" y2="61"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--v" />
        {/* y */}
        <rect x="142" y="30" width="14" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="149" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">y</text>
      </g>
    </svg>
  );
}
