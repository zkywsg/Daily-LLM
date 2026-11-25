import type { MiniArchProps } from "./types";

export function MiniLLaVA({
  width = 160,
  height = 70,
  ariaLabel = "LLaVA 架构缩图",
}: MiniArchProps) {
  // 极简:CLIP visual + linear projection + LLaMA
  // 突出 projection 是单一的小块,体现"极简胜过 Q-Former"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:图像 */}
        <rect x="2" y="14" width="20" height="20" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="12" y="44" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.65">img</text>
        {/* CLIP encoder(中等大小,frozen) */}
        <rect x="28" y="12" width="22" height="26" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.6" />
        <text x="39" y="22" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.75">CLIP</text>
        <text x="39" y="29" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">ViT</text>
        <text x="39" y="36" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">frozen</text>
        {/* 极简 projection — 单一小竖条强调"就一个 linear" */}
        <rect x="56" y="20" width="4" height="14" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="58" y="44" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">linear</text>
        <text x="58" y="50" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">↓</text>
        <text x="58" y="55" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.65">4M</text>
        {/* LLaMA 大块 — 全微调 */}
        <rect x="68" y="8" width="58" height="36" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="97" y="22" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">LLaMA</text>
        <text x="97" y="30" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.75">Vicuna 13B</text>
        <text x="97" y="38" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">full FT</text>
        {/* 右侧输出 */}
        <rect x="132" y="20" width="24" height="14" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="144" y="29" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.75">response</text>
        {/* 连接箭头 */}
        <line x1="22" y1="24" x2="28" y2="24"
          className="illustration__branch illustration__branch--q" />
        <line x1="50" y1="27" x2="56" y2="27"
          className="illustration__branch illustration__branch--q" />
        <line x1="60" y1="27" x2="68" y2="27"
          className="illustration__branch illustration__branch--q" />
        <line x1="126" y1="27" x2="132" y2="27"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="64" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">
          单 linear projection · visual instruction tuning
        </text>
      </g>
    </svg>
  );
}
