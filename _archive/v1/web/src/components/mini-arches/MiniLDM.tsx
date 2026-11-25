import type { MiniArchProps } from "./types";

export function MiniLDM({
  width = 160,
  height = 70,
  ariaLabel = "LDM / Stable Diffusion 架构缩图",
}: MiniArchProps) {
  // VAE encoder → 小 latent + diffusion 在 latent 上 → VAE decoder
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:大图像 */}
        <rect x="4" y="14" width="24" height="24" rx="2"
          className="illustration__layer illustration__layer--input" />
        {/* VAE Encoder(梯形) */}
        <path d="M 28 18 L 40 26 L 40 34 L 28 38 Z"
          className="illustration__proj illustration__proj--ffn" />
        {/* Latent(小方块) */}
        <rect x="44" y="24" width="10" height="10" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        {/* Diffusion stack(中央 5 层小条) */}
        {Array.from({ length: 5 }, (_, i) => (
          <rect key={i} x="62" y={20 + i * 4} width="38" height="2.4" rx="0.5"
            className="illustration__proj illustration__proj--v" />
        ))}
        {/* 生成的 latent */}
        <rect x="106" y="24" width="10" height="10" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        {/* VAE Decoder(梯形) */}
        <path d="M 116 26 L 128 18 L 128 38 L 116 34 Z"
          className="illustration__proj illustration__proj--ffn" />
        {/* 右侧:生成图像 */}
        <rect x="128" y="14" width="24" height="24" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        {/* 连接箭头 */}
        <line x1="54" y1="29" x2="62" y2="29"
          className="illustration__branch illustration__branch--q" />
        <line x1="100" y1="29" x2="106" y2="29"
          className="illustration__branch illustration__branch--q" />
        {/* 底部:64× compress 标 */}
        <text x="49" y="50" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.7">VAE↓</text>
        <text x="80" y="50" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.75">latent diff</text>
        <text x="122" y="50" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.7">VAE↑</text>
        <text x="80" y="64" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.55">
          64× 压缩 · 消费 GPU 可跑
        </text>
      </g>
    </svg>
  );
}
