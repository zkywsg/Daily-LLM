import type { MiniArchProps } from "./types";

export function MiniQLoRA({
  width = 160,
  height = 70,
  ariaLabel = "QLoRA 架构缩图",
}: MiniArchProps) {
  // 4-bit NF4 base + fp16 LoRA + paged optimizer
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* x */}
        <rect x="2" y="30" width="12" height="10" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="8" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">x</text>
        {/* W₀ 量化为 4-bit NF4 — 压缩外观:窄方块 */}
        <rect x="20" y="12" width="48" height="14" rx="2"
          fill="none"
          strokeDasharray="2 2"
          className="illustration__block" />
        <text x="44" y="18" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.75">W₀  4-bit NF4</text>
        <text x="44" y="23" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.6">❄ 1/4 显存</text>
        {/* LoRA fp16(A + B,小块) */}
        <rect x="20" y="40" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="31" y="46" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">A bf16</text>
        <text x="31" y="51" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">trainable</text>
        <rect x="46" y="40" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="57" y="46" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">B bf16</text>
        <text x="57" y="51" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">trainable</text>
        {/* dequant + 加箭头到 + */}
        <text x="74" y="20" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.65">→ bf16</text>
        {/* + */}
        <circle cx="82" cy="35" r="6"
          className="illustration__proj illustration__proj--v" />
        <text x="82" y="38" textAnchor="middle" fontSize="9" fontWeight="600"
          fill="currentColor" opacity="0.85">+</text>
        {/* h */}
        <rect x="96" y="30" width="14" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="103" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">h</text>
        {/* 右侧:Paged Optimizer */}
        <rect x="118" y="6" width="38" height="22" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="137" y="14" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.75">Paged Opt</text>
        <text x="137" y="20" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.6">GPU ⇄ CPU</text>
        <text x="137" y="25" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.55">AdamW state</text>
        {/* 显存指标 */}
        <rect x="118" y="34" width="38" height="22" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="137" y="42" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">65B</text>
        <text x="137" y="48" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">150 GB → 41 GB</text>
        <text x="137" y="54" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">单卡 48GB OK</text>
        {/* 箭头:x → W₀ + LoRA */}
        <line x1="14" y1="35" x2="20" y2="19"
          className="illustration__branch illustration__branch--q" />
        <line x1="14" y1="35" x2="20" y2="47"
          className="illustration__branch illustration__branch--q" />
        <line x1="68" y1="19" x2="82" y2="29"
          className="illustration__branch illustration__branch--v" />
        <line x1="68" y1="47" x2="82" y2="41"
          className="illustration__branch illustration__branch--v" />
        <line x1="88" y1="35" x2="96" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="60" y="63" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          消费级 GPU 微调 65B
        </text>
      </g>
    </svg>
  );
}
