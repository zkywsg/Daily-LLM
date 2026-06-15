import type { MiniArchProps } from "./types";

export function MiniO1({
  width = 160,
  height = 70,
  ariaLabel = "OpenAI o1 架构缩图",
}: MiniArchProps) {
  // Q → 大块隐藏 reasoning(虚线边框 + 反思/回溯符号) → A
  // 突出"长链推理藏在内部,推理时算力大"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Q */}
        <rect x="4" y="28" width="16" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="12" y="38" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.8">Q</text>
        {/* 隐藏 reasoning 块(虚线 = 用户看不到) */}
        <rect x="28" y="8" width="100" height="40" rx="3"
          fill="none"
          className="illustration__block"
          strokeDasharray="3 2"
          opacity="0.85" />
        <text x="78" y="17" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.7">{'<reasoning>  hidden'}</text>
        {/* 推理路径:线性 + 一个回溯弧线 */}
        <rect x="34" y="22" width="14" height="9" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="41" y="29" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.8">try</text>
        <rect x="54" y="22" width="14" height="9" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="61" y="29" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.8">wait</text>
        <rect x="74" y="22" width="14" height="9" rx="1"
          className="illustration__proj illustration__proj--ffn" opacity="0.55" />
        <text x="81" y="29" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.65">↩ back</text>
        <rect x="94" y="22" width="14" height="9" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="101" y="29" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.8">retry</text>
        <rect x="114" y="22" width="12" height="9" rx="1"
          className="illustration__proj illustration__proj--v" />
        <text x="120" y="29" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">verify</text>
        {/* 路径连接 */}
        <line x1="48" y1="26" x2="54" y2="26"
          className="illustration__branch illustration__branch--q" />
        <line x1="68" y1="26" x2="74" y2="26"
          className="illustration__branch illustration__branch--q" />
        <line x1="88" y1="26" x2="94" y2="26"
          className="illustration__branch illustration__branch--q" />
        <line x1="108" y1="26" x2="114" y2="26"
          className="illustration__branch illustration__branch--v" />
        {/* thinking tokens 标签 */}
        <text x="78" y="44" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.65">~10⁴ thinking tokens</text>
        {/* Q → reasoning */}
        <line x1="20" y1="35" x2="28" y2="28"
          className="illustration__branch illustration__branch--q" />
        {/* A */}
        <rect x="136" y="28" width="20" height="14" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="146" y="38" textAnchor="middle" fontSize="7" fontWeight="600"
          fill="currentColor" opacity="0.85">A</text>
        <line x1="128" y1="35" x2="136" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="62" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">
          RL 训练 · 反思/回溯/自验证内化为模型行为
        </text>
      </g>
    </svg>
  );
}
