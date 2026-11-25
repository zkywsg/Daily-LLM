import type { MiniArchProps } from "./types";

export function MiniMixtral({
  width = 160,
  height = 70,
  ariaLabel = "Mixtral 8×7B 架构缩图",
}: MiniArchProps) {
  // 8 个大 expert + top-2;突出"开源"
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
        <rect x="2" y="30" width="14" height="10" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="9" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">x</text>
        {/* Router */}
        <rect x="22" y="24" width="22" height="22" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="33" y="33" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">Router</text>
        <text x="33" y="41" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">top-2</text>
        {/* 8 个 expert(大块,SwiGLU),2 个高亮 */}
        {[
          { y: 6, active: false, label: "E₁" },
          { y: 14, active: true, label: "E₂" },
          { y: 22, active: false, label: "E₃" },
          { y: 30, active: false, label: "E₄" },
          { y: 38, active: false, label: "E₅" },
          { y: 46, active: true, label: "E₆" },
          { y: 54, active: false, label: "E₇" },
          { y: 62, active: false, label: "E₈" },
        ].map((e, i) => (
          <g key={i}>
            {e.active
              ? <rect x="52" y={e.y} width="44" height="7" rx="1.5"
                  className="illustration__featuremap illustration__featuremap--ctx" />
              : <rect x="52" y={e.y} width="44" height="7" rx="1.5"
                  className="illustration__proj illustration__proj--ffn" opacity="0.3" />
            }
            <text x="74" y={e.y + 5.5} textAnchor="middle" fontSize="3.8"
              fill="currentColor" opacity={e.active ? 0.85 : 0.5}
              fontWeight={e.active ? 600 : 400}>
              {e.label} (7B)
            </text>
          </g>
        ))}
        <text x="74" y="3" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">8 experts (SwiGLU)</text>
        {/* 2 条 active 箭头 */}
        <line x1="44" y1="35" x2="52" y2="17"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="35" x2="52" y2="49"
          className="illustration__branch illustration__branch--q" />
        {/* Σ + y */}
        <rect x="102" y="26" width="20" height="18" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="112" y="38" textAnchor="middle" fontSize="8" fontWeight="600"
          fill="currentColor" opacity="0.85">Σ</text>
        <line x1="96" y1="17" x2="102" y2="30"
          className="illustration__branch illustration__branch--v" />
        <line x1="96" y1="49" x2="102" y2="40"
          className="illustration__branch illustration__branch--v" />
        <rect x="130" y="30" width="14" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="137" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">y</text>
        <line x1="122" y1="35" x2="130" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="69" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          46.7B 总参 / 13B 激活 · 开源 MoE 起点
        </text>
      </g>
    </svg>
  );
}
