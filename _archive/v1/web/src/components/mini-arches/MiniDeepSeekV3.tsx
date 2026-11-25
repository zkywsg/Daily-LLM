import type { MiniArchProps } from "./types";

export function MiniDeepSeekV3({
  width = 160,
  height = 70,
  ariaLabel = "DeepSeek-V3 架构缩图",
}: MiniArchProps) {
  // Shared expert(全员走)+ fine-grained routed experts(256 选 8)
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
        {/* Shared expert(上方,所有 token 走) */}
        <rect x="22" y="4" width="32" height="14" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="38" y="11" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.9">shared</text>
        <text x="38" y="16" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">(all tokens)</text>
        {/* Router + aux-free bias */}
        <rect x="22" y="26" width="32" height="20" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="38" y="33" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">Router</text>
        <text x="38" y="39" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.7">+ bias bᵢ</text>
        <text x="38" y="44" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.6">aux-free</text>
        {/* 256 细 expert 网格(8×8 小方块,~8 个高亮) */}
        <g>
          {Array.from({ length: 64 }).map((_, i) => {
            const col = i % 8;
            const row = Math.floor(i / 8);
            const cx = 64 + col * 4.2;
            const cy = 6 + row * 4.2;
            // 选 8 个分散的位置高亮
            const active = [3, 10, 18, 27, 33, 42, 51, 58].includes(i);
            return active
              ? <rect key={i} x={cx} y={cy} width="3.4" height="3.4" rx="0.4"
                  className="illustration__featuremap illustration__featuremap--ctx" />
              : <rect key={i} x={cx} y={cy} width="3.4" height="3.4" rx="0.4"
                  className="illustration__proj illustration__proj--ffn" opacity="0.25" />;
          })}
        </g>
        <text x="80" y="2" textAnchor="middle" fontSize="4" fontWeight="600"
          fill="currentColor" opacity="0.7">256 fine-grained experts</text>
        <text x="80" y="44" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.65">top-8 (highlighted)</text>
        {/* x → shared + router */}
        <line x1="14" y1="35" x2="22" y2="14"
          className="illustration__branch illustration__branch--q" />
        <line x1="14" y1="35" x2="22" y2="35"
          className="illustration__branch illustration__branch--q" />
        {/* Router → 网格 */}
        <line x1="54" y1="35" x2="64" y2="22"
          className="illustration__branch illustration__branch--q" />
        {/* 网格 + shared → Σ */}
        <rect x="100" y="26" width="20" height="20" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="110" y="39" textAnchor="middle" fontSize="9" fontWeight="600"
          fill="currentColor" opacity="0.85">Σ</text>
        <line x1="54" y1="11" x2="100" y2="30"
          className="illustration__branch illustration__branch--v" />
        <line x1="98" y1="22" x2="100" y2="32"
          className="illustration__branch illustration__branch--v" />
        {/* y */}
        <rect x="128" y="30" width="12" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="134" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">y</text>
        <line x1="120" y1="35" x2="128" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="55" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.55">
          671B / 37B 激活 · MLA · MTP · FP8 训练
        </text>
        <text x="80" y="62" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.5">
          训练成本 $5.6M · R1 的 base
        </text>
      </g>
    </svg>
  );
}
