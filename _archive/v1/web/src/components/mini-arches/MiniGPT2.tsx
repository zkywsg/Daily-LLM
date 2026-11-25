import type { MiniArchProps } from "./types";

export function MiniGPT2({
  width = 160,
  height = 70,
  ariaLabel = "GPT-2 架构缩图",
}: MiniArchProps) {
  // 和 GPT-1 同形但更深(48 层) + 更宽(d_model 1600)
  // 视觉:塔变得更高,且层条更宽 — 体现"GPT-2 是 GPT-1 的 13× 放大"
  const layers = 24; // 48 层在 70px 内画不下,用 24 表意,密度感更强
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 中央 decoder stack — 更高、更密 */}
        {Array.from({ length: layers }, (_, i) => (
          <rect
            key={i}
            x="56"
            y={6 + i * 2.4}
            width="48"
            height="1.8"
            rx="0.5"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* 输入/输出 */}
        <rect
          x="56"
          y="62"
          width="48"
          height="5"
          rx="2"
          className="illustration__layer illustration__layer--input"
        />
        <rect
          x="56"
          y="1"
          width="48"
          height="3"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        {/* 左侧:更长的 prompt(体现 1024 上下文 vs GPT-1 的 512) */}
        {[16, 24, 32, 40, 48].map((y, i) => (
          <rect
            key={i}
            x="4"
            y={y}
            width="14"
            height="5"
            rx="1"
            className="illustration__layer illustration__layer--input"
            opacity={0.85 - i * 0.12}
          />
        ))}
        <line
          x1="20"
          y1="34"
          x2="56"
          y2="34"
          className="illustration__branch illustration__branch--q"
        />
        {/* 右侧:多 token 生成(体现 zero-shot 长输出能力) */}
        {[28, 36].map((y, i) => (
          <rect
            key={i}
            x="142"
            y={y}
            width="14"
            height="4"
            rx="1"
            className="illustration__featuremap illustration__featuremap--ctx"
            opacity={0.95 - i * 0.2}
          />
        ))}
        <line
          x1="106"
          y1="34"
          x2="142"
          y2="34"
          className="illustration__branch illustration__branch--v"
        />
      </g>
    </svg>
  );
}
