import type { MiniArchProps } from "./types";

export function MiniDenseNet({
  width = 160,
  height = 70,
  ariaLabel = "DenseNet 架构缩图",
}: MiniArchProps) {
  // 稠密连接：每层接收前面所有层
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {[0, 1, 2, 3, 4].map((i) => (
          <rect key={i} x={i * 32} y="32" width="22" height="14" rx="2"
            className={i === 0 ? "illustration__featuremap" : "illustration__proj illustration__proj--v"} />
        ))}
        {/* 所有跨层连接 */}
        {[
          { from: 0, to: 2 }, { from: 0, to: 3 }, { from: 0, to: 4 },
          { from: 1, to: 3 }, { from: 1, to: 4 }, { from: 2, to: 4 },
        ].map((c, i) => {
          const x1 = c.from * 32 + 11;
          const x2 = c.to * 32 + 11;
          const dy = -8 - (c.to - c.from) * 3;
          return (
            <path key={i} d={`M ${x1} 32 C ${x1} ${dy}, ${x2} ${dy}, ${x2} 32`}
              className="illustration__residual" fill="none" />
          );
        })}
      </g>
    </svg>
  );
}
