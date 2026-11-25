import type { MiniArchProps } from "./types";

export function MiniGPT3({
  width = 160,
  height = 70,
  ariaLabel = "GPT-3 架构缩图",
}: MiniArchProps) {
  // In-context learning 可视化:
  // 左侧 3 对 (input, output) few-shot 例子 + 1 个查询 + 1 个生成答案
  // 中间一个粗块代表"权重不动的 LLM"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 中央 LLM 块 — 一整个粗块,表示"庞大且权重冻结" */}
        <rect
          x="62"
          y="22"
          width="36"
          height="26"
          rx="3"
          className="illustration__proj illustration__proj--v"
        />
        <text
          x="80"
          y="38"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.75"
        >
          LLM
        </text>
        {/* 左侧 3 对 few-shot 例子 + 1 个 query */}
        {[
          { y: 10, type: "ex" },
          { y: 22, type: "ex" },
          { y: 34, type: "ex" },
          { y: 50, type: "query" },
        ].map((row, i) => (
          <g key={i}>
            {/* 输入 token(琥珀) */}
            <rect
              x="4"
              y={row.y}
              width="10"
              height="6"
              rx="1"
              className="illustration__layer illustration__layer--input"
            />
            <text
              x="9"
              y={row.y + 5}
              textAnchor="middle"
              fontSize="5"
              fill="currentColor"
              opacity="0.6"
            >
              x
            </text>
            {/* 输出 token(few-shot 例子是绿,query 留空让 LLM 填) */}
            {row.type === "ex" ? (
              <rect
                x="20"
                y={row.y}
                width="10"
                height="6"
                rx="1"
                className="illustration__featuremap illustration__featuremap--ctx"
              />
            ) : (
              <rect
                x="20"
                y={row.y}
                width="10"
                height="6"
                rx="1"
                fill="none"
                strokeDasharray="2,1"
                className="illustration__block"
              />
            )}
          </g>
        ))}
        {/* 左侧整体 → LLM 的箭头 */}
        <line
          x1="32"
          y1="35"
          x2="62"
          y2="35"
          className="illustration__branch illustration__branch--q"
        />
        {/* LLM → 右侧生成的答案 */}
        <line
          x1="98"
          y1="35"
          x2="138"
          y2="53"
          className="illustration__branch illustration__branch--v"
        />
        <rect
          x="138"
          y="50"
          width="14"
          height="6"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
      </g>
    </svg>
  );
}
