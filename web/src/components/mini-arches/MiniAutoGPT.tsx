import type { MiniArchProps } from "./types";

export function MiniAutoGPT({
  width = 160,
  height = 70,
  ariaLabel = "AutoGPT 架构缩图",
}: MiniArchProps) {
  // Goal → 任务分解 → 循环 (execute + critique) → 直到完成
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Goal */}
        <rect x="2" y="28" width="20" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="12" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">Goal</text>
        {/* Task queue 竖排 */}
        <rect x="28" y="8" width="32" height="54" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="44" y="15" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">task queue</text>
        <rect x="31" y="18" width="26" height="7" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="44" y="23.5" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">① research</text>
        <rect x="31" y="27" width="26" height="7" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="44" y="32.5" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">② analyze</text>
        <rect x="31" y="36" width="26" height="7" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="44" y="41.5" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">③ draft</text>
        <rect x="31" y="45" width="26" height="7" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="44" y="50.5" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">④ save</text>
        <text x="44" y="59" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.5">(dynamic)</text>
        {/* Execute 块(LLM + tools) */}
        <rect x="66" y="14" width="36" height="42" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="84" y="26" textAnchor="middle" fontSize="5.5" fontWeight="600"
          fill="currentColor" opacity="0.85">LLM</text>
        <text x="84" y="34" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.75">+ tools</text>
        <text x="84" y="42" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.75">+ memory</text>
        <text x="84" y="50" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.75">+ critique</text>
        {/* 循环箭头(回到 task queue) */}
        <path d="M 66 18 Q 60 8 60 18"
          fill="none"
          className="illustration__branch illustration__branch--q" />
        <path d="M 66 52 Q 60 62 60 52"
          fill="none"
          className="illustration__branch illustration__branch--v" />
        <text x="62" y="9" fontSize="4.5"
          fill="currentColor" opacity="0.65">replan ↺</text>
        {/* Output */}
        <rect x="108" y="22" width="22" height="26" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="119" y="32" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">report</text>
        <text x="119" y="40" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">.md</text>
        {/* 工具集 */}
        <rect x="136" y="14" width="22" height="10" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="147" y="21" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.75">search</text>
        <rect x="136" y="26" width="22" height="10" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="147" y="33" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.75">browse</text>
        <rect x="136" y="38" width="22" height="10" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="147" y="45" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.75">file</text>
        <rect x="136" y="50" width="22" height="10" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="147" y="57" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.75">code</text>
        {/* 箭头 */}
        <line x1="22" y1="35" x2="28" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="60" y1="35" x2="66" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="102" y1="35" x2="108" y2="35"
          className="illustration__branch illustration__branch--v" />
        <line x1="102" y1="20" x2="136" y2="20"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--q" opacity="0.6" />
      </g>
    </svg>
  );
}
