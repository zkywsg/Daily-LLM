import type { MiniArchProps } from "./types";

export function MiniBERT({
  width = 160,
  height = 70,
  ariaLabel = "BERT 架构缩图",
}: MiniArchProps) {
  // Encoder-only Transformer:[CLS] + tokens(其中几个被 mask)+ [SEP]
  // 体现双向 attention + MLM
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 中央 encoder stack 12 层 — 双向(无 mask) */}
        {Array.from({ length: 12 }, (_, i) => (
          <rect
            key={i}
            x="40"
            y={8 + i * 3.4}
            width="80"
            height="2.4"
            rx="0.5"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* 输入 tokens 排:[CLS] + 词 + [MASK](高亮)+ 词 + [SEP] */}
        {[
          { x: 40, label: "[CLS]", cls: "illustration__featuremap illustration__featuremap--ctx" },
          { x: 50, label: "tok", cls: "illustration__layer illustration__layer--input" },
          { x: 60, label: "tok", cls: "illustration__layer illustration__layer--input" },
          { x: 70, label: "M", cls: "illustration__proj illustration__proj--v" },
          { x: 80, label: "tok", cls: "illustration__layer illustration__layer--input" },
          { x: 90, label: "M", cls: "illustration__proj illustration__proj--v" },
          { x: 100, label: "tok", cls: "illustration__layer illustration__layer--input" },
          { x: 110, label: "[SEP]", cls: "illustration__featuremap illustration__featuremap--ctx" },
        ].map((t, i) => (
          <rect
            key={i}
            x={t.x}
            y="58"
            width="8"
            height="6"
            rx="1"
            className={t.cls}
          />
        ))}
        {/* 双向 attention 指示:两端往中央的弧线 */}
        <path
          d="M 12 32 Q 26 8, 40 32"
          fill="none"
          className="illustration__residual"
        />
        <path
          d="M 148 32 Q 134 8, 120 32"
          fill="none"
          className="illustration__residual"
        />
        {/* 文字标签 */}
        <text x="14" y="42" fontSize="6" fill="currentColor" opacity="0.6">←→</text>
        <text x="142" y="42" fontSize="6" fill="currentColor" opacity="0.6">←→</text>
      </g>
    </svg>
  );
}
