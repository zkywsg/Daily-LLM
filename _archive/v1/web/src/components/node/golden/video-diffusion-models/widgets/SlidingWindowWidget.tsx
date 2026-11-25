import { useState } from "react";
import { slidingWindow } from "../lib/data";

const WINDOW_SIZE = 8;
const TOTAL_CELLS = 24;

// 滑动窗口自回归扩展:每点一次"生成下一窗口",总帧数增加,
// 窗口覆盖的区间随之前移 —— 展示 reconstruction guidance 如何
// 用已生成帧作条件继续扩展视频长度。

export function SlidingWindowWidget() {
  const [generated, setGenerated] = useState(WINDOW_SIZE);
  const { start, end } = slidingWindow(WINDOW_SIZE, generated);

  const cellW = 24;

  return (
    <div>
      <svg viewBox={`0 0 ${TOTAL_CELLS * cellW + 20} 80`} style={{ width: "100%", height: "auto" }} role="img" aria-label={`已生成 ${generated} 帧,当前窗口覆盖 ${start} 到 ${end}`}>
        {Array.from({ length: TOTAL_CELLS }, (_, i) => {
          const inGenerated = i < generated;
          const inWindow = i >= start && i < end;
          const fill = inWindow ? "#d946ef" : inGenerated ? "#f5d0fe" : "var(--bg-subtle)";
          return <rect key={i} x={10 + i * cellW} y={20} width={cellW - 2} height={30} fill={fill} rx={2} />;
        })}
        <text x={10} y={70} fontSize={10} fill="var(--ink-muted)">已生成 {generated} / {TOTAL_CELLS} 帧,当前窗口 [{start}, {end})</text>
      </svg>
      <div style={{ display: "flex", gap: 8, marginTop: "var(--space-2)" }}>
        <button
          type="button"
          onClick={() => setGenerated((g) => Math.min(g + 4, TOTAL_CELLS))}
          disabled={generated >= TOTAL_CELLS}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: generated >= TOTAL_CELLS ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: generated >= TOTAL_CELLS ? 0.5 : 1 }}
        >
          生成下一窗口
        </button>
        <button
          type="button"
          onClick={() => setGenerated(WINDOW_SIZE)}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        深粉色 = 当前生成窗口;浅粉色 = 之前已生成、现在作为条件的帧。窗口不断前移,视频长度随之延长。
      </p>
    </div>
  );
}
