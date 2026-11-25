import { useState } from "react";

const W = 700;
const H = 320;

// 交互式对比:1 个大 kernel(5×5 / 7×7)vs 堆叠小 kernel(2×3×3 / 3×3×3)
// 用同心方块示意感受野在原图上的等效覆盖范围完全相同,但堆叠版本层数更多、参数更少。
export function KernelStackingDiagram() {
  const [mode, setMode] = useState<"large" | "stacked">("large");
  const isStacked = mode === "stacked";

  const cx = 350;
  const cy = 150;
  const cell = 12;

  // 7×7 感受野,用 7x7 网格表示原图上被覆盖的像素范围
  const rf = 7;
  const gridStartX = cx - (rf * cell) / 2;
  const gridStartY = cy - (rf * cell) / 2;

  // 堆叠版本:3 层 3×3,依次收缩的三个方框示意每一层各自的卷积核范围
  const layers = isStacked ? 3 : 1;
  const layerSize = isStacked ? 3 : 7;

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
        role="img"
        aria-label={isStacked ? "3 层 3×3 卷积堆叠,等价 7×7 感受野" : "1 层 7×7 卷积"}
      >
        <text x={W / 2} y={24} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          {isStacked ? "3 层 3×3 堆叠 — 等价 7×7 感受野" : "1 层 7×7 卷积"}
        </text>

        {/* 原图像素网格,覆盖 7x7 感受野范围外扩一圈做背景 */}
        {Array.from({ length: rf + 2 }).map((_, r) =>
          Array.from({ length: rf + 2 }).map((_, c) => (
            <rect
              key={`bg-${r}-${c}`}
              x={gridStartX - cell + c * cell}
              y={gridStartY - cell + r * cell}
              width={cell - 1}
              height={cell - 1}
              fill="#f3f4f6"
              stroke="#e5e7eb"
              strokeWidth={0.6}
            />
          ))
        )}

        {/* 感受野覆盖范围(两种模式下完全一致,7×7) */}
        <rect
          x={gridStartX}
          y={gridStartY}
          width={rf * cell}
          height={rf * cell}
          fill="#fce7f3"
          stroke="#ec4899"
          strokeWidth={2}
        />

        {isStacked
          ? // 层层收缩的方框:第 1 层(最外)→ 第 3 层(最内),示意堆叠的 3 次 3×3 卷积
            Array.from({ length: layers }).map((_, i) => {
              const inset = i * cell * 2;
              const size = rf * cell - inset * 2;
              return (
                <rect
                  key={`stack-${i}`}
                  x={gridStartX + inset}
                  y={gridStartY + inset}
                  width={size}
                  height={size}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth={1.6}
                  strokeDasharray={i === layers - 1 ? "0" : "4 2"}
                />
              );
            })
          : (
              <rect
                x={gridStartX}
                y={gridStartY}
                width={rf * cell}
                height={rf * cell}
                fill="none"
                stroke="#3b82f6"
                strokeWidth={2}
              />
            )}

        <text x={cx} y={gridStartY - 16} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          原图等效感受野 7×7(两种方式完全相同)
        </text>

        <text x={cx} y={H - 60} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
          {isStacked ? "深度 +2 层 · 非线性(ReLU) +2 次" : "深度 +0 · 非线性 +0"}
        </text>
        <text x={cx} y={H - 36} textAnchor="middle" fontSize={16} fontWeight={700} fill={isStacked ? "#3b82f6" : "#ec4899"}>
          {isStacked ? "27C² 参数(−45%)" : "49C² 参数"}
        </text>
      </svg>

      <div style={{ display: "flex", gap: 6, marginTop: 8, justifyContent: "center" }}>
        <button
          type="button"
          onClick={() => setMode("large")}
          style={{
            padding: "4px 12px",
            fontSize: "var(--fs-sm)",
            borderRadius: "var(--radius-sm)",
            border: `1px solid ${!isStacked ? "var(--accent-link)" : "var(--border)"}`,
            background: !isStacked ? "var(--accent-link)" : "var(--bg-surface)",
            color: !isStacked ? "var(--bg-surface)" : "var(--ink-secondary)",
            cursor: "pointer",
          }}
        >
          1 层 7×7
        </button>
        <button
          type="button"
          onClick={() => setMode("stacked")}
          style={{
            padding: "4px 12px",
            fontSize: "var(--fs-sm)",
            borderRadius: "var(--radius-sm)",
            border: `1px solid ${isStacked ? "var(--accent-link)" : "var(--border)"}`,
            background: isStacked ? "var(--accent-link)" : "var(--bg-surface)",
            color: isStacked ? "var(--bg-surface)" : "var(--ink-secondary)",
            cursor: "pointer",
          }}
        >
          3 层 3×3 堆叠
        </button>
      </div>
    </div>
  );
}
