import { useState } from "react";

const W = 700;
const H = 340;

export function LocalVsGlobalConnectionDiagram() {
  const [mode, setMode] = useState<"conv" | "mlp">("conv");
  const isConv = mode === "conv";

  // 简化的 8x8 输入网格,示意用
  const grid = 8;
  const cell = 16;
  const gridX = 60;
  const gridY = 50;

  // 局部感受野(conv):以 (3,3) 为中心的 3x3 patch
  const center = 3;
  const patch = [center - 1, center, center + 1];

  const hiddenX = 420;
  const hiddenY0 = 60;
  const hiddenGap = 26;
  const hiddenCount = 6;

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
        role="img"
        aria-label={isConv ? "卷积:局部连接 + 权值共享" : "MLP:每个像素连到每个隐藏单元"}
      >
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          {isConv ? "卷积 — 局部连接 + 权值共享" : "MLP — 压平图像,全连接"}
        </text>

        {/* 输入网格 */}
        {Array.from({ length: grid }).map((_, r) =>
          Array.from({ length: grid }).map((_, c) => {
            const isInPatch = isConv && patch.includes(r) && patch.includes(c);
            return (
              <rect
                key={`${r}-${c}`}
                x={gridX + c * cell}
                y={gridY + r * cell}
                width={cell - 1.5}
                height={cell - 1.5}
                fill={isInPatch ? "#fce7f3" : "#f3f4f6"}
                stroke={isInPatch ? "#ec4899" : "#e5e7eb"}
                strokeWidth={isInPatch ? 1.6 : 1}
              />
            );
          })
        )}
        <text x={gridX + (grid * cell) / 2} y={gridY - 10} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          输入图像(像素网格)
        </text>

        {/* 隐藏单元 */}
        {Array.from({ length: hiddenCount }).map((_, i) => (
          <circle
            key={i}
            cx={hiddenX}
            cy={hiddenY0 + i * hiddenGap}
            r={9}
            fill="#dbeafe"
            stroke="#3b82f6"
            strokeWidth={1.6}
          />
        ))}
        <text x={hiddenX} y={hiddenY0 - 20} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          {isConv ? "输出特征图(1 个卷积核)" : "隐藏层(1024 → hidden)"}
        </text>

        {/* 连接线 */}
        {isConv ? (
          // 只有 patch 内的像素连到唯一一个输出神经元(第一个隐藏单元代表卷积核输出的一个位置)
          patch.flatMap((r) =>
            patch.map((c) => (
              <line
                key={`conv-${r}-${c}`}
                x1={gridX + c * cell + cell / 2}
                y1={gridY + r * cell + cell / 2}
                x2={hiddenX - 10}
                y2={hiddenY0}
                stroke="#ec4899"
                strokeWidth={1}
                opacity={0.5}
              />
            ))
          )
        ) : (
          // 每个隐藏单元连接到全部像素(用稀疏采样示意,否则线条过密)
          Array.from({ length: hiddenCount }).flatMap((_, hi) =>
            Array.from({ length: grid }).flatMap((_, r) =>
              (r % 2 === 0
                ? Array.from({ length: grid }).filter((_, c) => c % 2 === 0)
                : []
              ).map((_, ci) => {
                const c = ci * 2;
                return (
                  <line
                    key={`mlp-${hi}-${r}-${c}`}
                    x1={gridX + c * cell + cell / 2}
                    y1={gridY + r * cell + cell / 2}
                    x2={hiddenX - 10}
                    y2={hiddenY0 + hi * hiddenGap}
                    stroke="#3b82f6"
                    strokeWidth={0.5}
                    opacity={0.15}
                  />
                );
              })
            )
          )
        )}

        <text x={W - 40} y={H - 60} textAnchor="end" fontSize={10} fill="var(--ink-secondary)">
          {isConv ? "5×5 卷积核" : "1024 → 1024 隐层"}
        </text>
        <text x={W - 40} y={H - 44} textAnchor="end" fontSize={16} fontWeight={700} fill={isConv ? "#ec4899" : "#3b82f6"}>
          {isConv ? "≈ 25 个权重(每个卷积核,全图共享)" : "≈ 1,048,576 个权重(仅这一层)"}
        </text>
      </svg>

      <div style={{ display: "flex", gap: 6, marginTop: 8, justifyContent: "center" }}>
        <button
          type="button"
          onClick={() => setMode("conv")}
          style={{
            padding: "4px 12px",
            fontSize: "var(--fs-sm)",
            borderRadius: "var(--radius-sm)",
            border: `1px solid ${isConv ? "var(--accent-link)" : "var(--border)"}`,
            background: isConv ? "var(--accent-link)" : "var(--bg-surface)",
            color: isConv ? "var(--bg-surface)" : "var(--ink-secondary)",
            cursor: "pointer",
          }}
        >
          卷积(局部 + 共享)
        </button>
        <button
          type="button"
          onClick={() => setMode("mlp")}
          style={{
            padding: "4px 12px",
            fontSize: "var(--fs-sm)",
            borderRadius: "var(--radius-sm)",
            border: `1px solid ${!isConv ? "var(--accent-link)" : "var(--border)"}`,
            background: !isConv ? "var(--accent-link)" : "var(--bg-surface)",
            color: !isConv ? "var(--bg-surface)" : "var(--ink-secondary)",
            cursor: "pointer",
          }}
        >
          MLP(全连接)
        </button>
      </div>
    </div>
  );
}
