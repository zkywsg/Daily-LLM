import { INCEPTION_BRANCHES } from "../lib/data";

const W = 700;
const H = 460;

const BRANCH_COLORS: Record<string, { bg: string; stroke: string }> = {
  conv1x1: { bg: "#fce7f3", stroke: "#ec4899" },
  conv3x3: { bg: "#dbeafe", stroke: "#3b82f6" },
  conv5x5: { bg: "#fef3c7", stroke: "#f59e0b" },
  pool: { bg: "#f3f4f6", stroke: "#9ca3af" },
};

export function InceptionModuleDiagram() {
  const colW = 150;
  const gap = 20;
  const startX = (W - (colW * 4 + gap * 3)) / 2;
  const branchTop = 96;
  const stepH = 38;
  const noteY = 260;
  const concatY = 300;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="Inception block:输入并行走 4 条分支,最后在通道维 concat"
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Inception block — 4 分支并行 + 通道维 concat
      </text>

      {/* 输入 */}
      <rect x={W / 2 - 70} y={40} width={140} height={28} rx={4} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.4} />
      <text x={W / 2} y={58} textAnchor="middle" fontSize={11} fill="var(--ink-primary)">
        输入 x
      </text>

      {INCEPTION_BRANCHES.map((branch, i) => {
        const x = startX + i * (colW + gap);
        const cx = x + colW / 2;
        const color = BRANCH_COLORS[branch.kind];
        const lastStepBottom = branchTop + branch.steps.length * (stepH + 8) - 8;

        return (
          <g key={branch.id}>
            {/* 从输入连到该分支 */}
            <line x1={W / 2} y1={68} x2={cx} y2={branchTop} stroke="#9ca3af" strokeWidth={1.2} />

            <text x={cx} y={branchTop - 8} textAnchor="middle" fontSize={10} fontWeight={700} fill={color.stroke}>
              {branch.label}
            </text>

            {branch.steps.map((step, si) => (
              <g key={`${branch.id}-${si}`}>
                <rect
                  x={x}
                  y={branchTop + si * (stepH + 8)}
                  width={colW}
                  height={stepH}
                  rx={4}
                  fill={color.bg}
                  stroke={color.stroke}
                  strokeWidth={1.4}
                />
                <text
                  x={cx}
                  y={branchTop + si * (stepH + 8) + stepH / 2 + 4}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--ink-primary)"
                >
                  {step}
                </text>
              </g>
            ))}

            {/* 分支说明,置于该分支步骤下方固定行 */}
            <foreignObject x={x - 6} y={noteY} width={colW + 12} height={40}>
              <div
                style={{
                  fontSize: 9,
                  lineHeight: 1.4,
                  color: "var(--ink-muted)",
                  textAlign: "center",
                  fontFamily: "system-ui",
                }}
              >
                {branch.note}
              </div>
            </foreignObject>

            {/* 连到 concat */}
            <line x1={cx} y1={lastStepBottom} x2={cx} y2={concatY - 4} stroke={color.stroke} strokeWidth={1.2} />
          </g>
        );
      })}

      {/* Concat */}
      <rect x={W / 2 - 90} y={concatY} width={180} height={30} rx={5} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.6} />
      <text x={W / 2} y={concatY + 20} textAnchor="middle" fontSize={12} fontWeight={700} fill="#10b981">
        Concat(通道维拼接)
      </text>

      {/* 参数对比 callout */}
      <line x1={W / 2} y1={concatY + 30} x2={W / 2} y2={concatY + 55} stroke="#10b981" strokeWidth={1.2} />
      <rect x={W / 2 - 220} y={concatY + 55} width={440} height={44} rx={5} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} />
      <text x={W / 2} y={concatY + 74} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        5×5 直接算 1.6M vs 1×1 先压到 64 再算 0.2M
      </text>
      <text x={W / 2} y={concatY + 90} textAnchor="middle" fontSize={11} fontWeight={700} fill="#f59e0b">
        降 8× —— 这是 GoogLeNet 5M 整网参数的根本来源
      </text>
    </svg>
  );
}
