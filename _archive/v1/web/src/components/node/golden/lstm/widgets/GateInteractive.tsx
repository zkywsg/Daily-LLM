import { lstmStep } from "../lib/lstm";

interface Props {
  f: number;
  i: number;
  o: number;
  g: number;
  prevC: number;
}

const W = 700;
const H = 280;

// viewer 拖 f/i/o/g/prevC sliders,实时看 newC 和 newH 怎么算出来。
// 数学:newC = f·prevC + i·g; newH = o·tanh(newC)
// 用 4 个竖条 + 2 个结果条可视化每一项的贡献。

export function GateInteractive({ f, i, o, g, prevC }: Props) {
  const step = lstmStep(f, i, o, g, prevC, 0);

  const PAD = 30;
  const cellW = 80;
  const gap = 12;
  const baseY = 200;
  const maxH = 130;

  const renderBar = (idx: number, value: number, label: string, color: string, sub?: string) => {
    const x = PAD + idx * (cellW + gap);
    const h = Math.min(maxH, Math.abs(value) * maxH);
    const isNeg = value < 0;
    return (
      <g key={label}>
        <rect x={x} y={isNeg ? baseY : baseY - h} width={cellW} height={h} rx={4} fill={color} opacity={0.85} />
        <text x={x + cellW / 2} y={baseY + (isNeg ? h + 14 : -h - 6)} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
          {value.toFixed(2)}
        </text>
        <text x={x + cellW / 2} y={baseY + 28} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ink-primary)">
          {label}
        </text>
        {sub && (
          <text x={x + cellW / 2} y={baseY + 42} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
            {sub}
          </text>
        )}
      </g>
    );
  };

  // baseline 0 line
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`LSTM step: f=${f.toFixed(2)} i=${i.toFixed(2)} o=${o.toFixed(2)} g=${g.toFixed(2)}`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        单步 LSTM:newC = f·prevC + i·g · newH = o·tanh(newC)
      </text>

      {/* 0 线 */}
      <line x1={PAD - 4} x2={W - PAD + 4} y1={baseY} y2={baseY} stroke="var(--ink-muted)" strokeDasharray="2 3" />

      {renderBar(0, prevC, "prevC", "#fce7f3", "上一步 cell")}
      {renderBar(1, f, "f", "#f59e0b", "forget 0-1")}
      {renderBar(2, i, "i", "#ec4899", "input 0-1")}
      {renderBar(3, g, "g", "#3b82f6", "candidate ±1")}
      {renderBar(4, o, "o", "#10b981", "output 0-1")}

      {/* 分隔 */}
      <line x1={PAD + 5 * (cellW + gap) - gap / 2} y1={baseY - maxH - 20} x2={PAD + 5 * (cellW + gap) - gap / 2} y2={baseY + 50} stroke="var(--border)" strokeDasharray="3 3" />

      {/* 结果 */}
      {renderBar(5, step.newC, "newC", "#fce7f3", `= ${(f * prevC).toFixed(2)} + ${(i * g).toFixed(2)}`)}
      {renderBar(6, step.newH, "newH", "#ecfdf5", `= o·tanh(C)`)}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        拖左侧 sliders 看 cell state 和 hidden 怎么由 4 个门联合生成
      </text>
    </svg>
  );
}
