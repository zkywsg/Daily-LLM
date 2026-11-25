import { buildTileGrid } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  activeI: number;
  activeJ: number;
}

const TR = 4;
const TC = 4;

export function TilingDiagram({ activeI, activeJ }: Props) {
  const grid = buildTileGrid(TR, TC);
  const cell = 56;
  const PAD_L = 120;
  const PAD_T = 70;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Q/K/V tiling 分块示意">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Tiling — N×N 矩阵从不整体物化,逐块在 SRAM 里算
      </text>

      <text x={PAD_L - 12} y={PAD_T - 14} textAnchor="end" fontSize={10} fontWeight={700} fill="#6b7280">Q 行块(B_r)</text>
      <text x={PAD_L + (TC * cell) / 2} y={PAD_T - 30} textAnchor="middle" fontSize={10} fontWeight={700} fill="#6b7280">K/V 列块(B_c)</text>

      {grid.map(({ i, j }) => {
        const x = PAD_L + j * cell;
        const y = PAD_T + i * cell;
        const isActive = i === activeI && j === activeJ;
        const isDone = i < activeI || (i === activeI && j < activeJ);
        const fill = isActive ? "#fef3c7" : isDone ? "#dbeafe" : "var(--bg-surface)";
        const stroke = isActive ? "#f59e0b" : isDone ? "#3b82f6" : "var(--border)";
        return (
          <g key={`${i}-${j}`}>
            <rect x={x} y={y} width={cell - 3} height={cell - 3} fill={fill} stroke={stroke} strokeWidth={isActive ? 2.4 : 1} rx={4} />
            {isActive && (
              <text x={x + (cell - 3) / 2} y={y + (cell - 3) / 2 + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="#b45309">S_ij</text>
            )}
          </g>
        );
      })}

      <text x={PAD_L - 12} y={PAD_T + activeI * cell + (cell - 3) / 2 + 4} textAnchor="end" fontSize={10} fill="#f59e0b" fontWeight={700}>Q_{activeI}</text>
      <text x={PAD_L + activeJ * cell + (cell - 3) / 2} y={PAD_T - 46} textAnchor="middle" fontSize={10} fill="#f59e0b" fontWeight={700}>K_{activeJ}/V_{activeJ}</text>

      <text x={W / 2} y={H - 40} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        黄色 = 当前在 SRAM 内处理的 (Q_i, K_j, V_j) 块;蓝色 = 已处理完、结果已 rescale 进 O_i,不再占用 SRAM
      </text>
      <text x={W / 2} y={H - 22} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        整个 N×N 矩阵从未在 HBM 上以完整形式存在,显存从 O(N²) 降到 O(N)
      </text>
    </svg>
  );
}
