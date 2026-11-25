import { MEMORY_TASK_SEQ } from "../lib/data";

const W = 700;
const H = 240;

interface Props {
  distance: number; // 模拟"记住"能力随距离衰减
}

export function MemoryTaskDemo({ distance }: Props) {
  const TILE_W = 90;
  const gap = 10;
  const startX = (W - (MEMORY_TASK_SEQ.length * (TILE_W + gap) - gap)) / 2;
  const y = 80;

  // 记忆强度随 distance 衰减（模拟)
  const retention = Math.max(0, 1 - distance / 8);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RNN memory task demo">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        序列记忆任务 — RNN 能记住多远的历史?
      </text>

      {MEMORY_TASK_SEQ.map((tok, i) => {
        const isTarget = i === 0;
        const isQuery = i === MEMORY_TASK_SEQ.length - 1;
        return (
          <g key={i}>
            <rect x={startX + i * (TILE_W + gap)} y={y} width={TILE_W} height={36} rx={4}
                  fill={isTarget ? "#fce7f3" : isQuery ? "#fef3c7" : "#f3f4f6"}
                  stroke={isTarget ? "#ec4899" : isQuery ? "#f59e0b" : "#d1d5db"} strokeWidth={isTarget || isQuery ? 2 : 1} />
            <text x={startX + i * (TILE_W + gap) + TILE_W / 2} y={y + 23} textAnchor="middle" fontSize={12} fontWeight={isTarget ? 700 : 500} fill="#1f2937">
              {tok}
            </text>
          </g>
        );
      })}

      {/* memory retention bar from target to query */}
      <rect x={startX} y={140} width={distance * (TILE_W + gap)} height={16} rx={3}
            fill="#ec4899" fillOpacity={retention} stroke="#ec4899" strokeWidth={1} />
      <text x={startX + (distance * (TILE_W + gap)) / 2} y={152} textAnchor="middle" fontSize={9} fill="#831843">
        记忆强度 {(retention * 100).toFixed(0)}%
      </text>

      <text x={W / 2} y={190} textAnchor="middle" fontSize={12} fontWeight={700} fill={retention > 0.5 ? "#065f46" : "#831843"}>
        {retention > 0.5 ? "还能回忆起 'cat'" : retention > 0.1 ? "回忆模糊" : "已经忘记了 'cat'"}
      </text>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        距离 = {distance} 步 · 简单 RNN 在 T&gt;10-20 后长依赖学习困难(Bengio 1994)
      </text>
    </svg>
  );
}
