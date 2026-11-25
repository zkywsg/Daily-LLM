import { CASCADE_STAGES } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  highlightIdx: number;
}

function Arrow({ x1, y1, x2, y2, color, id }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function CascadePipeline({ highlightIdx }: Props) {
  const boxW = 140;
  const gap = 40;
  const startX = 40;
  const y = 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Imagen cascade diffusion pipeline">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        三级 Cascade Diffusion — 64² → 256² → 1024²
      </text>

      {/* T5-XXL 起点 */}
      <rect x={startX} y={40} width={boxW} height={40} rx={4} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} />
      <text x={startX + boxW / 2} y={64} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">T5-XXL(冻结)</text>

      <Arrow x1={startX + boxW / 2} y1={80} x2={startX + boxW / 2} y2={y - 5} color="#3b82f6" id="cp-a0" />

      {CASCADE_STAGES.map((s, i) => {
        const x = startX + i * (boxW + gap);
        const isHigh = i === highlightIdx || highlightIdx === -1;
        const size = 20 + Math.log2(s.resolution) * 3;
        return (
          <g key={i} opacity={isHigh ? 1 : 0.35}>
            <rect x={x} y={y} width={boxW} height={70} rx={4}
                  fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />
            <text x={x + boxW / 2} y={y + 20} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">{s.name}</text>
            <rect x={x + boxW / 2 - size / 2} y={y + 26} width={size} height={size} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1} />
            <text x={x + boxW / 2} y={y + 60} textAnchor="middle" fontSize={9} fill="#6b7280">{s.resolution}² · {s.params}</text>

            <text x={x + boxW / 2} y={y + 95} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#374151">{s.role}</text>

            {i < CASCADE_STAGES.length - 1 && (
              <Arrow x1={x + boxW} y1={y + 35} x2={x + boxW + gap} y2={y + 35} color="#ec4899" id={`cp-a${i}`} />
            )}
          </g>
        );
      })}

      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        每个 diffusion 单独训练,前一阶段输出作为下一阶段条件 · 第一阶段学语义,后续学高频细节
      </text>
    </svg>
  );
}
