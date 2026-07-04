import { getSubwords, hashSubword } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  buckets: number;
}

const DEMO_SUBWORDS = [
  ...getSubwords("apple"),
  ...getSubwords("applle"),
  ...getSubwords("apply"),
].filter((v, i, arr) => arr.indexOf(v) === i).slice(0, 16);

export function HashBucketDiagram({ buckets }: Props) {
  const bucketBoxes = Math.min(buckets, 12); // 演示用小 bucket 数,凸显碰撞
  const bucketW = (W - 80) / bucketBoxes;

  const assigned = DEMO_SUBWORDS.map((g) => ({ g, b: hashSubword(g, bucketBoxes) }));
  const collisions = new Map<number, number>();
  assigned.forEach(({ b }) => collisions.set(b, (collisions.get(b) ?? 0) + 1));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="subword hashing 到 bucket 的演示">
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          Hashing Trick — {bucketBoxes} 个 bucket(演示;论文用 B=2,000,000)
        </text>

        <g transform="translate(40, 40)">
          {Array.from({ length: bucketBoxes }).map((_, i) => {
            const x = i * bucketW;
            const n = collisions.get(i) ?? 0;
            const collided = n > 1;
            return (
              <g key={i}>
                <rect x={x} y={0} width={bucketW - 6} height={40} fill={collided ? "#fce7f3" : "#dbeafe"} stroke={collided ? "#ec4899" : "#3b82f6"} strokeWidth={collided ? 2 : 1.2} rx={4} />
                <text x={x + (bucketW - 6) / 2} y={16} textAnchor="middle" fontSize={9} fill="#374151">bucket {i}</text>
                <text x={x + (bucketW - 6) / 2} y={31} textAnchor="middle" fontSize={11} fontWeight={700} fill={collided ? "#be185d" : "#1e40af"}>{n}</text>
              </g>
            );
          })}
        </g>

        <g transform="translate(40, 100)">
          {assigned.map(({ g, b }, i) => {
            const col = i % 8;
            const row = Math.floor(i / 8);
            const x = col * 78;
            const y = row * 34;
            const bucketX = b * bucketW + (bucketW - 6) / 2;
            return (
              <g key={i}>
                <rect x={x} y={y} width={70} height={22} fill="#fef3c7" stroke="#f59e0b" rx={3} />
                <text x={x + 35} y={y + 15} textAnchor="middle" fontSize={9} fontFamily="monospace" fill="#374151">{g}</text>
                <line x1={x + 35} y1={y} x2={bucketX} y2={-20} stroke="#d1d5db" strokeWidth={0.8} strokeDasharray="2 2" opacity={0.5} />
              </g>
            );
          })}
        </g>

        <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          粉色 bucket = 碰撞(≥2 个不相关 subword 共享向量)。B 越大碰撞越少,论文 B=2M 时碰撞对精度影响 &lt; 1%。
        </text>
      </svg>
  );
}
