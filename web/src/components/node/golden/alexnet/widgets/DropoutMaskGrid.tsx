const W = 700;
const H = 260;

interface Props {
  p: number;       // drop probability
  seed: number;
}

// 简单 PRNG
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 24×8 = 192 cells 代表 4096 unit FC 的缩略
const COLS = 32;
const ROWS = 8;

export function DropoutMaskGrid({ p, seed }: Props) {
  const rng = mulberry32(seed);
  const cells: boolean[] = [];
  let dropped = 0;
  for (let i = 0; i < COLS * ROWS; i++) {
    const drop = rng() < p;
    cells.push(drop);
    if (drop) dropped++;
  }
  const kept = COLS * ROWS - dropped;

  const PAD = 30;
  const TOP = 50;
  const cellW = (W - PAD * 2) / COLS;
  const cellH = 18;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Dropout random mask visualization">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Dropout mask — p = {p.toFixed(2)} · 256 unit (代表 4096 FC) · 关闭 {dropped} / 保留 {kept}
      </text>

      {cells.map((drop, i) => {
        const r = Math.floor(i / COLS);
        const c = i % COLS;
        const x = PAD + c * cellW;
        const y = TOP + r * (cellH + 2);
        return (
          <rect
            key={i}
            x={x + 1}
            y={y}
            width={cellW - 2}
            height={cellH}
            rx={2}
            fill={drop ? "#fce7f3" : "#ecfdf5"}
            stroke={drop ? "#ec4899" : "#10b981"}
            strokeWidth={0.8}
            opacity={drop ? 0.55 : 1}
          />
        );
      })}

      <text x={W / 2} y={H - 28} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        绿 = 这步参与训练 · 粉 = 被 mask 关掉(输出 = 0,反传也不更新)
      </text>
      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="#6b7280">
        每个 batch 重新采样 → 4096 unit 有 2^4096 种 mask,每次都是不同子网
      </text>
    </svg>
  );
}
