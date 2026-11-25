import { WORD_POINTS_2D } from "../lib/data";

const W = 700;
const H = 400;

// 2D 投影展示 king - man + woman ≈ queen 的线性算术.
// x = royalty 维度 (高 = 更"皇室"), y = gender (正 = 女)
export function LinearArithmetic2D() {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // x ∈ [0, 1] → screen
  const xOf = (x: number) => PAD_L + x * plotW;
  // y ∈ [-1, 1] → screen (反向)
  const yOf = (y: number) => PAD_T + ((1 - y) / 2) * plotH;

  const get = (w: string) => WORD_POINTS_2D.find((p) => p.word === w)!;
  const king = get("king");
  const man = get("man");
  const woman = get("woman");
  const queen = get("queen");

  // result = king - man + woman
  const result = {
    x: king.x - man.x + woman.x,
    y: king.y - man.y + woman.y,
  };

  function Arrow({ x1, y1, x2, y2, color, id, dashed }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string; dashed?: boolean }) {
    return (
      <g>
        <defs>
          <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
          </marker>
        </defs>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.8} markerEnd={`url(#${id})`} strokeDasharray={dashed ? "5 4" : undefined} />
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="king - man + woman ≈ queen linear arithmetic">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        线性结构 — vec("king") − vec("man") + vec("woman") ≈ vec("queen")
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T + plotH / 2} x2={W - PAD_R} y2={PAD_T + plotH / 2} stroke="#e5e7eb" />
      <line x1={PAD_L + plotW / 4} y1={PAD_T} x2={PAD_L + plotW / 4} y2={PAD_T + plotH} stroke="#e5e7eb" />
      <text x={W - PAD_R - 4} y={PAD_T + plotH / 2 + 14} fontSize={10} fill="#9ca3af">→ royalty 方向</text>
      <text x={PAD_L + plotW / 4 + 6} y={PAD_T + 12} fontSize={10} fill="#9ca3af">↑ female</text>
      <text x={PAD_L + plotW / 4 + 6} y={PAD_T + plotH - 4} fontSize={10} fill="#9ca3af">↓ male</text>

      {/* 所有词点 */}
      {WORD_POINTS_2D.map((p) => {
        const isHero = ["king", "man", "woman", "queen"].includes(p.word);
        const color = p.group === "royalty" ? "#ec4899" : p.group === "gender" ? "#f59e0b" : "#3b82f6";
        return (
          <g key={p.word} opacity={isHero ? 1 : 0.35}>
            <circle cx={xOf(p.x)} cy={yOf(p.y)} r={isHero ? 6 : 4} fill={color} stroke="#fff" strokeWidth={1.5} />
            <text x={xOf(p.x) + 9} y={yOf(p.y) + 4} fontSize={isHero ? 12 : 10} fontWeight={isHero ? 700 : 500} fill={isHero ? "#1f2937" : "#6b7280"}>
              {p.word}
            </text>
          </g>
        );
      })}

      {/* king → -man (减去 man 向量) */}
      <Arrow id="step1" x1={xOf(king.x)} y1={yOf(king.y)} x2={xOf(king.x - man.x)} y2={yOf(king.y - man.y)} color="#ec4899" />
      {/* + woman */}
      <Arrow id="step2" x1={xOf(king.x - man.x)} y1={yOf(king.y - man.y)} x2={xOf(result.x)} y2={yOf(result.y)} color="#f59e0b" />

      {/* result marker */}
      <circle cx={xOf(result.x)} cy={yOf(result.y)} r={9} fill="none" stroke="#10b981" strokeWidth={2.2} strokeDasharray="3 3" />
      <text x={xOf(result.x) + 14} y={yOf(result.y) - 8} fontSize={11} fontWeight={700} fill="#065f46">
        ≈ queen
      </text>

      {/* 虚线连接 result → queen */}
      <Arrow id="match" x1={xOf(result.x)} y1={yOf(result.y)} x2={xOf(queen.x) - 6} y2={yOf(queen.y)} color="#10b981" dashed />

      {/* 公式标注 */}
      <text x={PAD_L} y={H - 18} fontSize={11} fontStyle="italic" fill="#6b7280">
        粉箭头 = − man,黄箭头 = + woman,落点正好落在 queen 附近 — gender 方向在向量空间里是个固定平移
      </text>
    </svg>
  );
}
