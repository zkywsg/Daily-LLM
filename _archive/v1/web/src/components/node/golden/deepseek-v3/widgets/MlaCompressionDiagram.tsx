import { KV_CACHE_COMPARE } from "../lib/data";

interface Props {
  mode: "mha" | "mla";
}

const W = 700;
const H = 340;

// MHA:每个 head 独立存完整 K/V(多个小方块堆叠)。
// MLA:压缩到一个共享的低秩 latent 向量,推理时按需解压出各 head 的 K/V。

const NUM_HEADS = 8;

export function MlaCompressionDiagram({ mode }: Props) {
  const row = KV_CACHE_COMPARE[mode === "mha" ? 0 : 1];
  const barMaxH = 220;
  const barW = 140;
  const left1 = 140;
  const left2 = 420;

  const relH = (row.relativeSize / 4) * barMaxH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`${row.method} KV cache 示意`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {row.method} — KV cache 相对大小 ≈ {row.relativeSize}×
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        {row.note}
      </text>

      {mode === "mha" ? (
        <g>
          {Array.from({ length: NUM_HEADS }, (_, i) => {
            const hh = barMaxH / NUM_HEADS;
            const y = 70 + i * hh;
            return (
              <g key={i}>
                <rect x={left1} y={y + 1} width={barW / 2 - 3} height={hh - 2} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} />
                <rect x={left1 + barW / 2 + 3} y={y + 1} width={barW / 2 - 3} height={hh - 2} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} />
              </g>
            );
          })}
          <text x={left1 + barW / 2} y={70 + barMaxH + 22} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
            8 head × 独立 K/V
          </text>
        </g>
      ) : (
        <g>
          <rect x={left1 + barW / 2 - 30} y={70 + barMaxH - relH} width={60} height={relH} rx={6} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.6} />
          <text x={left1 + barW / 2} y={70 + barMaxH - relH / 2 + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="#047857">
            latent
          </text>
          <text x={left1 + barW / 2} y={70 + barMaxH + 22} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
            共享低秩向量
          </text>
          {/* 解压箭头 */}
          <text x={left1 + barW / 2} y={70 + barMaxH + 40} textAnchor="middle" fontSize={9} fontStyle="italic" fill="var(--ink-muted)">
            推理时按需解压出各 head K/V
          </text>
        </g>
      )}

      {/* 高度对比条 */}
      <g transform={`translate(${left2}, 70)`}>
        <text x={60} y={-14} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--ink-secondary)">
          KV cache 大小对比
        </text>
        {KV_CACHE_COMPARE.map((r, i) => {
          const h = (r.relativeSize / 4) * barMaxH;
          const x = i * 70;
          const active = r.method === row.method;
          return (
            <g key={r.method}>
              <rect x={x} y={barMaxH - h} width={44} height={h} rx={3} fill={active ? (mode === "mha" ? "#3b82f6" : "#10b981") : "#e5e7eb"} opacity={active ? 0.9 : 0.5} />
              <text x={x + 22} y={barMaxH - h - 6} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
                {r.relativeSize}×
              </text>
              <text x={x + 22} y={barMaxH + 16} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">
                {r.method === "传统 MHA" ? "MHA" : "MLA"}
              </text>
            </g>
          );
        })}
      </g>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        MLA 把 KV cache 压到传统 MHA 的 ~1/4 —— 是 V3 支撑 128K 长上下文的关键
      </text>
    </svg>
  );
}
