import { KV_CACHE_COMPARE } from "../lib/data";

const W = 700;
const H = 220;

export function KvCacheCompareChart() {
  const PAD_L = 120;
  const PAD_T = 50;
  const plotW = 380;
  const rowH = 44;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="MHA vs GQA vs MQA KV cache 大小对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        KV Cache 相对大小 — 共享 K/V 头数越少,cache 越小
      </text>

      {KV_CACHE_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const w = row.relativeCacheSize * plotW;
        const color = row.name === "MHA" ? "#9ca3af" : row.name.startsWith("GQA") ? "#3b82f6" : "#ec4899";
        const bg = row.name === "MHA" ? "#f3f4f6" : row.name.startsWith("GQA") ? "#dbeafe" : "#fce7f3";
        return (
          <g key={row.name}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.name}</text>
            <rect x={PAD_L} y={y} width={plotW} height={22} fill={bg} stroke={color} strokeWidth={1} rx={3} opacity={0.4} />
            <rect x={PAD_L} y={y} width={Math.max(w, 4)} height={22} fill={color} rx={3} />
            <text x={PAD_L + Math.max(w, 4) + 8} y={y + 16} fontSize={10} fontWeight={700} fill={color}>
              {(row.relativeCacheSize * 100).toFixed(1)}%(共享 {row.numHeads / row.groups} 头/组)
            </text>
          </g>
        );
      })}
    </svg>
  );
}
