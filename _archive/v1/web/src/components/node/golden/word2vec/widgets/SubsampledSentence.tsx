import { DEMO_SENTENCE, discardProb, frequency } from "../lib/data";

const W = 700;
const H = 200;

interface Props {
  t: number;
  seed: number;
}

// 简单 PRNG, 让 seed 决定哪些位置被丢
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

export function SubsampledSentence({ t, seed }: Props) {
  const rng = mulberry32(seed);
  // For each token: decide drop based on discardProb
  const decisions = DEMO_SENTENCE.map((tok) => {
    const f = frequency(tok);
    const p = f === 0 ? 0 : discardProb(f, t);
    const r = rng();
    return { tok, dropped: r < p, p };
  });

  const TILE_H = 32;
  const PAD = 8;
  let xCursor = PAD;
  const yRow1 = 50;
  const yRow2 = 110;

  let kept = 0;
  decisions.forEach((d) => { if (!d.dropped) kept++; });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Sentence after subsampling demo">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        一句训练语料 — 丢弃前 vs 丢弃后(t = {t.toExponential(0)})
      </text>

      <text x={PAD} y={42} fontSize={10} fontWeight={600} fill="#6b7280">原始 (17 tokens):</text>
      {(() => {
        xCursor = PAD;
        return decisions.map((d, i) => {
          const tw = Math.max(d.tok.length * 8 + 12, 36);
          const x = xCursor;
          xCursor += tw + 4;
          return (
            <g key={i}>
              <rect x={x} y={yRow1} width={tw} height={TILE_H} fill={d.dropped ? "#fce7f3" : "#fef3c7"} stroke={d.dropped ? "#ec4899" : "#f59e0b"} strokeWidth={1.2} rx={3} opacity={d.dropped ? 0.7 : 1} />
              <text x={x + tw / 2} y={yRow1 + TILE_H / 2 + 4} textAnchor="middle" fontSize={11} fontWeight={500} fill={d.dropped ? "#831843" : "#92400e"} textDecoration={d.dropped ? "line-through" : undefined}>
                {d.tok}
              </text>
            </g>
          );
        });
      })()}

      <text x={PAD} y={102} fontSize={10} fontWeight={600} fill="#6b7280">保留 ({kept} tokens):</text>
      {(() => {
        let xc = PAD;
        return decisions.filter((d) => !d.dropped).map((d, i) => {
          const tw = Math.max(d.tok.length * 8 + 12, 36);
          const x = xc;
          xc += tw + 4;
          return (
            <g key={i}>
              <rect x={x} y={yRow2} width={tw} height={TILE_H} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.2} rx={3} />
              <text x={x + tw / 2} y={yRow2 + TILE_H / 2 + 4} textAnchor="middle" fontSize={11} fontWeight={500} fill="#065f46">
                {d.tok}
              </text>
            </g>
          );
        });
      })()}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        高频词 "the / of / a / and" 被概率丢弃,算力流向 "fox / king / brown" 这些有信息量的位置
      </text>
    </svg>
  );
}
