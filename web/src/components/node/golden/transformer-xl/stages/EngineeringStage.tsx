import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TRANSFORMER_XL_SOURCE_PATH } from "../lib/prose";
import { INFERENCE_SPEEDUP } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

function SpeedupChart() {
  const W = 700;
  const H = 220;
  const PAD_L = 240;
  const PAD_R = 60;
  const PAD_T = 40;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 50;
  const maxVal = Math.log10(INFERENCE_SPEEDUP[1].relativeSpeed) * 1.15;
  const wOf = (v: number) => (Math.log10(Math.max(v, 1)) / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="推理速度对比(log scale)">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        推理速度对比(log scale)— Segment Cache 比 stride-1 sliding window 快 1874×
      </text>
      {INFERENCE_SPEEDUP.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isXl = row.method.startsWith("Transformer-XL");
        const color = isXl ? "#3b82f6" : "#9ca3af";
        const bg = isXl ? "#dbeafe" : "#f3f4f6";
        return (
          <g key={row.method}>
            <text x={PAD_L - 8} y={y + 15} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>{row.method}</text>
            <rect x={PAD_L} y={y} width={Math.max(wOf(row.relativeSpeed), 4)} height={22} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + Math.max(wOf(row.relativeSpeed), 4) + 6} y={y + 16} fontSize={10} fontWeight={700} fill={color}>
              {row.relativeSpeed}×
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function EngineeringStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Left-shift Trick + 工程加速 — 让 relative PE 实际跑得起
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        朴素实现相对 PE 需要为每对 (i, j) 显式构造 R_{"{i-j}"},额外开销
        O(N²×d)。Dai 提出 left-shift trick:先算一个"假设所有位置都是绝对距离"的
        大矩阵,再通过矩阵元素左移 + reshape 直接得到等价的相对位置矩阵——把额外
        开销从 O(N²·d) 压到约 10%,这是相对 PE 能被工程化落地的关键一步。
      </p>

      <SpeedupChart />
      <p className={styles.caption}>
        ↑ Segment cache 每段只算一次完整 attention,不需要逐 token 滑窗重算,论文 Table 5 报告推理快 1874 倍。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              缺一不可
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>只有 segment cache,没有 relative PE — 跨段位置歧义,缓存无意义</li>
              <li>只有 relative PE,没有 segment cache — 有效上下文仍是单段 N</li>
              <li>没有 left-shift trick — O(N²·d) 开销,工程上跑不动</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
