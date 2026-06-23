import { paramAccount, type LoraConfig } from "../lib/math";

interface Props {
  config: LoraConfig;
}

const W = 700;
const H = 160;

function fmt(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}

// 横向条形对比:全量微调 vs LoRA 的 trainable 参数量。
// 用对数刻度避免 LoRA 那条几乎看不见 —— 这样 viewer 直接读出"1000× 减少"。
export function ParamAccountBar({ config }: Props) {
  const acc = paramAccount(config);
  const minV = Math.max(1, acc.loraTrainable);
  const maxV = acc.fullTrainable;
  const xLog = (v: number) => {
    const logMin = Math.log10(minV);
    const logMax = Math.log10(maxV);
    return ((Math.log10(Math.max(1, v)) - logMin) / Math.max(0.001, logMax - logMin)) * (W - 200);
  };
  const barH = 28;
  const py = 50;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LoRA vs full fine-tune trainable param count">
      <text x={20} y={20} fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        trainable 参数量(log scale)
      </text>

      {/* 全量微调 */}
      <text x={20} y={py + 18} fontSize={11} fill="var(--ink-secondary)">
        Full fine-tune
      </text>
      <rect x={120} y={py + 6} width={Math.max(2, xLog(acc.fullTrainable))} height={barH} fill="#e5e7eb" stroke="#6b7280" rx={3} />
      <text x={130 + xLog(acc.fullTrainable)} y={py + 24} fontSize={11} fontWeight={600} fill="var(--ink-primary)">
        {fmt(acc.fullTrainable)}
      </text>

      {/* LoRA */}
      <text x={20} y={py + barH + 30} fontSize={11} fill="var(--ink-secondary)">
        LoRA (r={config.r})
      </text>
      <rect
        x={120}
        y={py + barH + 18}
        width={Math.max(2, xLog(acc.loraTrainable))}
        height={barH}
        fill="#dbeafe"
        stroke="#3b82f6"
        rx={3}
      />
      <text x={130 + xLog(acc.loraTrainable)} y={py + barH + 36} fontSize={11} fontWeight={600} fill="#1d4ed8">
        {fmt(acc.loraTrainable)} · {(acc.fraction * 100).toFixed(3)}%
      </text>

      {/* 注脚 */}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        d={config.d} · k={config.k} · 单层 LoRA = A({config.r}×{config.k}) + B({config.d}×{config.r}) = {fmt(acc.loraTrainable)} 参数
      </text>
    </svg>
  );
}
