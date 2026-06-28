import { SFT_DEMOS } from "../lib/data";

interface Props {
  demoIdx: number;
}

const W = 700;
const H = 380;

// 同一个 prompt 输入,SFT 前后两栏对比。
// 让 viewer 直接看到 \"对齐\" 在做什么 —— 不是让模型更聪明,而是\"听懂指令\"。

export function SFTBeforeAfter({ demoIdx }: Props) {
  const demo = SFT_DEMOS[demoIdx];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="SFT before / after comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        SFT 前后对比
      </text>

      {/* Prompt 横条 */}
      <rect x={20} y={36} width={W - 40} height={40} rx={5} fill="#fef3c7" stroke="#f59e0b" />
      <text x={32} y={56} fontSize={10} fontWeight={700} fill="#92400e">Prompt</text>
      <text x={32} y={72} fontSize={11} fill="var(--ink-primary)">
        {demo.prompt}
      </text>

      {/* Before */}
      <rect x={20} y={96} width={W - 40} height={120} rx={5} fill="#fef2f2" stroke="#fca5a5" />
      <text x={32} y={116} fontSize={11} fontWeight={700} fill="#7f1d1d">
        ❌ GPT-3 (pretrain only,无 SFT)
      </text>
      <text x={32} y={140} fontSize={10} fontFamily="ui-monospace, monospace" fill="#374151">
        {demo.before.length > 80 ? demo.before.slice(0, 80) + "…" : demo.before}
      </text>
      {demo.before.length > 80 && (
        <text x={32} y={156} fontSize={10} fontFamily="ui-monospace, monospace" fill="#374151">
          {demo.before.slice(80, 160)}
        </text>
      )}
      <text x={32} y={196} fontSize={10} fontStyle="italic" fill="#7f1d1d">
        ↑ 像在续写 \"提示词文档\",不是在执行指令
      </text>

      {/* After */}
      <rect x={20} y={232} width={W - 40} height={130} rx={5} fill="#ecfdf5" stroke="#86efac" />
      <text x={32} y={252} fontSize={11} fontWeight={700} fill="#065f46">
        ✓ SFT-GPT-3 (人类示范 fine-tune 之后)
      </text>
      {demo.after.split("\n").map((line, i) => (
        <text key={i} x={32} y={276 + i * 18} fontSize={11} fill="var(--ink-primary)">
          {line.length > 60 ? line.slice(0, 60) + "…" : line}
        </text>
      ))}
    </svg>
  );
}
