interface Props {
  /** 当前 step:0 = 训练前,>0 = 训了几步 */
  step: number;
}

const W = 700;
const H = 220;

// 演示:B=0 + A=Kaiming 让 ΔW(step 0) = B·A = 0 → 行为完全等价于原模型。
// 这是 LoRA 工业可用的关键 —— 接上 adapter 不会"立即破坏"模型,
// 训练再渐进偏离。step slider 控制 B 从 0 渐变成有值。

export function InitDiagramSVG({ step }: Props) {
  // step ∈ [0, 100],映射 B 矩阵的 fill 强度
  const bIntensity = Math.min(1, step / 100);
  const aIntensity = 1; // A 一开始就有值(Kaiming)

  const BX = 80;
  const BY = 60;
  const AX = 200;
  const AY = 60;
  const DWX = 360;
  const DWY = 60;
  const boxW = 80;
  const boxH = 100;

  const cellColor = (intensity: number, baseHue: number) =>
    intensity === 0 ? "#ffffff" : `hsl(${baseHue}, 70%, ${95 - intensity * 50}%)`;

  const dwIntensity = bIntensity * aIntensity; // ΔW = BA,所以正比

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`LoRA init at step ${step}`}
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Step {step}:B(初始化为 0)+ A(Kaiming) → ΔW = B·A
      </text>

      {/* B */}
      <rect
        x={BX}
        y={BY}
        width={boxW}
        height={boxH}
        fill={cellColor(bIntensity, 220)}
        stroke="#3b82f6"
        strokeWidth={1.5}
        rx={4}
      />
      <text x={BX + boxW / 2} y={BY - 6} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1d4ed8">
        B (d × r)
      </text>
      <text x={BX + boxW / 2} y={BY + boxH + 14} textAnchor="middle" fontSize={10} fill="#3b82f6">
        {bIntensity === 0 ? "全 0" : `≈ ${(bIntensity * 100).toFixed(0)}% 已学到`}
      </text>

      <text x={BX + boxW + 20} y={H / 2 + 4} fontSize={20} fontWeight={600} fill="var(--ink-primary)">
        ·
      </text>

      {/* A */}
      <rect
        x={AX}
        y={AY}
        width={boxW}
        height={boxH}
        fill={cellColor(aIntensity, 220)}
        stroke="#3b82f6"
        strokeWidth={1.5}
        rx={4}
      />
      <text x={AX + boxW / 2} y={AY - 6} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1d4ed8">
        A (r × k)
      </text>
      <text x={AX + boxW / 2} y={AY + boxH + 14} textAnchor="middle" fontSize={10} fill="#3b82f6">
        Kaiming 初始化(有值)
      </text>

      <text x={AX + boxW + 30} y={H / 2 + 4} fontSize={16} fontWeight={600} fill="var(--ink-primary)">
        =
      </text>

      {/* ΔW = BA */}
      <rect
        x={DWX}
        y={DWY}
        width={boxW * 1.5}
        height={boxH}
        fill={cellColor(dwIntensity, 330)}
        stroke="#ec4899"
        strokeWidth={1.5}
        rx={4}
      />
      <text x={DWX + boxW * 1.5 / 2} y={DWY - 6} textAnchor="middle" fontSize={11} fontWeight={600} fill="#831843">
        ΔW (d × k)
      </text>
      <text x={DWX + boxW * 1.5 / 2} y={DWY + boxH + 14} textAnchor="middle" fontSize={10} fill="#831843">
        {dwIntensity === 0 ? "= 0 → 模型行为 = W₀" : `≈ ${(dwIntensity * 100).toFixed(0)}% 偏离`}
      </text>

      {/* 解释行 */}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill={step === 0 ? "#10b981" : "#ec4899"}>
        {step === 0
          ? "✓ Step 0:ΔW = 0,接上 LoRA 不影响原模型推理"
          : `Step ${step}:ΔW 已偏离原模型,新行为来自 trainable 的 B/A`}
      </text>
    </svg>
  );
}
