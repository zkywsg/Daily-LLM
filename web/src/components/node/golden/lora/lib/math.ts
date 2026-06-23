// LoRA 数学:全量微调 vs LoRA 的参数账 + W = W₀ + (α/r)·BA 分解。
// widget 只画,数学集中在这里。

export interface LoraConfig {
  /** 大权重维度 d × k(常见 attention proj 是 d_model × d_model) */
  d: number;
  k: number;
  /** 低秩 rank r */
  r: number;
  /** scaling factor α */
  alpha: number;
}

export interface ParamAccount {
  /** 原权重 W₀ 参数量(冻结,不在 trainable 里) */
  frozen: number;
  /** LoRA 新增 trainable 参数:A (r×k) + B (d×r) */
  loraTrainable: number;
  /** 等效"如果走全量微调"的 trainable 参数 */
  fullTrainable: number;
  /** LoRA 占全量的比例(用于显示百分比) */
  fraction: number;
}

export function paramAccount({ d, k, r }: LoraConfig): ParamAccount {
  const frozen = d * k;
  const loraTrainable = r * k + d * r;
  const fullTrainable = d * k;
  return {
    frozen,
    loraTrainable,
    fullTrainable,
    fraction: loraTrainable / fullTrainable,
  };
}

/**
 * 给定 rank,LoRA 等效 ΔW = (α/r)·B·A 的"覆盖维度"是 r(秩)。
 * 全量微调时 ΔW 可以是任意 d×k 满秩矩阵。
 * 这个比例直接对比"可学方向数"。
 */
export function rankCoverage({ d, k, r }: LoraConfig): {
  fullRankCap: number;
  loraRank: number;
  fraction: number;
} {
  const fullRankCap = Math.min(d, k);
  return {
    fullRankCap,
    loraRank: r,
    fraction: r / fullRankCap,
  };
}

/**
 * 经验性 stage 数据:全量 / LoRA(rank) / Adapter 的对比维度。
 * 数值取自 LoRA 论文 RoBERTa-large(参考量级,不是精确复现)。
 */
export interface MethodMetrics {
  label: string;
  /** 训练参数百分比(相对全量) */
  paramPct: number;
  /** 推理是否增加延迟(true = 有额外延迟) */
  inferenceLatency: boolean;
  /** GLUE 平均分(相对全量的相对差,正负数表示) */
  glueDelta: number;
  /** 简短一句话特征 */
  note: string;
  /** 显示色调(family-11 调) */
  color: string;
}

export const METHOD_COMPARE: MethodMetrics[] = [
  {
    label: "Full fine-tune",
    paramPct: 100,
    inferenceLatency: false,
    glueDelta: 0,
    note: "基线 — 改全部参数",
    color: "#9ca3af",
  },
  {
    label: "Adapter (Houlsby)",
    paramPct: 0.5,
    inferenceLatency: true,
    glueDelta: -0.4,
    note: "插小 MLP 层,串联到 transformer 块里 → 推理多一层",
    color: "#f59e0b",
  },
  {
    label: "Prefix-Tuning",
    paramPct: 0.1,
    inferenceLatency: true,
    glueDelta: -0.6,
    note: "学一组连续 prefix token,挤占 context 长度",
    color: "#fb923c",
  },
  {
    label: "LoRA (r=8)",
    paramPct: 0.3,
    inferenceLatency: false,
    glueDelta: -0.1,
    note: "并联低秩分支,推理时可合并回 W₀ → 零额外延迟",
    color: "#3b82f6",
  },
];
