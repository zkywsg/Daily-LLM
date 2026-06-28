// CoT demo 数据:示例题(standard vs CoT)+ scaling 曲线 + 提示词变体。

export interface QAExample {
  question: string;
  /** Standard prompt 输出(直接给答案) */
  standardAnswer: string;
  /** 标准 prompt 给的最终答案,可能错 */
  standardCorrect: boolean;
  /** CoT 拆解的逐步推理(每条 1 步) */
  cotSteps: string[];
  /** CoT 最终答案 */
  cotAnswer: string;
  /** CoT 是否答对 */
  cotCorrect: boolean;
}

export const QA_EXAMPLES: QAExample[] = [
  {
    question:
      "Roger 有 5 个网球。他又买了 2 罐网球,每罐 3 个。他现在有多少个网球?",
    standardAnswer: "11",
    standardCorrect: true,
    cotSteps: [
      "Roger 一开始有 5 个网球。",
      "他买了 2 罐,每罐 3 个,所以新买 2 × 3 = 6 个。",
      "总共 5 + 6 = 11 个。",
    ],
    cotAnswer: "11",
    cotCorrect: true,
  },
  {
    question:
      "餐厅有 23 个苹果。如果他们用 20 个做了午饭,然后又买了 6 个,现在有多少苹果?",
    standardAnswer: "27",
    standardCorrect: false,
    cotSteps: [
      "餐厅一开始有 23 个苹果。",
      "做午饭用了 20 个,剩下 23 - 20 = 3 个。",
      "又买了 6 个,所以现在有 3 + 6 = 9 个。",
    ],
    cotAnswer: "9",
    cotCorrect: true,
  },
  {
    question:
      "停车场有 3 辆车。如果又来了 2 辆,然后开走了 1 辆,现在停车场有几辆车?",
    standardAnswer: "5",
    standardCorrect: false,
    cotSteps: [
      "一开始有 3 辆车。",
      "又来了 2 辆,变成 3 + 2 = 5 辆。",
      "开走 1 辆,剩下 5 - 1 = 4 辆。",
    ],
    cotAnswer: "4",
    cotCorrect: true,
  },
];

/** Zero-shot CoT 魔法咒语变体的准确率(GSM8K 量级,数据点取自 Kojima 2022) */
export const SPELL_VARIANTS = [
  { spell: "(none)", note: "直接答", acc: 0.176, hue: 0 },
  { spell: "Let's think step by step", note: "原始魔法咒语", acc: 0.408, hue: 330 },
  { spell: "Let's think about this logically", note: "形容词变体", acc: 0.301, hue: 220 },
  { spell: "Don't think. Just answer.", note: "反向控制", acc: 0.182, hue: 30 },
];

/** CoT 涌现曲线:standard vs CoT 在不同模型规模下的 GSM8K 准确率 */
export interface ScalePoint {
  model: string;
  /** 参数量 */
  params: number;
  standard: number;
  cot: number;
}

// 数量级取自 Wei 2022 Fig 4(具体数字稍简化,呈现"~62B 涌现"的形状)
export const SCALE_POINTS: ScalePoint[] = [
  { model: "GPT-3 350M", params: 0.35e9, standard: 0.04, cot: 0.03 },
  { model: "GPT-3 1.3B", params: 1.3e9, standard: 0.05, cot: 0.04 },
  { model: "GPT-3 6.7B", params: 6.7e9, standard: 0.07, cot: 0.06 },
  { model: "GPT-3 13B", params: 13e9, standard: 0.09, cot: 0.10 },
  { model: "GPT-3 175B", params: 175e9, standard: 0.16, cot: 0.55 },
  { model: "PaLM 540B", params: 540e9, standard: 0.18, cot: 0.58 },
];

export function fmtParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return `${n}`;
}
