// InstructGPT demo 数据 —— 示范对 / 偏好排序 / RM 打分 / KL 影响 / 对齐税。

export interface DemoPair {
  prompt: string;
  /** SFT 之前(原始 GPT-3)的典型输出 */
  before: string;
  /** SFT 之后的输出 */
  after: string;
}

export const SFT_DEMOS: DemoPair[] = [
  {
    prompt: "解释下午经济学里的 \"机会成本\"。",
    before:
      "Explain the concept of opportunity cost in economics. Provide a real-world example. Mark for full credit...",
    after:
      "机会成本是你做出选择时,所放弃的次优选项的价值。比如,你选择上大学,机会成本就是同时间打工能赚的钱。",
  },
  {
    prompt: "写一首关于秋天的短诗。",
    before:
      "Write a short poem about autumn. The poem should be 4 lines. The first line should rhyme...",
    after:
      "落叶染金风渐凉,\n枫林一片入残阳。\n寒霜悄上窗前石,\n秋意深时人未央。",
  },
  {
    prompt: "用一句话总结相对论。",
    before:
      "Summarize the theory of relativity in one sentence. The student should be able to understand...",
    after: "时间和空间不是绝对的 —— 它们随观察者的运动而改变。",
  },
];

/** 一个 prompt 对应 4 个 candidate response,labeler 排序 */
export interface PreferenceExample {
  prompt: string;
  candidates: Array<{
    text: string;
    /** labeler 排名 (1=best) */
    rank: number;
  }>;
}

export const PREFERENCE_EXAMPLE: PreferenceExample = {
  prompt: "用一句话告诉小孩什么是月亮。",
  candidates: [
    {
      text: "月亮是地球的卫星,会在夜晚发出柔和的光。",
      rank: 1,
    },
    {
      text: "月亮是天上的圆球,晚上能看到。",
      rank: 2,
    },
    {
      text: "Moon (Earth's natural satellite) is a celestial body...",
      rank: 3,
    },
    {
      text: "It is a thing in the sky and humans landed on it in 1969.",
      rank: 4,
    },
  ],
};

/** RM 给两个 response 打的 logit 分数,差就是偏好强度 */
export function rmScore(rank: number, totalRanks: number): number {
  // rank 1 → 高分,rank N → 低分。用 (totalRanks - rank + 1) 作为基础
  return (totalRanks - rank + 1) * 2 + (Math.random() - 0.5) * 0.5;
}

/** KL 约束强度对 reward / 输出多样性 / 模型崩坏的影响 */
export interface KLPoint {
  beta: number;
  reward: number;
  /** 0 → 输出完全跟 SFT 一样 / 1 → 完全自由乱学 */
  drift: number;
  /** 是否退化(reward hacking) */
  hacked: boolean;
}

// β 是 KL 系数。β 小 → policy 允许偏离 SFT 多 → 高 reward 但容易 reward hacking
//                β 大 → policy 紧贴 SFT → 没收益
export const KL_CURVE: KLPoint[] = [
  { beta: 0.001, reward: 8.2, drift: 0.95, hacked: true },
  { beta: 0.01, reward: 6.5, drift: 0.6, hacked: false },
  { beta: 0.05, reward: 5.4, drift: 0.3, hacked: false },
  { beta: 0.1, reward: 4.6, drift: 0.18, hacked: false },
  { beta: 0.5, reward: 2.9, drift: 0.05, hacked: false },
  { beta: 1.0, reward: 1.5, drift: 0.02, hacked: false },
];

/** Alignment Tax:RLHF 对齐后某些任务掉点 */
export interface TaxRow {
  task: string;
  pretrainScore: number;
  alignedScore: number;
}

export const ALIGNMENT_TAX: TaxRow[] = [
  { task: "MMLU 通识", pretrainScore: 0.43, alignedScore: 0.41 },
  { task: "HellaSwag 常识", pretrainScore: 0.78, alignedScore: 0.76 },
  { task: "代码补全 HumanEval", pretrainScore: 0.21, alignedScore: 0.18 },
  { task: "TriviaQA 事实", pretrainScore: 0.66, alignedScore: 0.59 },
  { task: "人类偏好(API 实测)", pretrainScore: 0.10, alignedScore: 0.85 },
];
