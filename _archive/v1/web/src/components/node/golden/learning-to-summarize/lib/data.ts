// 人工评测:RLHF vs SFT vs 参考摘要 vs 人类高质量摘要(论文 Figure 1,Reddit TL;DR)
export interface HumanPrefRow {
  label: string;
  winRate: number; // % 时间被人工偏好胜过参考摘要
  group: "baseline" | "sft" | "rlhf" | "human";
}
export const HUMAN_PREF_SCORES: HumanPrefRow[] = [
  { label: "参考摘要(基准)", winRate: 50, group: "baseline" },
  { label: "监督微调 1.3B", winRate: 35, group: "sft" },
  { label: "监督微调 6.7B", winRate: 41, group: "sft" },
  { label: "人类写的高质量摘要", winRate: 70, group: "human" },
  { label: "RLHF 1.3B", winRate: 62, group: "rlhf" },
  { label: "RLHF 6.7B", winRate: 74, group: "rlhf" },
];

// 三阶段流程节点
export interface PipelineStage {
  key: string;
  label: string;
  detail: string;
}
export const PIPELINE_STAGES: PipelineStage[] = [
  { key: "sft", label: "Stage 1: SFT", detail: "在 Reddit TL;DR / CNN-DM 参考摘要上微调预训练 LLM" },
  { key: "label", label: "人工标注偏好", detail: "SFT 模型生成 4 候选,标注员两两比较,64K 对" },
  { key: "rm", label: "Stage 2: 训练 RM", detail: "Bradley-Terry loss,RM 初始化为 SFT 权重 + scalar head" },
  { key: "ppo", label: "Stage 3: PPO", detail: "max E[r(x,y)] − β·KL(π‖π_SFT)" },
];

// 候选摘要样例(用于 PreferenceComparisonDiagram 演示)
export interface CandidateSummary {
  id: "A" | "B";
  text: string;
}
export const CANDIDATE_SUMMARIES: CandidateSummary[] = [
  { id: "A", text: "楼主换了新工作,通勤时间从 10 分钟变成 1 小时,正在纠结要不要辞职换回原来的生活方式。" },
  { id: "B", text: "楼主找到了一份加薪的新工作,但通勤时间变长了很多,他觉得不太值得。" },
];

// Bradley-Terry reward model 打分示例(winner 显著高于 loser)
export interface RewardScoreRow {
  candidate: "winner" | "loser";
  score: number;
}
export const REWARD_SCORE_DEMO: RewardScoreRow[] = [
  { candidate: "winner", score: 2.3 },
  { candidate: "loser", score: -0.6 },
];

// KL 系数 β 对 reward / policy drift 的权衡(演示用,非论文实测曲线)
export interface KlTradeoffRow {
  beta: number;
  rewardScore: number; // RM 打分(相对值)
  klDrift: number; // 与 SFT 的 KL 散度(相对值)
  hacked: boolean;
}
export const KL_TRADEOFF: KlTradeoffRow[] = [
  { beta: 0.005, rewardScore: 9.2, klDrift: 8.5, hacked: true },
  { beta: 0.02, rewardScore: 7.4, klDrift: 3.1, hacked: false },
  { beta: 0.05, rewardScore: 5.6, klDrift: 1.4, hacked: false },
  { beta: 0.1, rewardScore: 3.8, klDrift: 0.6, hacked: false },
  { beta: 0.3, rewardScore: 1.9, klDrift: 0.2, hacked: false },
];

// 训练规模一览(6.7B 版)
export interface TrainingDetailRow {
  dim: string;
  value: string;
}
export const TRAINING_DETAILS: TrainingDetailRow[] = [
  { dim: "Backbone", value: "GPT-3 1.3B / 6.7B,decoder-only Transformer" },
  { dim: "Stage 1 SFT", value: "Reddit TL;DR + CNN/DM 微调" },
  { dim: "Stage 2 RM 数据", value: "64K 对人工比较" },
  { dim: "Stage 3 PPO", value: "~10K updates,每 update ~512 prompt" },
  { dim: "硬件", value: "8 × V100(SFT/RM 天级,PPO 周级)" },
  { dim: "标注成本", value: "64K 对 × ~$0.20 ≈ $13K" },
];
