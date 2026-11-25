// ReAct 论文里的 Aurora Borealis trace
export interface ReactStep {
  thought: string;
  action: string;
  observation: string;
}
export const REACT_TRACE: ReactStep[] = [
  {
    thought: "我需要先查 Aurora Borealis 的颜色",
    action: "Search[Aurora Borealis]",
    observation: "Aurora Borealis(北极光)是天空中绿色/红色为主的极光现象...",
  },
  {
    thought: "主要是绿色。现在我需要查绿色极光是怎么形成的",
    action: "Search[Aurora green color cause]",
    observation: "绿色极光由氧原子在 100-300 km 高空被太阳风带电粒子激发后...",
  },
  {
    thought: "我已经有了答案 —— 绿色 + 氧原子激发",
    action: "Finish[绿色,由氧原子被太阳风带电粒子激发]",
    observation: "(任务结束)",
  },
];

// 对比:纯 CoT 处理同一问题(容易幻觉)
export const COT_HALLUCINATION = {
  thought: "Aurora Borealis 通常是绿色,绿色由氢原子激发产生...",
  answer: "绿色,氢原子激发",
  isWrong: true,
  correction: "实际是氧原子,不是氢原子 — CoT 全程脑内推理,没有查证机会",
};

// ReAct 工具集
export interface ToolInfo {
  name: string;
  desc: string;
}
export const TOOLS: ToolInfo[] = [
  { name: "Search[entity]", desc: "查 Wikipedia,返回相关摘要" },
  { name: "Lookup[keyword]", desc: "在当前页面内查找关键词" },
  { name: "Calculator[expr]", desc: "计算数学表达式" },
  { name: "Finish[answer]", desc: "给出最终答案,终止循环" },
];

// 三种 prompting 范式 + ReAct 对比(论文 HotpotQA / Fever 数据)
export interface ParadigmRow {
  pattern: string;
  form: string;
  hotpotEM: number;
  feverAcc: number;
}
export const PARADIGM_COMPARE: ParadigmRow[] = [
  { pattern: "Standard", form: "x → y", hotpotEM: 28.7, feverAcc: 57.1 },
  { pattern: "CoT", form: "x → thought → y", hotpotEM: 30.6, feverAcc: 56.3 },
  { pattern: "Act-only", form: "x → action → obs → y", hotpotEM: 25.7, feverAcc: 58.9 },
  { pattern: "ReAct", form: "x → (thought→action→obs)* → y", hotpotEM: 35.1, feverAcc: 62.0 },
  { pattern: "CoT + Self-Consistency", form: "多次 CoT 投票", hotpotEM: 33.4, feverAcc: 60.4 },
];

// Agent 任务对比(ALFWorld / WebShop)
export interface AgentTaskRow {
  task: string;
  standard: number;
  imitationLearning: number;
  react: number;
}
export const AGENT_TASK_COMPARE: AgentTaskRow[] = [
  { task: "ALFWorld(成功率)", standard: 6, imitationLearning: 37, react: 71 },
  { task: "WebShop(成功率)", standard: 9, imitationLearning: 29, react: 40 },
];
