// 高级目标与自动分解出的子任务
export const GOAL_EXAMPLE = "帮我研究一下电动汽车市场,找出 2024 年三大趋势,写一份 1000 字报告并保存为 markdown 文件。";

export const DECOMPOSED_TASKS: string[] = [
  "search latest EV market reports 2024",
  "identify top 3 trends from reports",
  "analyze growth data from Q1-Q3 2024",
  "draft 1000-word report",
  "save report as markdown file",
];

// Tool Loop + Memory:每一步执行后 memory 增长
export interface LoopStep {
  task: string;
  tool: string;
  observation: string;
}
export const TOOL_LOOP_STEPS: LoopStep[] = [
  { task: "search latest EV market reports 2024", tool: "web_search", observation: "找到 8 篇 2024 EV 市场报告..." },
  { task: "identify top 3 trends from reports", tool: "web_browse", observation: "提炼出:电池成本下降/自动驾驶普及/中国车企出海" },
  { task: "analyze growth data from Q1-Q3 2024", tool: "execute_python", observation: "Q1-Q3 销量同比 +23%,图表已生成" },
];

// Self-Critique + Replan:反思前后的 task queue 变化
export interface CritiqueExample {
  before: string[];
  after: string[];
  reason: string;
}
export const CRITIQUE_EXAMPLE: CritiqueExample = {
  before: ["draft 1000-word report", "save report as markdown file"],
  after: ["verify trend data sources are credible", "draft 1000-word report", "add citation footnotes", "save report as markdown file"],
  reason: "LLM 反思发现:趋势数据来源未经核实,直接写报告风险高,插入验证 + 引用步骤",
};

// AutoGPT 默认工具集
export const TOOL_SET: string[] = ["web_search", "web_browse", "read_file", "write_file", "execute_python", "delegate_task"];

// benchmark 对比(论文/项目数据)
export interface BenchRow {
  benchmark: string;
  gpt4React: number;
  autogptStyle: number;
  human: number | null;
}
export const BENCHMARK_TABLE: BenchRow[] = [
  { benchmark: "AgentBench(/10)", gpt4React: 4.01, autogptStyle: 4.42, human: 7.50 },
  { benchmark: "GAIA(%)", gpt4React: 15, autogptStyle: 30, human: 92 },
  { benchmark: "WebArena(%)", gpt4React: 14, autogptStyle: 22, human: 78 },
  { benchmark: "SWE-bench(%)", gpt4React: 1.7, autogptStyle: 12.5, human: null },
  { benchmark: "OSWorld(%)", gpt4React: 12, autogptStyle: 18, human: 72 },
];
