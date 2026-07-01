// 任务对不同层的权重分布(论文 Table)
export interface LayerWeight {
  task: string;
  charCnn: number;   // Layer 0
  lstmL1: number;    // Layer 1
  lstmL2: number;    // Layer 2
  favors: "syntax" | "middle" | "semantic";
}

export const LAYER_WEIGHTS: LayerWeight[] = [
  { task: "POS Tagging",    charCnn: 0.49, lstmL1: 0.30, lstmL2: 0.21, favors: "syntax" },
  { task: "Coref",          charCnn: 0.31, lstmL1: 0.36, lstmL2: 0.33, favors: "middle" },
  { task: "SQuAD",          charCnn: 0.27, lstmL1: 0.39, lstmL2: 0.34, favors: "middle" },
  { task: "SST-5 Sentiment",charCnn: 0.34, lstmL1: 0.34, lstmL2: 0.32, favors: "middle" },
  { task: "WSD",            charCnn: 0.25, lstmL1: 0.30, lstmL2: 0.45, favors: "semantic" },
];

// 6 个 NLP 任务的 ELMo 提升
export interface TaskGain {
  task: string;
  baseline: number;
  prevSota: number;
  withElmo: number;
  metric: string;
}

export const TASK_GAINS: TaskGain[] = [
  { task: "SQuAD",  prevSota: 84.4,  baseline: 81.1,  withElmo: 85.8,  metric: "F1"  },
  { task: "SNLI",   prevSota: 88.6,  baseline: 88.0,  withElmo: 88.7,  metric: "Acc" },
  { task: "SRL",    prevSota: 81.7,  baseline: 81.4,  withElmo: 84.6,  metric: "F1"  },
  { task: "Coref",  prevSota: 67.2,  baseline: 67.2,  withElmo: 70.4,  metric: "F1"  },
  { task: "NER",    prevSota: 91.93, baseline: 90.15, withElmo: 92.22, metric: "F1"  },
  { task: "SST-5",  prevSota: 53.7,  baseline: 51.4,  withElmo: 54.7,  metric: "Acc" },
];

// 上下文示例:bank 的两种用法
export interface ContextExample {
  sentence: string[];
  bankIdx: number;
  meaning: "river" | "money";
  color: string;
}

export const CONTEXT_EXAMPLES: ContextExample[] = [
  { sentence: ["I", "walked", "along", "the", "river", "bank"], bankIdx: 5, meaning: "river", color: "#10b981" },
  { sentence: ["I", "deposited", "money", "at", "the", "bank"], bankIdx: 5, meaning: "money", color: "#3b82f6" },
];

// 静态 vs contextualized 对比:cosine similarity
export const SIM_STATIC = 1.0;
export const SIM_ELMO = 0.42;

// 模拟 3 层输出(char / L1 / L2)在两个上下文里的向量差
// 用简化的 2D 投影表达"底层相似 / 顶层区分"
export interface LayerVector {
  layer: "char" | "L1" | "L2";
  riverBank: [number, number];    // 2D 投影
  moneyBank: [number, number];
}

export const LAYER_VECTORS: LayerVector[] = [
  { layer: "char", riverBank: [0.15, 0.10], moneyBank: [0.20, 0.15] },  // char 几乎一样
  { layer: "L1",   riverBank: [0.35, 0.55], moneyBank: [0.50, -0.30] }, // L1 开始分开
  { layer: "L2",   riverBank: [0.80, 0.75], moneyBank: [0.75, -0.75] }, // L2 完全分开
];
