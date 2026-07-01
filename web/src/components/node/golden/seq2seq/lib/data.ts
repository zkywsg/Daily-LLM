// BLEU vs 源句长度(信息瓶颈曲线)
export interface BleuLenPoint {
  length: number;
  bleu: number;
}
export const BLEU_VS_LENGTH: BleuLenPoint[] = [
  { length: 10, bleu: 35.8 },
  { length: 20, bleu: 35.0 },
  { length: 30, bleu: 33.0 },
  { length: 40, bleu: 30.5 },
  { length: 50, bleu: 28.0 },
  { length: 60, bleu: 25.5 },
  { length: 70, bleu: 24.0 },
];

// Sutskever 三件工程 trick 累加 BLEU
export interface TrickStep {
  label: string;
  bleu: number;
  delta: number;
}
export const TRICK_PROGRESSION: TrickStep[] = [
  { label: "基础(GRU+正序+greedy)", bleu: 28.0, delta: 0 },
  { label: "+ 4 层深 LSTM",          bleu: 31.0, delta: 3.0 },
  { label: "+ 倒序输入",             bleu: 33.5, delta: 2.5 },
  { label: "+ Beam Search",          bleu: 34.8, delta: 1.3 },
];
export const SMT_BASELINE = 33.3;

// encoder-decoder pipeline 演示句子
export const DEMO_SRC = ["I", "love", "cats"];
export const DEMO_TGT = ["我", "爱", "猫"];

// 倒序 vs 正序:x_1 到 c 的距离对比
export interface OrderDistance {
  order: "正序" | "倒序";
  tokens: string[];
  distanceToC: number[]; // 每个 token 距离 c 的步数
}
export const ORDER_COMPARE: OrderDistance[] = [
  { order: "正序", tokens: ["I", "love", "cats"], distanceToC: [3, 2, 1] },
  { order: "倒序", tokens: ["cats", "love", "I"], distanceToC: [3, 2, 1] },
];
