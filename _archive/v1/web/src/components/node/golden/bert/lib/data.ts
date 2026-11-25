// BERT demo 数据:示例句子 + 手工 curated 的 MLM top-k 候选。
// 真要跑 BERT 推理太重,这里给一组让 viewer 摸到"双向上下文"在做什么的 demo。

export interface MaskPrediction {
  /** mask 位置(0-indexed,对应 tokens 数组下标) */
  pos: number;
  /** top-k 候选,按概率降序 */
  candidates: Array<{ word: string; prob: number }>;
}

export interface DemoSentence {
  tokens: string[];
  /** 每个可 mask 位置的 top-k 预测;key 是位置 idx */
  predictions: Record<number, MaskPrediction["candidates"]>;
}

export const DEMO_SENTENCES: DemoSentence[] = [
  {
    tokens: ["The", "cat", "sat", "on", "the", "mat", "."],
    predictions: {
      1: [
        { word: "cat", prob: 0.42 },
        { word: "dog", prob: 0.28 },
        { word: "child", prob: 0.08 },
        { word: "bird", prob: 0.06 },
        { word: "man", prob: 0.04 },
      ],
      2: [
        { word: "sat", prob: 0.51 },
        { word: "slept", prob: 0.18 },
        { word: "lay", prob: 0.12 },
        { word: "jumped", prob: 0.07 },
        { word: "stood", prob: 0.05 },
      ],
      5: [
        { word: "mat", prob: 0.35 },
        { word: "floor", prob: 0.22 },
        { word: "couch", prob: 0.15 },
        { word: "bed", prob: 0.11 },
        { word: "chair", prob: 0.08 },
      ],
    },
  },
  {
    tokens: ["She", "opened", "the", "book", "and", "started", "reading", "."],
    predictions: {
      1: [
        { word: "opened", prob: 0.36 },
        { word: "picked", prob: 0.21 },
        { word: "grabbed", prob: 0.12 },
        { word: "took", prob: 0.10 },
        { word: "held", prob: 0.07 },
      ],
      3: [
        { word: "book", prob: 0.48 },
        { word: "door", prob: 0.16 },
        { word: "window", prob: 0.11 },
        { word: "letter", prob: 0.09 },
        { word: "box", prob: 0.05 },
      ],
      6: [
        { word: "reading", prob: 0.62 },
        { word: "writing", prob: 0.11 },
        { word: "studying", prob: 0.08 },
        { word: "talking", prob: 0.06 },
        { word: "thinking", prob: 0.04 },
      ],
    },
  },
  {
    tokens: ["巴黎", "是", "法国", "的", "首都", "。"],
    predictions: {
      0: [
        { word: "巴黎", prob: 0.41 },
        { word: "伦敦", prob: 0.18 },
        { word: "罗马", prob: 0.13 },
        { word: "柏林", prob: 0.09 },
        { word: "马德里", prob: 0.06 },
      ],
      2: [
        { word: "法国", prob: 0.52 },
        { word: "英国", prob: 0.15 },
        { word: "德国", prob: 0.10 },
        { word: "意大利", prob: 0.08 },
        { word: "西班牙", prob: 0.05 },
      ],
      4: [
        { word: "首都", prob: 0.71 },
        { word: "中心", prob: 0.08 },
        { word: "象征", prob: 0.05 },
        { word: "标志", prob: 0.04 },
        { word: "心脏", prob: 0.03 },
      ],
    },
  },
];

// 三种 masking 策略的比例(原论文 15% 选中后):
//   80% → [MASK]
//   10% → 随机 token
//   10% → 保持原词
// 这是 BERT 解决 "训练-推理分布不匹配" 的关键招式。
export const MASKING_STRATEGY = [
  { label: "80% → [MASK]", pct: 80, color: "#ec4899", note: "标准 mask" },
  { label: "10% → 随机 token", pct: 10, color: "#f59e0b", note: "防止只学 [MASK]" },
  { label: "10% → 保持原词", pct: 10, color: "#10b981", note: "缩小训推差距" },
];
