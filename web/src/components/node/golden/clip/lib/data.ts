// CLIP demo 数据。真要跑 CLIP 推理太重 —— 这里用 emoji 当"图像缩略图",
// 配 caption 文本,并给一组手工挑的 2D 投影坐标让 viewer 看到"图文对齐"是什么。

export interface ImageTextPair {
  /** 用 emoji 代替图像缩略图(viewer 看得见) */
  emoji: string;
  caption: string;
  /** 2D 投影坐标(给 SharedSpace widget 用),image 和 text 应该聚得很近 */
  imagePos: { x: number; y: number };
  textPos: { x: number; y: number };
  /** 给类别区分上色 */
  group: "animal" | "vehicle" | "food";
}

// 6 对图文,3 类(动物/交通工具/食物)。每对的 imagePos 和 textPos 故意挨近
// (差几像素的扰动),让 viewer 看到"对齐"效果。
export const PAIRS: ImageTextPair[] = [
  {
    emoji: "🐱",
    caption: "a photo of a cat",
    imagePos: { x: 120, y: 80 },
    textPos: { x: 132, y: 90 },
    group: "animal",
  },
  {
    emoji: "🐶",
    caption: "a photo of a dog",
    imagePos: { x: 180, y: 100 },
    textPos: { x: 168, y: 115 },
    group: "animal",
  },
  {
    emoji: "🐦",
    caption: "a photo of a bird",
    imagePos: { x: 240, y: 70 },
    textPos: { x: 252, y: 85 },
    group: "animal",
  },
  {
    emoji: "🚗",
    caption: "a photo of a car",
    imagePos: { x: 380, y: 180 },
    textPos: { x: 370, y: 195 },
    group: "vehicle",
  },
  {
    emoji: "🚲",
    caption: "a photo of a bicycle",
    imagePos: { x: 440, y: 200 },
    textPos: { x: 452, y: 215 },
    group: "vehicle",
  },
  {
    emoji: "🍕",
    caption: "a photo of a pizza",
    imagePos: { x: 540, y: 80 },
    textPos: { x: 525, y: 95 },
    group: "food",
  },
];

export const GROUP_COLOR: Record<ImageTextPair["group"], string> = {
  animal: "#ec4899",
  vehicle: "#3b82f6",
  food: "#f59e0b",
};

// Zero-shot 演示:模型从未见过这些类别,直接用 prompt template 推理。
// 每个 emoji 配 5 个候选类别,手工 curated 相似度让 viewer 看到推理过程。
export interface ZeroShotExample {
  emoji: string;
  trueLabel: string;
  /** 候选类别 → similarity score(余弦相似度,-1..1)。降序排好。 */
  candidates: Array<{ label: string; sim: number }>;
}

export const ZERO_SHOT_EXAMPLES: ZeroShotExample[] = [
  {
    emoji: "🐱",
    trueLabel: "cat",
    candidates: [
      { label: "cat", sim: 0.32 },
      { label: "dog", sim: 0.21 },
      { label: "tiger", sim: 0.18 },
      { label: "rabbit", sim: 0.14 },
      { label: "car", sim: 0.03 },
    ],
  },
  {
    emoji: "🚗",
    trueLabel: "car",
    candidates: [
      { label: "car", sim: 0.41 },
      { label: "truck", sim: 0.28 },
      { label: "bicycle", sim: 0.19 },
      { label: "motorcycle", sim: 0.16 },
      { label: "cat", sim: 0.02 },
    ],
  },
  {
    emoji: "🍕",
    trueLabel: "pizza",
    candidates: [
      { label: "pizza", sim: 0.45 },
      { label: "bread", sim: 0.24 },
      { label: "sandwich", sim: 0.20 },
      { label: "cake", sim: 0.15 },
      { label: "car", sim: 0.01 },
    ],
  },
];

// Prompt template 对比 —— CLIP 论文重要发现:不同 prompt 影响 zero-shot 精度
export const PROMPT_TEMPLATES = [
  { template: "{class}", accuracy: 0.589, note: "裸类别名 — 模糊" },
  { template: "a photo of a {class}", accuracy: 0.633, note: "原始 CLIP 推荐" },
  { template: "a blurry photo of a {class}", accuracy: 0.601, note: "误导性形容词" },
  { template: "an ensemble of 80 prompts", accuracy: 0.683, note: "prompt ensemble (论文 §3.1.4)" },
];
