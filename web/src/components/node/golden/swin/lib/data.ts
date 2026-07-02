// Window attention 复杂度对比 (patch 数越大差距越大)
export interface ComplexityRow {
  resolution: number;
  patches: number;
  fullAttnOps: number;    // (patches)^2
  windowAttnOps: number;  // patches * windowSize^2 (window=7 → 49)
}
export function computeComplexity(resolution: number): ComplexityRow {
  const patches = (resolution / 4) ** 2; // patch size 4
  return {
    resolution,
    patches,
    fullAttnOps: patches ** 2,
    windowAttnOps: patches * 49,
  };
}
export const COMPLEXITY_TABLE: ComplexityRow[] = [224, 384, 512, 800].map(computeComplexity);

// Swin 4 stage 规格
export interface StageSpec {
  name: string;
  resolution: number;
  channels: number;
  blocks: number;
}
export const SWIN_STAGES: StageSpec[] = [
  { name: "Stage 1", resolution: 56, channels: 96,  blocks: 2 },
  { name: "Stage 2", resolution: 28, channels: 192, blocks: 2 },
  { name: "Stage 3", resolution: 14, channels: 384, blocks: 6 },
  { name: "Stage 4", resolution: 7,  channels: 768, blocks: 2 },
];

// 模型规格表
export interface ModelSpec {
  name: string;
  depths: string;
  channels: number;
  params: number; // M
  flops: number;  // G
}
export const MODEL_SPECS: ModelSpec[] = [
  { name: "Swin-T", depths: "(2,2,6,2)",  channels: 96,  params: 28,  flops: 4.5 },
  { name: "Swin-S", depths: "(2,2,18,2)", channels: 96,  params: 50,  flops: 8.7 },
  { name: "Swin-B", depths: "(2,2,18,2)", channels: 128, params: 88,  flops: 15.4 },
  { name: "Swin-L", depths: "(2,2,18,2)", channels: 192, params: 197, flops: 34.5 },
];

// 多任务性能对比: classification / detection / segmentation
export interface TaskCompareRow {
  task: string;
  metric: string;
  resnet: number;
  swin: number;
}
export const TASK_COMPARE: TaskCompareRow[] = [
  { task: "ImageNet 分类",     metric: "top-1", resnet: 76.1, swin: 81.3 },
  { task: "COCO Detection",    metric: "box mAP", resnet: 38.6, swin: 46.0 },
  { task: "ADE20K Segmentation", metric: "mIoU", resnet: 44.9, swin: 45.8 },
];
