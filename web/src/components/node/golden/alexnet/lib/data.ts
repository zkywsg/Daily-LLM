// AlexNet 各层 shape — 用于架构图
export interface LayerSpec {
  name: string;
  kind: "conv" | "pool" | "fc";
  outC: number;
  outHW: number; // H = W
  params: number; // 参数数量 (大约,M 级)
}

export const ALEXNET_LAYERS: LayerSpec[] = [
  { name: "Input",  kind: "pool", outC: 3,    outHW: 224, params: 0 },
  { name: "Conv1 11×11 s4", kind: "conv", outC: 96,  outHW: 55,  params: 0.035 },
  { name: "Pool1 3×3 s2",   kind: "pool", outC: 96,  outHW: 27,  params: 0 },
  { name: "Conv2 5×5",      kind: "conv", outC: 256, outHW: 27,  params: 0.614 },
  { name: "Pool2 3×3 s2",   kind: "pool", outC: 256, outHW: 13,  params: 0 },
  { name: "Conv3 3×3",      kind: "conv", outC: 384, outHW: 13,  params: 0.885 },
  { name: "Conv4 3×3",      kind: "conv", outC: 384, outHW: 13,  params: 1.327 },
  { name: "Conv5 3×3",      kind: "conv", outC: 256, outHW: 13,  params: 0.885 },
  { name: "Pool5 3×3 s2",   kind: "pool", outC: 256, outHW: 6,   params: 0 },
  { name: "FC6 4096",       kind: "fc",   outC: 4096, outHW: 1,  params: 37.75 },
  { name: "FC7 4096",       kind: "fc",   outC: 4096, outHW: 1,  params: 16.78 },
  { name: "FC8 1000",       kind: "fc",   outC: 1000, outHW: 1,  params: 4.10 },
];

// ImageNet ILSVRC 历年 Top-5 错误率
export interface ImageNetYear {
  year: number;
  method: string;
  topFive: number;
  isCNN: boolean;
}

export const IMAGENET_TIMELINE: ImageNetYear[] = [
  { year: 2010, method: "NEC-UIUC (SIFT+SVM)",    topFive: 28.2, isCNN: false },
  { year: 2011, method: "XRCE (Fisher Vector)",    topFive: 25.8, isCNN: false },
  { year: 2012, method: "AlexNet",                 topFive: 16.4, isCNN: true },
  { year: 2013, method: "ZFNet",                    topFive: 11.7, isCNN: true },
  { year: 2014, method: "VGG / GoogLeNet",          topFive:  6.7, isCNN: true },
  { year: 2015, method: "ResNet-152",               topFive:  3.6, isCNN: true },
];

// 激活函数评估
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function sigmoidGrad(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

export function tanhGrad(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

export function relu(x: number): number {
  return Math.max(0, x);
}

export function reluGrad(x: number): number {
  return x > 0 ? 1 : 0;
}
