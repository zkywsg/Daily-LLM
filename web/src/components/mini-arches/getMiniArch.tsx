import type { ComponentType } from "react";
import type { MiniArchProps } from "./types";
import {
  MiniLeNet,
  MiniAlexNet,
  MiniVGG,
  MiniGoogLeNet,
  MiniResNet,
  MiniDenseNet,
  MiniEfficientNet,
  MiniConvNeXt,
} from ".";

const MAP: Record<string, ComponentType<MiniArchProps>> = {
  "01-lenet": MiniLeNet,
  "02-alexnet": MiniAlexNet,
  "03-vgg": MiniVGG,
  "04-inception": MiniGoogLeNet,
  "05-resnet": MiniResNet,
  "06-densenet": MiniDenseNet,
  "07-efficientnet": MiniEfficientNet,
  "08-convnext": MiniConvNeXt,
};

export function getMiniArch(
  nodePath: string
): ComponentType<MiniArchProps> | null {
  const slug = nodePath.split("/").pop()?.replace(/\.md$/, "") ?? "";
  return MAP[slug] ?? null;
}
