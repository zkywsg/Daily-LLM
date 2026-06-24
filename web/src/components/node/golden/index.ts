import { lazy } from "react";
import type { ComponentType, LazyExoticComponent } from "react";

export const goldenSamples: Record<
  string,
  LazyExoticComponent<ComponentType>
> = {
  "01-cnn/05-resnet": lazy(() => import("./resnet/NodePageResNet")),
  "05-transformer/01-transformer": lazy(
    () => import("./transformer/NodePageTransformer"),
  ),
  "10-diffusion/01-ddpm": lazy(() => import("./ddpm/NodePageDDPM")),
  "11-peft-lora/03-lora": lazy(() => import("./lora/NodePageLoRA")),
  "06-bert-family/01-bert": lazy(() => import("./bert/NodePageBERT")),
};
