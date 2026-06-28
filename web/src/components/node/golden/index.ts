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
  "09-multimodal-clip/01-clip": lazy(() => import("./clip/NodePageCLIP")),
  "07-gpt-scaling/03-gpt3": lazy(() => import("./gpt3/NodePageGPT3")),
  "04-gan/01-gan": lazy(() => import("./gan/NodePageGAN")),
  "08-vit/01-vit": lazy(() => import("./vit/NodePageViT")),
  "15-reasoning-o1-r1/01-cot": lazy(() => import("./cot/NodePageCoT")),
  "12-rlhf-alignment/02-instructgpt": lazy(() => import("./instructgpt/NodePageInstructGPT")),
};
