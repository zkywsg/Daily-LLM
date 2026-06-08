import { lazy } from "react";
import type { ComponentType, LazyExoticComponent } from "react";

export const goldenSamples: Record<
  string,
  LazyExoticComponent<ComponentType>
> = {
  "01-cnn/05-resnet": lazy(() => import("./resnet/NodePageResNet")),
  // 未来新增其他金标本：
  // "05-transformer/01-transformer": lazy(() => import("./transformer/NodePageTransformer")),
};
