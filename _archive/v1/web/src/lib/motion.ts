import type { Variants } from "framer-motion";

export const duration = {
  fast: 0.15,
  base: 0.25,
  slow: 0.4,
} as const;

// framer-motion 期望 cubic bezier 是 mutable number[];
// 用普通数组(不加 as const)避免每个调用点都要 `as unknown as number[]` 强转
export const ease: Record<"out" | "inOut" | "spring", [number, number, number, number]> = {
  out: [0.16, 1, 0.3, 1],
  inOut: [0.65, 0, 0.35, 1],
  spring: [0.34, 1.56, 0.64, 1],
};

export const fadeUp: Variants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

// 从光标位置 pop 出来:scale + fade,配合 transform-origin: top left
// 让小卡看起来"从指针处展开",而不是滑入
export const popScale: Variants = {
  initial: { opacity: 0, scale: 0.6 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.7 },
};

export const fadeIn: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
};

export const staggerContainer: Variants = {
  animate: { transition: { staggerChildren: 0.05 } },
};
