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
  MiniRNN,
  MiniLSTM,
  MiniGRU,
  MiniSeq2Seq,
  MiniAttention,
  MiniTransformer,
  MiniTransformerXL,
  MiniSparseAttention,
  MiniRoPE,
  MiniFlashAttention,
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
  "01-rnn": MiniRNN,
  "02-lstm": MiniLSTM,
  "03-gru": MiniGRU,
  "04-seq2seq": MiniSeq2Seq,
  "05-attention": MiniAttention,
  "01-transformer": MiniTransformer,
  "02-transformer-xl": MiniTransformerXL,
  "03-sparse-attention": MiniSparseAttention,
  "04-rope": MiniRoPE,
  "05-flash-attention": MiniFlashAttention,
};

export function getMiniArch(
  nodePath: string
): ComponentType<MiniArchProps> | null {
  const slug = nodePath.split("/").pop()?.replace(/\.md$/, "") ?? "";
  return MAP[slug] ?? null;
}
