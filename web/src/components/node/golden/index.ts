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
  "14-rag-agent/01-rag": lazy(() => import("./rag/NodePageRAG")),
  "13-moe-efficient/03-mixtral": lazy(() => import("./mixtral/NodePageMixtral")),
  "02-rnn-lstm/02-lstm": lazy(() => import("./lstm/NodePageLSTM")),
  "03-word-embedding/01-word2vec": lazy(() => import("./word2vec/NodePageWord2Vec")),
  "01-cnn/02-alexnet": lazy(() => import("./alexnet/NodePageAlexNet")),
  "10-diffusion/02-ldm": lazy(() => import("./ldm/NodePageLDM")),
  "07-gpt-scaling/02-gpt2": lazy(() => import("./gpt2/NodePageGPT2")),
  "12-rlhf-alignment/04-dpo": lazy(() => import("./dpo/NodePageDPO")),
  "02-rnn-lstm/05-attention": lazy(() => import("./bahdanau/NodePageBahdanau")),
  "10-diffusion/05-dit": lazy(() => import("./dit/NodePageDiT")),
  "07-gpt-scaling/04-scaling-laws": lazy(() => import("./scaling-laws/NodePageScalingLaws")),
  "04-gan/04-stylegan": lazy(() => import("./stylegan/NodePageStyleGAN")),
  "03-word-embedding/04-elmo": lazy(() => import("./elmo/NodePageELMo")),
  "07-gpt-scaling/01-gpt1": lazy(() => import("./gpt1/NodePageGPT1")),
  "02-rnn-lstm/03-gru": lazy(() => import("./gru/NodePageGRU")),
  "02-rnn-lstm/04-seq2seq": lazy(() => import("./seq2seq/NodePageSeq2Seq")),
  "10-diffusion/03-imagen": lazy(() => import("./imagen/NodePageImagen")),
  "09-multimodal-clip/03-flamingo": lazy(() => import("./flamingo/NodePageFlamingo")),
  "08-vit/03-swin": lazy(() => import("./swin/NodePageSwin")),
  "05-transformer/04-rope": lazy(() => import("./rope/NodePageRoPE")),
  "02-rnn-lstm/01-rnn": lazy(() => import("./rnn/NodePageRNN")),
};
