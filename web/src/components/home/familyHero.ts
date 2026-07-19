import type { FamilyId } from "../../types/family";

// 每个家族挑一个最具代表性的工作作为 FamilyGridView 卡片上的视觉锚定。
// 用 mini-arches 已有的卡通图,所以挑的节点必须在 getMiniArch 映射里
// (目前 67 节点全有)。选择口径:开宗立派 / 最被熟知 / 最常被引用。
export const FAMILY_HERO: Record<FamilyId, string> = {
  "01-cnn": "01-cnn/05-resnet.md",
  "02-rnn-lstm": "02-rnn-lstm/02-lstm.md",
  "03-word-embedding": "03-word-embedding/01-word2vec.md",
  "04-gan": "04-gan/01-gan.md",
  "05-transformer": "05-transformer/01-transformer.md",
  "06-bert-family": "06-bert-family/01-bert.md",
  "07-gpt-scaling": "07-gpt-scaling/03-gpt3.md",
  "08-vit": "08-vit/01-vit.md",
  "09-multimodal-clip": "09-multimodal-clip/01-clip.md",
  "10-diffusion": "10-diffusion/02-ldm.md",
  "11-peft-lora": "11-peft-lora/03-lora.md",
  "12-rlhf-alignment": "12-rlhf-alignment/02-instructgpt.md",
  "13-moe-efficient": "13-moe-efficient/03-mixtral.md",
  "14-rag-agent": "14-rag-agent/01-rag.md",
  "15-reasoning-o1-r1": "15-reasoning-o1-r1/03-o1.md",
  "16-world-models": "16-world-models/01-world-models.md",
};
