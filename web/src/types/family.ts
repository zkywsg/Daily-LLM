// AUTO-CONSUMED. Schema must match scripts/generate_timeline.py output.

export type FamilyId =
  | "01-cnn"
  | "02-rnn-lstm"
  | "03-word-embedding"
  | "04-gan"
  | "05-transformer"
  | "06-bert-family"
  | "07-gpt-scaling"
  | "08-vit"
  | "09-multimodal-clip"
  | "10-diffusion"
  | "11-peft-lora"
  | "12-rlhf-alignment"
  | "13-moe-efficient"
  | "14-rag-agent"
  | "15-reasoning-o1-r1";

export interface NodeData {
  name: string;
  year: number;
  family: FamilyId;
  order: number;
  paper: string;
  authors: string[];
  key_idea: string;
  path: string; // repo-root-relative, e.g. "01-cnn/05-resnet.md"
  assets: string[]; // repo-root-relative SVG paths
}

export interface FamilyData {
  id: FamilyId;
  label: string;
  blurb: string;
  yearRange: [number, number] | null;
  colorToken: string; // "--family-NN"
  nodes: NodeData[]; // sorted by order asc
}

export interface FamiliesData {
  generatedAt: string;
  families: FamilyData[];
}
