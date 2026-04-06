# Phase 02 · Language Track (2013–2019)

From words as coordinates, to context deciding meaning, to attention and pretraining turning language modeling into infrastructure.
This track converges with vision in 2021 around CLIP.

## Where This Problem Came From
> From 2013 to 2019, language modeling moved from “look at a local window” to “read the full context before generating.”
> Word2Vec, Seq2Seq, Attention, Transformer, and pretraining solved representation, memory, bottleneck, and transfer one step at a time.

## Learning Goals
By the end you should be able to answer: 1. Why fixed windows and pure recurrence were insufficient  2. How attention reduced the information bottleneck  3. Why pretraining turned NLP from task-specific models into reusable backbones

## 1. Intuition
Language is not like an image with stable local texture; it behaves more like a sequence that must be tracked over time.
RNNs first tried to compress history into state, but that state became harder and harder to preserve.
Attention let the model look back at the relevant positions at each step instead of forcing everything into one vector.
Pretraining then made “learn language first, adapt later” the default, so language models became foundations rather than single-purpose tools.

## 2. Mechanism
This track advances through four steps:
1. RNN / Seq2Seq solves “how do we model order?”
2. Attention solves “how do we revisit context on demand?”
3. Transformer solves “how do we remove recurrence and scale parallelism?”
4. Pretraining solves “how do we reuse language ability across tasks?”

## Contents

### [Recurrent Networks and Seq2Seq](recurrent-networks/README_EN.md)
RNN, LSTM, GRU, and the encoder-decoder paradigm
- Why fixed-window sequence methods fall short
- Recurrent state, BPTT, and long-range dependency issues
- Gated memory with LSTM / GRU
- How Seq2Seq created the pressure for Attention

### [Attention Mechanisms](attention-mechanisms/README_EN.md)
From Bahdanau Attention to Self-Attention
- The information bottleneck in Seq2Seq
- Bahdanau / Luong attention
- Self-attention and multi-head attention
- Positional encoding

### [Transformer Architecture](transformer-architecture/README_EN.md)
Dissecting *Attention Is All You Need*
- Encoder / Decoder structure
- Residual connections + LayerNorm
- Training tricks: warm-up, label smoothing

### [Pretrained Models](pretrained-models/README_EN.md)
BERT, GPT, and T5 compared
- ELMo as contextual word representations
- GPT-1/2 as autoregressive pretraining
- BERT as masked language modeling
- T5 as text-to-text unification
- Variants: RoBERTa, ALBERT, DistilBERT, XLNet

## Timeline Nodes

| Year | Milestone | Why It Mattered |
|------|-----------|-----------------|
| 2013 | Word2Vec | Established the idea of dense word embeddings with semantic geometry. |
| 2014 | GloVe / Seq2Seq / Attention | Global co-occurrence embeddings; end-to-end translation; the first fix for the bottleneck. |
| 2016 | FastText | Added subword-level embeddings and addressed OOV words. |
| 2017 | Transformer | Replaced recurrence with full attention and unlocked parallelism. |
| 2018 | ELMo / GPT-1 / BERT | Contextual embeddings; autoregressive pretraining; bidirectional masked pretraining. |
| 2019 | GPT-2 / T5 / RoBERTa | Scaled generation, unified text-to-text tasks, and showed BERT had been undertrained. |
| 2019 | ALBERT / DistilBERT | Parameter sharing for compression; distillation for lighter models. |

## 3. Engineering Pitfalls
- Fixed windows are too short: local context drops long-range dependencies immediately.
- Recurrent state is fragile: early information becomes harder to backpropagate through BPTT, and training is slower.
- Unidirectional encoders are narrow: left-to-right generation alone loses bidirectional context for understanding tasks.
- Pretraining and fine-tuning can diverge: the model learns general representations, but downstream work still needs aligned data and objectives.

## Evolution Notes
> This phase's legacy: language modeling became reusable representation learning; the new problems it left behind were scale, alignment, and multimodality.
→ See [03-Scale-Multimodal](../03-Scale-Multimodal/)

**Next Phase**: Phase 03 content is currently available only in Chinese. See the [03-Scale-Multimodal directory](../03-Scale-Multimodal/).
