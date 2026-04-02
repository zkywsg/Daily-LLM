# Phase 02 · Language Track (2013-2019)

From words as coordinates, to context-dependent meaning, to attention replacing recurrence.
This track converges with vision in 2021 around CLIP.

Phase 02 is where language modeling stops treating text as a short local context problem and becomes a representation-learning problem. The sequence here matters: recurrence introduces reusable state, attention removes the fixed-vector bottleneck, and pretraining turns these architectures into general-purpose language backbones.

## Timeline Highlights

| Year | Milestone | Why It Mattered |
|------|-----------|-----------------|
| 2013 | Word2Vec | Showed that words could be learned as dense vectors with meaningful geometry. |
| 2014 | Seq2Seq + Attention era begins | Encoder-decoder models made sequence generation practical, and attention exposed the limits of fixed-length context. |
| 2017 | Transformer | Replaced recurrence with fully attention-based sequence modeling and far better parallelism. |
| 2018 | BERT | Made bidirectional pretraining a default recipe for language understanding. |
| 2018 | GPT-1 | Established large-scale autoregressive pretraining as a foundation for generation. |

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

**Next Phase**: English landing page not yet available for Phase 03 (`../03-Scale-Multimodal/README_EN.md` is missing).
