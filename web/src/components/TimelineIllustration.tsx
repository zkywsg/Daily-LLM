import type * as React from "react";

/**
 * 按年份渲染的"完整架构示意图"。
 * 设计标准（v2）：
 *  - 所有图统一 viewBox 宽度 1100，高度按内容定（360–740）
 *  - 全部使用 illustration__svg--tall，破 card 内边距吃满横向空间
 *  - 文字尺寸：label 14 / strong 17 / small 12（CSS 控制）
 *  - 每张图以"完整结构"为目标：不再做简化示意，而是把每一层 / 每一个连接 / 每一个公式都画出来
 */

type TimelineIllustrationProps = {
  year: string;
};

export function TimelineIllustration({ year }: TimelineIllustrationProps) {
  const renderer = ILLUSTRATIONS[year];
  if (!renderer) return null;

  return (
    <figure className="illustration" aria-label={renderer.caption}>
      {renderer.render()}
      <figcaption>{renderer.caption}</figcaption>
    </figure>
  );
}

type Renderer = { caption: string; render: () => React.ReactElement };

const ILLUSTRATIONS: Record<string, Renderer> = {
  "2012": {
    caption:
      "AlexNet 完整 8 层：输入 224² → 5 个 Conv（含 ReLU/MaxPool）→ 3 个 FC → 1000 类 Softmax。逐层标注 spatial 维度 × 通道数",
    render: () => <AlexNetDiagram />,
  },
  "2013": {
    caption:
      "Word2Vec：Skip-gram 用中心词预测上下文窗口；学到的稠密向量使得 king − man + woman ≈ queen 这样的线性类比成立",
    render: () => <Word2VecDiagram />,
  },
  "2014": {
    caption:
      "GAN 完整对抗训练回路：z → G(z) 假样本 vs 真样本同时进入 D；D 最大化辨别能力、G 最小化被识破概率，两人零和博弈到纳什均衡",
    render: () => <GanDiagram />,
  },
  "2015": {
    caption:
      "ResNet：单个残差块内部 Conv-BN-ReLU-Conv-BN ⊕ identity skip；恒等捷径让梯度直通，使得 152 层网络可训练（相比 VGG-19 是 8× 深度）",
    render: () => <ResNetDiagram />,
  },
  "2016": {
    caption:
      "AlphaGo · 两套神经网络（策略网络选择落子 + 价值网络评估胜率）+ MCTS 搜索 + 自我对弈 RL，把围棋这种天文搜索空间问题压到可计算",
    render: () => <AlphaGoDiagram />,
  },
  "2017": {
    caption:
      "Transformer 编码器一个完整 block：token → 嵌入 + 位置编码 → Q/K/V 投影 → 多头注意力 → Add&Norm → FFN → Add&Norm；右上 ×6 表示这个 block 串联六次",
    render: () => <TransformerDiagram />,
  },
  "2018": {
    caption:
      "BERT vs GPT-1：BERT 用双向 Transformer 编码器做 MLM（猜被 mask 的 token），GPT-1 用单向解码器做自回归（猜下一个 token）；都先大规模预训练，再小数据集微调",
    render: () => <BertGptDiagram />,
  },
  "2019": {
    caption:
      "T5 text-to-text 范式：翻译 / 摘要 / 情感 / 问答 全部统一为「task: 输入 → 输出」字符串接口，一个模型一套 API 处理所有 NLP 任务；GPT-2 同时证明 zero-shot 可行",
    render: () => <Gpt2T5Diagram />,
  },
  "2020": {
    caption:
      "GPT-3 与 Scaling Laws：参数从 117M 涨到 175B 沿幂律提升能力；右侧 in-context few-shot 提示让大模型免微调完成新任务",
    render: () => <ScalingDiagram />,
  },
  "2021": {
    caption:
      "CLIP 完整对比训练：N 张图 + N 段文本各自编码 → N×N 余弦相似度矩阵，对角线为正样本，其余 N²−N 个为负样本，InfoNCE 损失双向拉近",
    render: () => <ClipDiagram />,
  },
  "2022": {
    caption:
      "RLHF 三阶段：① SFT 用人写的演示监督微调 → ② Reward Model 学人类对多个回答的偏好排序 → ③ PPO 用 RM 当奖励信号 + KL 约束更新策略",
    render: () => <RlhfDiagram />,
  },
  "2023": {
    caption:
      "2023 同年双轨：左 GPT-4 闭源能力跃迁（多模态推理、长上下文、专家级表现）；右 LLaMA 把权重开源 → 引爆 Alpaca / Vicuna / Code-LLaMA 等社区微调浪潮",
    render: () => <Gpt4LlamaDiagram />,
  },
  "2024": {
    caption:
      "MoE 稀疏激活：每个 token 由 Router 选 top-2 expert FFN（共 8 个），输出按 router 权重加权聚合；每步只激活 25% 参数，容量大但推理便宜",
    render: () => <MoeDiagram />,
  },
  "2025": {
    caption:
      "DeepSeek R1 · 用「可验证奖励」（数学答案、代码 unit test）替代 RLHF 的 reward model；模型在推理时生成长链 chain-of-thought，算力开始从训练期向推理期迁移",
    render: () => <DeepSeekR1Diagram />,
  },
};

/* =====================================================================
 * Shared: <defs> arrowhead marker
 * 每张 SVG 顶部都需要它来画带箭头的连接线
 * ===================================================================== */
function ArrowDefs() {
  return (
    <defs>
      <marker
        id="arrow-head"
        viewBox="0 0 10 10"
        refX="8"
        refY="5"
        markerWidth="6"
        markerHeight="6"
        orient="auto-start-reverse"
      >
        <path d="M 0 0 L 10 5 L 0 10 z" className="illustration__arrow-head" />
      </marker>
    </defs>
  );
}

/* =====================================================================
 * 2012 — AlexNet 完整 8 层 CNN
 * ===================================================================== */
function AlexNetDiagram() {
  // 8 个 conv/fc 层，逐层给出名字、空间尺寸、通道数
  const layers = [
    { label: "Input", spatial: "224×224", ch: 3, x: 50, w: 60, color: "input" },
    { label: "Conv1 (11×11, s=4) + ReLU + Pool", spatial: "55×55→27×27", ch: 96, x: 150, w: 50, color: "conv" },
    { label: "Conv2 (5×5) + Pool", spatial: "27×27→13×13", ch: 256, x: 280, w: 42, color: "conv" },
    { label: "Conv3 (3×3)", spatial: "13×13", ch: 384, x: 400, w: 36, color: "conv" },
    { label: "Conv4 (3×3)", spatial: "13×13", ch: 384, x: 510, w: 36, color: "conv" },
    { label: "Conv5 (3×3) + Pool", spatial: "13×13→6×6", ch: 256, x: 620, w: 28, color: "conv" },
    { label: "FC1", spatial: "—", ch: 4096, x: 730, w: 14, color: "fc" },
    { label: "FC2", spatial: "—", ch: 4096, x: 810, w: 14, color: "fc" },
    { label: "FC3 + Softmax", spatial: "—", ch: 1000, x: 900, w: 14, color: "fc" },
  ];

  // 每层显示的视觉高度：随通道数 log 比例缩放，但有上下限
  const layerH = (ch: number) => Math.max(40, Math.min(120, 24 + Math.log2(ch) * 12));

  return (
    <svg viewBox="0 0 1100 460" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      {/* 大标题 */}
      <text x="30" y="30" className="illustration__label illustration__label--strong">
        AlexNet · 8 层（5 Conv + 3 FC）· 60M 参数 · 在 ImageNet 上把 Top-5 误差打到 15.3%
      </text>

      {/* 每层立方体（用 3 个错位矩形叠加表示通道维度） */}
      {layers.map((L, i) => {
        const h = layerH(L.ch);
        const yMid = 200;
        const yTop = yMid - h / 2;
        const layerCls =
          L.color === "input"
            ? "illustration__layer illustration__layer--input"
            : L.color === "conv"
            ? "illustration__layer illustration__layer--conv"
            : "illustration__layer illustration__layer--fc";
        return (
          <g key={L.label} style={{ animationDelay: `${i * 80}ms` }} className="illustration__appear">
            {/* 三层错位矩形，做 3D 感 */}
            <rect x={L.x + 6} y={yTop - 6} width={L.w} height={h} rx="3" className={`${layerCls} illustration__layer--back`} />
            <rect x={L.x + 3} y={yTop - 3} width={L.w} height={h} rx="3" className={`${layerCls} illustration__layer--mid`} />
            <rect x={L.x} y={yTop} width={L.w} height={h} rx="3" className={layerCls} />

            {/* 通道数顶部 */}
            <text x={L.x + L.w / 2} y={yTop - 14} textAnchor="middle" className="illustration__label illustration__label--small">
              {L.ch} ch
            </text>
            {/* 空间尺寸底部 */}
            <text x={L.x + L.w / 2} y={yTop + h + 16} textAnchor="middle" className="illustration__label illustration__label--small">
              {L.spatial}
            </text>
          </g>
        );
      })}

      {/* 层名（旋转 35°，避免拥挤） */}
      {layers.map((L, i) => (
        <text
          key={`name-${i}`}
          x={L.x + L.w / 2}
          y="358"
          textAnchor="end"
          className="illustration__label illustration__label--small"
          transform={`rotate(-30 ${L.x + L.w / 2} 358)`}
        >
          {L.label}
        </text>
      ))}

      {/* 层间流动箭头 */}
      {layers.slice(0, -1).map((L, i) => {
        const next = layers[i + 1];
        const x1 = L.x + L.w + 4;
        const x2 = next.x - 4;
        return (
          <line
            key={`flow-${i}`}
            x1={x1}
            y1="200"
            x2={x2}
            y2="200"
            className="illustration__flow"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        );
      })}

      {/* 底部公式与说明 */}
      <text x="30" y="402" className="illustration__label">
        关键创新：① ReLU 替代 sigmoid → 收敛 6× 快  ② GPU 并行训练 5–6 天  ③ Dropout 0.5 在 FC1/FC2  ④ 数据增强（裁剪+镜像+PCA 颜色抖动）
      </text>
      <text x="30" y="424" className="illustration__label illustration__label--small">
        感受野从 11×11 局部边缘 → 经过 5 层卷积 → 13×13 特征图每个位置已"看到"整张图的语义；FC 把 9216 维空间特征压成 1000 类概率
      </text>

      {/* 输出：Top-5 概率柱状（cat/dog/car/...） */}
      <g transform="translate(960, 160)">
        <text x="0" y="-8" className="illustration__label illustration__label--small">Top-5 输出</text>
        {[
          { label: "cat", v: 0.78 },
          { label: "dog", v: 0.12 },
          { label: "car", v: 0.05 },
          { label: "...", v: 0.02 },
        ].map((p, i) => (
          <g key={p.label}>
            <rect
              x="0"
              y={i * 20}
              width={p.v * 70 + 4}
              height="14"
              rx="2"
              className="illustration__bar"
              style={{ animationDelay: `${800 + i * 80}ms` }}
            />
            <text x={p.v * 70 + 10} y={i * 20 + 12} className="illustration__label illustration__label--small">
              {p.label} {(p.v * 100).toFixed(0)}%
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2013 — Word2Vec Skip-gram + 类比向量
 * ===================================================================== */
function Word2VecDiagram() {
  return (
    <svg viewBox="0 0 1100 500" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        Word2Vec Skip-gram · 用中心词预测窗口内的每个上下文词，副产品就是稠密词向量
      </text>

      {/* 训练样本：句子 + 滑动窗口（窗口大小 c=2） */}
      <text x="30" y="68" className="illustration__label illustration__label--strong">
        ① 训练样本（窗口 c=2）
      </text>
      {["the", "quick", "brown", "fox", "jumps", "over", "lazy"].map((w, i) => (
        <g key={w} transform={`translate(${60 + i * 78}, 84)`}>
          <rect
            width="68"
            height="32"
            rx="6"
            className={i === 3 ? "illustration__block illustration__block--alt illustration__token--center" : "illustration__block"}
          />
          <text x="34" y="22" textAnchor="middle" className="illustration__block-label">
            {w}
          </text>
          {/* 窗口范围圈出 */}
        </g>
      ))}
      {/* 窗口高亮（覆盖 fox 左右各 2 词） */}
      <rect x="60" y="78" width={5 * 78} height="44" rx="10" className="illustration__window" />
      <text x={60 + 5 * 78 / 2} y="138" textAnchor="middle" className="illustration__label illustration__label--small">
        中心词 = fox，正样本 = (fox→the), (fox→quick), (fox→brown), (fox→jumps), (fox→over)
      </text>

      {/* ② Skip-gram 网络结构 */}
      <text x="30" y="180" className="illustration__label illustration__label--strong">
        ② Skip-gram 网络（无隐藏层非线性，只有两个嵌入矩阵）
      </text>

      {/* one-hot 输入 */}
      <g transform="translate(60, 210)">
        <text x="0" y="-6" className="illustration__label illustration__label--small">one-hot |V|</text>
        {Array.from({ length: 8 }).map((_, i) => (
          <rect
            key={i}
            x="0"
            y={i * 10}
            width="40"
            height="8"
            rx="1"
            className={i === 3 ? "illustration__featuremap illustration__featuremap--ctx" : "illustration__attn-cell illustration__attn-cell--raw"}
          />
        ))}
        <text x="48" y="38" className="illustration__label illustration__label--small">fox</text>
      </g>

      {/* W (输入嵌入矩阵 |V|×d) */}
      <g transform="translate(180, 210)">
        <line x1="-60" y1="40" x2="0" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <rect width="84" height="80" rx="6" className="illustration__proj illustration__proj--q" />
        <text x="42" y="36" textAnchor="middle" className="illustration__block-label">W</text>
        <text x="42" y="54" textAnchor="middle" className="illustration__label illustration__label--small">|V|×d</text>
      </g>

      {/* 中心词向量 v_fox */}
      <g transform="translate(300, 210)">
        <line x1="-16" y1="40" x2="0" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="0" y="-6" className="illustration__label illustration__label--small">v_fox (d=300)</text>
        {Array.from({ length: 10 }).map((_, i) => (
          <rect key={i} x={i * 7} y={20 + (i % 3) * 20} width="5" height="40" rx="1" className="illustration__featuremap" style={{ animationDelay: `${i * 30}ms` }} />
        ))}
      </g>

      {/* W' (输出嵌入矩阵 d×|V|) */}
      <g transform="translate(440, 210)">
        <line x1="-60" y1="40" x2="0" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <rect width="84" height="80" rx="6" className="illustration__proj illustration__proj--v" />
        <text x="42" y="36" textAnchor="middle" className="illustration__block-label">W'</text>
        <text x="42" y="54" textAnchor="middle" className="illustration__label illustration__label--small">d×|V|</text>
      </g>

      {/* softmax 输出 = 上下文词概率 */}
      <g transform="translate(560, 210)">
        <line x1="-16" y1="40" x2="0" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="0" y="-6" className="illustration__label illustration__label--small">softmax over |V|</text>
        {[
          { w: "the", v: 0.22 },
          { w: "quick", v: 0.18 },
          { w: "brown", v: 0.25 },
          { w: "jumps", v: 0.20 },
          { w: "over", v: 0.10 },
          { w: "其他", v: 0.05 },
        ].map((p, i) => (
          <g key={p.w}>
            <rect x="0" y={i * 14} width={p.v * 160 + 4} height="10" rx="1" className="illustration__bar" style={{ animationDelay: `${400 + i * 60}ms` }} />
            <text x={p.v * 160 + 10} y={i * 14 + 8} className="illustration__label illustration__label--small">{p.w}</text>
          </g>
        ))}
      </g>

      {/* 损失：cross-entropy 目标 = 让上下文词概率最大 */}
      <text x="800" y="252" className="illustration__label illustration__label--small">
        loss = − Σ log p(context | center)
      </text>
      <text x="800" y="272" className="illustration__label illustration__label--small">
        实践用 negative sampling 加速
      </text>

      {/* ③ 学到的向量空间 —— 类比关系 */}
      <text x="30" y="360" className="illustration__label illustration__label--strong">
        ③ 学到的向量空间 · 副产品：语义关系变成线性方向
      </text>

      {/* 2D 类比平面 */}
      <g transform="translate(60, 380)">
        <line x1="0" y1="80" x2="500" y2="80" className="illustration__rail" />
        <line x1="0" y1="80" x2="0" y2="0" className="illustration__rail" />
        <text x="500" y="98" className="illustration__label illustration__label--small">性别 (gender)</text>
        <text x="-4" y="-4" className="illustration__label illustration__label--small" textAnchor="end">王 (royalty)</text>

        {/* 4 个词向量点 */}
        {[
          { x: 90, y: 60, label: "man" },
          { x: 360, y: 60, label: "woman" },
          { x: 90, y: 18, label: "king" },
          { x: 360, y: 18, label: "queen" },
        ].map((p, i) => (
          <g key={p.label}>
            <circle cx={p.x} cy={p.y} r="6" className="illustration__neuron illustration__neuron--big" style={{ animationDelay: `${600 + i * 100}ms` }} />
            <text x={p.x + 10} y={p.y + 4} className="illustration__label">{p.label}</text>
          </g>
        ))}

        {/* 向量箭头：king - man + woman = queen */}
        <line x1="90" y1="60" x2="360" y2="60" className="illustration__residual" markerEnd="url(#arrow-head)" />
        <text x="225" y="76" textAnchor="middle" className="illustration__label illustration__label--small">man → woman</text>

        <line x1="90" y1="18" x2="360" y2="18" className="illustration__residual" markerEnd="url(#arrow-head)" />
        <text x="225" y="14" textAnchor="middle" className="illustration__label illustration__label--small">king → queen</text>

        <line x1="90" y1="60" x2="90" y2="18" className="illustration__branch illustration__branch--v" markerEnd="url(#arrow-head)" />
        <line x1="360" y1="60" x2="360" y2="18" className="illustration__branch illustration__branch--v" markerEnd="url(#arrow-head)" />
      </g>

      {/* 公式 */}
      <g transform="translate(640, 400)">
        <rect width="430" height="78" rx="10" className="illustration__block illustration__block--alt" />
        <text x="20" y="28" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>
          v(king) − v(man) + v(woman) ≈ v(queen)
        </text>
        <text x="20" y="52" className="illustration__label illustration__label--small">
          两条"性别向量"平行；两条"王室向量"平行 —— 语义关系是空间中的方向，不再依赖人工规则。
        </text>
        <text x="20" y="70" className="illustration__label illustration__label--small">
          这是 NLP 第一次有"可迁移的语义表示"，奠定后续所有预训练语言模型的范式。
        </text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2014 — GAN 完整训练回路
 * ===================================================================== */
function GanDiagram() {
  return (
    <svg viewBox="0 0 1100 520" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        GAN · 生成器 G 与判别器 D 的零和博弈
      </text>

      {/* 上半：真实数据通路 */}
      <text x="30" y="68" className="illustration__label">
        ① 真实数据通路
      </text>
      <g transform="translate(80, 80)">
        <text x="0" y="-6" className="illustration__label illustration__label--small">真实图像 x ~ p_data</text>
        {[0, 1, 2].map((i) => (
          <rect key={i} x={i * 38 + i * 4} y="0" width="36" height="36" rx="4" className="illustration__featuremap illustration__featuremap--ctx" />
        ))}
      </g>

      {/* 下半：噪声 → G → 假数据 */}
      <text x="30" y="160" className="illustration__label">
        ② 假数据生成通路
      </text>
      <g transform="translate(80, 174)">
        <text x="0" y="-6" className="illustration__label illustration__label--small">z ~ N(0,1) · 100 维</text>
        {Array.from({ length: 16 }).map((_, i) => (
          <circle
            key={i}
            cx={(i % 8) * 14 + 6}
            cy={Math.floor(i / 8) * 14 + 6}
            r="5"
            className="illustration__neuron illustration__neuron--noise"
            style={{ animationDelay: `${i * 40}ms` }}
          />
        ))}
      </g>

      <g transform="translate(240, 168)">
        <line x1="-32" y1="20" x2="0" y2="20" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <rect width="120" height="60" rx="8" className="illustration__block illustration__block--alt" />
        <text x="60" y="32" textAnchor="middle" className="illustration__block-label">Generator G</text>
        <text x="60" y="48" textAnchor="middle" className="illustration__label illustration__label--small">deconv / transposed conv</text>
      </g>

      <g transform="translate(390, 174)">
        <line x1="-30" y1="20" x2="0" y2="20" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="0" y="-6" className="illustration__label illustration__label--small">G(z) · 假图像</text>
        {[0, 1, 2].map((i) => (
          <rect
            key={i}
            x={i * 38 + i * 4}
            y="0"
            width="36"
            height="36"
            rx="4"
            className="illustration__pixel illustration__pixel--big"
            style={{ animationDelay: `${i * 100}ms` }}
          />
        ))}
      </g>

      {/* 两路汇入 D */}
      <path d="M 200 100 C 460 100, 460 280, 560 280" className="illustration__arrow" fill="none" markerEnd="url(#arrow-head)" />
      <path d="M 510 196 C 540 196, 540 290, 560 290" className="illustration__arrow" fill="none" markerEnd="url(#arrow-head)" />

      <g transform="translate(560, 250)">
        <rect width="140" height="70" rx="8" className="illustration__block illustration__block--alt" />
        <text x="70" y="32" textAnchor="middle" className="illustration__block-label">Discriminator D</text>
        <text x="70" y="50" textAnchor="middle" className="illustration__label illustration__label--small">conv → sigmoid</text>
      </g>

      {/* D 输出 real/fake 概率 */}
      <g transform="translate(740, 280)">
        <line x1="-40" y1="10" x2="0" y2="10" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="0" y="-2" className="illustration__label illustration__label--small">D(·) ∈ [0,1]</text>
        <rect x="0" y="6" width="110" height="14" rx="2" className="illustration__bar illustration__bar--real" style={{ animationDelay: "300ms" }} />
        <text x="116" y="18" className="illustration__label illustration__label--small">P(real)</text>
        <rect x="0" y="28" width="56" height="14" rx="2" className="illustration__bar illustration__bar--fake" style={{ animationDelay: "400ms" }} />
        <text x="62" y="40" className="illustration__label illustration__label--small">P(fake)</text>
      </g>

      {/* 两个 loss 公式块 */}
      <g transform="translate(80, 360)">
        <rect width="460" height="120" rx="10" className="illustration__block" />
        <text x="20" y="26" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-alignment-ink)" }}>
          D 的目标（最大化）
        </text>
        <text x="20" y="56" className="illustration__label">
          max  E[ log D(x) ]  +  E[ log(1 − D(G(z))) ]
        </text>
        <text x="20" y="82" className="illustration__label illustration__label--small">
          真图片 → 推到 1，假图片 → 推到 0
        </text>
        <text x="20" y="102" className="illustration__label illustration__label--small">
          每步先用 mini-batch 真 + mini-batch 假更新 D
        </text>
      </g>

      <g transform="translate(560, 360)">
        <rect width="460" height="120" rx="10" className="illustration__block" />
        <text x="20" y="26" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-multimodal-ink)" }}>
          G 的目标（最小化）
        </text>
        <text x="20" y="56" className="illustration__label">
          min  E[ log(1 − D(G(z))) ]
        </text>
        <text x="20" y="82" className="illustration__label illustration__label--small">
          实际常用 max E[ log D(G(z)) ] 缓解早期梯度消失
        </text>
        <text x="20" y="102" className="illustration__label illustration__label--small">
          再用同样 mini-batch 假更新 G（D 冻结）
        </text>
      </g>

      {/* 反向梯度示意 */}
      <path d="M 810 360 C 870 240, 760 240, 700 320" className="illustration__feedback" fill="none" markerEnd="url(#arrow-head)" />
      <text x="900" y="338" className="illustration__label illustration__label--small">
        ∂loss/∂G 反向回传到 G
      </text>
    </svg>
  );
}

/* =====================================================================
 * 2015 — ResNet 残差块详图 + 152 层堆叠
 * ===================================================================== */
function ResNetDiagram() {
  return (
    <svg viewBox="0 0 1100 520" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        ResNet · 残差块内部 + 152 层堆叠：恒等捷径让任意深度都可训
      </text>

      {/* 左半：单个 residual block 完整流程 */}
      <text x="30" y="64" className="illustration__label illustration__label--strong">
        ① 单个残差块（Basic Block）
      </text>

      <g transform="translate(30, 80)">
        {/* 输入 x */}
        <g transform="translate(40, 30)">
          <rect width="60" height="30" rx="6" className="illustration__block illustration__block--alt" />
          <text x="30" y="20" textAnchor="middle" className="illustration__block-label">x</text>
        </g>
        <line x1="100" y1="45" x2="120" y2="45" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        {/* 主路径：Conv1 → BN → ReLU → Conv2 → BN */}
        {[
          { label: "Conv 3×3", y: 16 },
          { label: "BN", y: 70 },
          { label: "ReLU", y: 124 },
          { label: "Conv 3×3", y: 178 },
          { label: "BN", y: 232 },
        ].map((b, i) => (
          <g key={i} transform={`translate(120, ${b.y})`}>
            <rect width="120" height="38" rx="6" className={i === 0 || i === 3 ? "illustration__proj illustration__proj--ffn" : i === 2 ? "illustration__proj illustration__proj--act" : "illustration__block illustration__block--alt"} />
            <text x="60" y="24" textAnchor="middle" className="illustration__block-label">{b.label}</text>
            {i < 4 && <line x1="60" y1="40" x2="60" y2={b.y + 56 - b.y} className="illustration__arrow" />}
            {i < 4 && <line x1="60" y1="40" x2="60" y2="54" className="illustration__arrow" markerEnd="url(#arrow-head)" />}
          </g>
        ))}

        {/* ⊕ Add */}
        <g transform="translate(160, 290)">
          <circle r="16" cx="20" cy="16" className="illustration__addnorm" />
          <text x="20" y="22" textAnchor="middle" className="illustration__block-label">⊕</text>
        </g>

        {/* 残差跳过：从 x 直接绕到 ⊕ */}
        <path d="M 70 60 C 12 60, 12 306, 160 306" className="illustration__residual" fill="none" markerEnd="url(#arrow-head)" />
        <text x="18" y="180" className="illustration__label illustration__label--small" transform="rotate(-90 18 180)">
          identity shortcut · 不增参
        </text>

        {/* F(x) 的箭头汇入 ⊕ */}
        <line x1="180" y1="270" x2="180" y2="290" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="240" y="200" className="illustration__label illustration__label--small">F(x)</text>

        {/* ⊕ 后接 ReLU + 输出 */}
        <line x1="180" y1="322" x2="180" y2="338" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(120, 338)">
          <rect width="120" height="34" rx="6" className="illustration__proj illustration__proj--act" />
          <text x="60" y="22" textAnchor="middle" className="illustration__block-label">ReLU</text>
        </g>
        <line x1="180" y1="372" x2="180" y2="388" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(150, 388)">
          <rect width="60" height="30" rx="6" className="illustration__block illustration__block--alt" />
          <text x="30" y="20" textAnchor="middle" className="illustration__block-label">H(x)</text>
        </g>
      </g>

      {/* 公式区 */}
      <g transform="translate(290, 130)">
        <rect width="320" height="120" rx="10" className="illustration__block illustration__block--alt" />
        <text x="20" y="32" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>
          H(x) = F(x) + x
        </text>
        <text x="20" y="58" className="illustration__label">
          ∂L/∂x = ∂L/∂H · (1 + ∂F/∂x)
        </text>
        <text x="20" y="84" className="illustration__label illustration__label--small">
          就算 ∂F/∂x → 0，梯度仍能通过"1"直通回去 ——
        </text>
        <text x="20" y="102" className="illustration__label illustration__label--small">
          这就是为什么 ResNet 把网络深度从 20 层推到 152 层
        </text>
      </g>

      {/* 右半：152 层堆叠（与 VGG-19 高度对比） */}
      <text x="650" y="64" className="illustration__label illustration__label--strong">
        ② 残差块串联得到 ResNet-152
      </text>

      {/* VGG-19 参照（19 层） */}
      <g transform="translate(650, 90)">
        <text x="0" y="0" className="illustration__label illustration__label--small">VGG-19（参照）</text>
        {Array.from({ length: 19 }).map((_, i) => (
          <rect key={i} x="0" y={8 + i * 6} width="44" height="4" rx="1" className="illustration__layer illustration__layer--fc" />
        ))}
        <text x="0" y={8 + 19 * 6 + 16} className="illustration__label illustration__label--small">19 层 · 训不深</text>
      </g>

      {/* ResNet-152 */}
      <g transform="translate(750, 90)">
        <text x="0" y="0" className="illustration__label illustration__label--small">ResNet-152</text>
        {Array.from({ length: 50 }).map((_, i) => (
          <rect key={i} x="0" y={8 + i * 6} width="44" height="4" rx="1" className="illustration__layer illustration__layer--conv" />
        ))}
        <text x="0" y={8 + 50 * 6 + 16} className="illustration__label illustration__label--small">前 50 个 Block …</text>
      </g>

      {/* ResNet-152 第二列剩余 */}
      <g transform="translate(850, 90)">
        <text x="0" y="0" className="illustration__label illustration__label--small">（同上，简化）</text>
        {Array.from({ length: 50 }).map((_, i) => (
          <rect key={i} x="0" y={8 + i * 6} width="44" height="4" rx="1" className="illustration__layer illustration__layer--conv" />
        ))}
        <text x="0" y={8 + 50 * 6 + 16} className="illustration__label illustration__label--small">152 层 · 训得动</text>
      </g>

      {/* 关键结果 */}
      <g transform="translate(640, 440)">
        <rect width="430" height="64" rx="10" className="illustration__block" />
        <text x="20" y="26" className="illustration__label">
          ImageNet Top-5 误差：VGG-19  7.3%  →  ResNet-152  3.57%
        </text>
        <text x="20" y="50" className="illustration__label illustration__label--small">
          首次低于人类（5.1%）；让"网络越深越好"成为可行的工程方向
        </text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2017 — Transformer 完整 encoder block（保留之前 bd26c3f 的版本）
 * ===================================================================== */
function TransformerDiagram() {
  const tokens = ["The", "cat", "sat", "on", "mat"];
  const tokenXs = [120, 240, 360, 480, 600];
  const tokenW = 56;

  const mainHead = [
    [0.92, 0.04, 0.02, 0.01, 0.01],
    [0.18, 0.62, 0.14, 0.04, 0.02],
    [0.08, 0.3, 0.46, 0.13, 0.03],
    [0.02, 0.06, 0.18, 0.58, 0.16],
    [0.01, 0.02, 0.06, 0.22, 0.69],
  ];

  return (
    <svg viewBox="0 0 1100 760" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        ① 输入 token 序列
      </text>
      {tokens.map((t, i) => (
        <g key={`tok-${i}`} transform={`translate(${tokenXs[i] - tokenW / 2}, 44)`} style={{ animationDelay: `${i * 90}ms` }} className="illustration__appear">
          <rect width={tokenW} height="30" rx="6" className="illustration__block illustration__block--alt" />
          <text x={tokenW / 2} y="20" className="illustration__block-label" textAnchor="middle">{t}</text>
        </g>
      ))}

      {tokenXs.map((x, i) => (
        <line key={`a-${i}`} x1={x} y1="78" x2={x} y2="108" className="illustration__arrow" markerEnd="url(#arrow-head)" />
      ))}

      <text x="30" y="124" className="illustration__label illustration__label--strong">② 嵌入 (d=512)</text>
      {tokenXs.map((x, i) =>
        Array.from({ length: 8 }).map((_, k) => (
          <rect key={`e-${i}-${k}`} x={x - 24 + k * 6} y="116" width="4" height="28" rx="1" className="illustration__pixel" style={{ animationDelay: `${200 + i * 60 + k * 20}ms` }} />
        )),
      )}

      <text x="30" y="170" className="illustration__label illustration__label--strong">③ + 位置编码 (sin/cos)</text>
      <PositionalWave x={90} y={172} width={620} />
      <text x="730" y="180" className="illustration__label">让序列顺序进入向量空间</text>

      <line x1="360" y1="200" x2="360" y2="232" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g>
        <rect x="40" y="232" width="1020" height="468" rx="14" className="illustration__group" />
        <g className="illustration__badge">
          <rect x="940" y="222" width="92" height="26" rx="13" />
          <text x="986" y="240" textAnchor="middle">× 6 blocks</text>
        </g>
        <text x="56" y="252" className="illustration__label illustration__label--strong">④ Encoder Block · 内部完整流程</text>
      </g>

      <rect x="58" y="262" width="984" height="232" rx="10" className="illustration__group illustration__group--inner" />
      <text x="72" y="280" className="illustration__label illustration__label--strong">a. Multi-Head Self-Attention</text>

      <g transform="translate(72, 292)">
        <text x="0" y="-2" className="illustration__label">X (输入)</text>
        {Array.from({ length: 5 }).map((_, r) => (
          <rect key={`x-${r}`} x="0" y={r * 14 + 6} width="56" height="11" rx="2" className="illustration__featuremap" style={{ animationDelay: `${400 + r * 50}ms` }} />
        ))}
      </g>

      <g transform="translate(140, 296)">
        <path d="M 0 30 C 30 30, 50 0, 80 0" className="illustration__branch illustration__branch--q" fill="none" markerEnd="url(#arrow-head)" />
        <path d="M 0 30 L 80 30" className="illustration__branch illustration__branch--k" fill="none" markerEnd="url(#arrow-head)" />
        <path d="M 0 30 C 30 30, 50 60, 80 60" className="illustration__branch illustration__branch--v" fill="none" markerEnd="url(#arrow-head)" />

        {(["W_Q", "W_K", "W_V"] as const).map((lbl, i) => (
          <g key={lbl} transform={`translate(82, ${i * 30 - 8})`}>
            <rect width="42" height="20" rx="4" className={`illustration__proj illustration__proj--${["q", "k", "v"][i]}`} />
            <text x="21" y="14" textAnchor="middle" className="illustration__block-label illustration__block-label--small">{lbl}</text>
          </g>
        ))}

        {(["Q", "K", "V"] as const).map((lbl, i) => (
          <g key={lbl} transform={`translate(134, ${i * 30 - 8})`}>
            <line x1="0" y1="10" x2="20" y2="10" className="illustration__arrow" markerEnd="url(#arrow-head)" />
            <text x="30" y="14" className="illustration__label illustration__label--inline">{lbl}</text>
          </g>
        ))}
      </g>

      <g transform="translate(360, 296)">
        <text x="0" y="-4" className="illustration__label">每头：scaled dot-product attention</text>

        <g transform="translate(0, 4)">
          <text x="36" y="-2" className="illustration__label illustration__label--small" textAnchor="middle">Q·Kᵀ / √d</text>
          {mainHead.map((row, r) =>
            row.map((_, c) => (
              <rect key={`qk-${r}-${c}`} x={c * 14} y={r * 14 + 4} width="12" height="12" rx="1.5" className="illustration__attn-cell illustration__attn-cell--raw" style={{ animationDelay: `${600 + (r * 5 + c) * 25}ms` }} />
            )),
          )}
        </g>

        <line x1="78" y1="44" x2="98" y2="44" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="88" y="38" textAnchor="middle" className="illustration__label illustration__label--small">softmax</text>

        <g transform="translate(102, 4)">
          <text x="36" y="-2" className="illustration__label illustration__label--small" textAnchor="middle">attention 概率</text>
          {mainHead.map((row, r) =>
            row.map((v, c) => (
              <rect key={`sm-${r}-${c}`} x={c * 14} y={r * 14 + 4} width="12" height="12" rx="1.5" className="illustration__attn-cell" style={{ opacity: 0.18 + v * 0.82, animationDelay: `${900 + (r * 5 + c) * 25}ms` }} />
            )),
          )}
        </g>

        <line x1="180" y1="44" x2="200" y2="44" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="190" y="38" textAnchor="middle" className="illustration__label illustration__label--small">@V</text>

        <g transform="translate(204, 4)">
          <text x="20" y="-2" className="illustration__label illustration__label--small" textAnchor="middle">head 输出</text>
          {Array.from({ length: 5 }).map((_, r) => (
            <rect key={`ho-${r}`} x="0" y={r * 14 + 4} width="40" height="12" rx="2" className="illustration__featuremap" style={{ animationDelay: `${1200 + r * 40}ms` }} />
          ))}
        </g>
      </g>

      <g transform="translate(660, 300)">
        <text x="0" y="-4" className="illustration__label">8 头并行</text>
        {Array.from({ length: 7 }).map((_, i) => (
          <g key={i} transform={`translate(${i * 5}, ${i * 5})`} style={{ opacity: 1 - i * 0.1 }}>
            <rect width="52" height="72" rx="4" className="illustration__head-stack" />
          </g>
        ))}
        <text x="80" y="40" className="illustration__label illustration__label--small">每头独立子空间</text>
        <text x="80" y="56" className="illustration__label illustration__label--small">建模不同关系</text>
      </g>

      <g transform="translate(820, 296)">
        <line x1="-20" y1="40" x2="0" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <rect width="78" height="32" y="24" rx="6" className="illustration__block illustration__block--alt" />
        <text x="39" y="44" textAnchor="middle" className="illustration__block-label">Concat</text>
        <text x="39" y="64" textAnchor="middle" className="illustration__label illustration__label--small">8·d/8 = d</text>

        <line x1="78" y1="40" x2="98" y2="40" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(98, 24)">
          <rect width="78" height="32" rx="6" className="illustration__proj illustration__proj--o" />
          <text x="39" y="20" textAnchor="middle" className="illustration__block-label">W_O</text>
        </g>
      </g>

      <line x1="552" y1="494" x2="552" y2="510" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g>
        <path d="M 100 332 C 24 332, 24 528, 460 528" className="illustration__residual" fill="none" markerEnd="url(#arrow-head)" />
        <text x="20" y="430" className="illustration__label illustration__label--small" transform="rotate(-90 20 430)">residual (恒等捷径)</text>

        <g transform="translate(478, 510)">
          <circle r="14" cx="14" cy="14" className="illustration__addnorm" />
          <text x="14" y="19" textAnchor="middle" className="illustration__block-label">⊕</text>
          <line x1="28" y1="14" x2="48" y2="14" className="illustration__arrow" markerEnd="url(#arrow-head)" />
          <rect x="50" y="0" width="88" height="28" rx="6" className="illustration__block illustration__block--alt" />
          <text x="94" y="18" textAnchor="middle" className="illustration__block-label">LayerNorm</text>
        </g>
      </g>

      <line x1="572" y1="538" x2="572" y2="560" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <rect x="58" y="560" width="984" height="80" rx="10" className="illustration__group illustration__group--inner" />
      <text x="72" y="578" className="illustration__label illustration__label--strong">c. Feed-Forward Network (position-wise，每个位置独立两层 MLP)</text>
      <g transform="translate(180, 596)">
        <rect width="160" height="32" rx="6" className="illustration__proj illustration__proj--ffn" />
        <text x="80" y="20" textAnchor="middle" className="illustration__block-label">Linear · d → 4d</text>

        <line x1="160" y1="16" x2="200" y2="16" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(200, 0)">
          <rect width="92" height="32" rx="6" className="illustration__proj illustration__proj--act" />
          <text x="46" y="20" textAnchor="middle" className="illustration__block-label">GELU</text>
        </g>

        <line x1="292" y1="16" x2="332" y2="16" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(332, 0)">
          <rect width="160" height="32" rx="6" className="illustration__proj illustration__proj--ffn" />
          <text x="80" y="20" textAnchor="middle" className="illustration__block-label">Linear · 4d → d</text>
        </g>

        <text x="540" y="20" className="illustration__label illustration__label--small">扩张 → 非线性 → 收缩</text>
      </g>

      <line x1="552" y1="644" x2="552" y2="658" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g>
        <path d="M 600 528 C 980 528, 980 670, 620 670" className="illustration__residual" fill="none" markerEnd="url(#arrow-head)" />
        <text x="990" y="600" className="illustration__label illustration__label--small" transform="rotate(-90 990 600)">residual</text>

        <g transform="translate(478, 660)">
          <circle r="14" cx="14" cy="14" className="illustration__addnorm" />
          <text x="14" y="19" textAnchor="middle" className="illustration__block-label">⊕</text>
          <line x1="28" y1="14" x2="48" y2="14" className="illustration__arrow" markerEnd="url(#arrow-head)" />
          <rect x="50" y="0" width="88" height="28" rx="6" className="illustration__block illustration__block--alt" />
          <text x="94" y="18" textAnchor="middle" className="illustration__block-label">LayerNorm</text>
        </g>
      </g>

      <line x1="572" y1="690" x2="572" y2="716" className="illustration__arrow" markerEnd="url(#arrow-head)" />
      <text x="30" y="734" className="illustration__label illustration__label--strong">⑤ 输出 · 每个位置已聚合全局上下文（与输入同形状，可直接喂下一个 block）</text>
      {tokenXs.map((x, i) =>
        Array.from({ length: 8 }).map((_, k) => (
          <rect key={`out-${i}-${k}`} x={x - 24 + k * 6} y="724" width="4" height="28" rx="1" className="illustration__featuremap illustration__featuremap--ctx" style={{ animationDelay: `${1500 + i * 60 + k * 20}ms` }} />
        )),
      )}
    </svg>
  );
}

function PositionalWave({ x, y, width }: { x: number; y: number; width: number }) {
  const samples = 80;
  const amp = 10;
  const points = (phase: number) =>
    Array.from({ length: samples }, (_, i) => {
      const t = i / (samples - 1);
      const px = x + t * width;
      const py = y + 12 + Math.sin(t * Math.PI * 4 + phase) * amp;
      return `${i === 0 ? "M" : "L"} ${px} ${py}`;
    }).join(" ");

  return (
    <g>
      <path d={points(0)} className="illustration__wave illustration__wave--sin" fill="none" />
      <path d={points(Math.PI / 2)} className="illustration__wave illustration__wave--cos" fill="none" />
    </g>
  );
}

/* =====================================================================
 * 2018 — BERT MLM vs GPT-1 自回归
 * ===================================================================== */
function BertGptDiagram() {
  return (
    <svg viewBox="0 0 1100 600" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        2018 同年双星 · BERT（双向编码器 MLM）vs GPT-1（单向解码器自回归）
      </text>

      {/* === 左半：BERT === */}
      <text x="40" y="64" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-language-ink)" }}>
        BERT · Bidirectional Encoder
      </text>

      {/* BERT 输入：带 [MASK] 的句子 */}
      {["The", "[MASK]", "sat", "on", "the", "[MASK]"].map((t, i) => {
        const x = 50 + i * 76;
        const masked = t === "[MASK]";
        return (
          <g key={`bt-${i}`} transform={`translate(${x}, 80)`}>
            <rect width="68" height="28" rx="6" className={masked ? "illustration__token--masked" : "illustration__block illustration__block--alt"} />
            <text x="34" y="18" textAnchor="middle" className="illustration__block-label illustration__block-label--small">{t}</text>
          </g>
        );
      })}
      <text x="40" y="128" className="illustration__label illustration__label--small">
        随机 mask 15% 的 token，让模型从上下文反推
      </text>

      {/* 双向 attention 示意（每个 token 看所有 token） */}
      <g transform="translate(40, 152)">
        <text x="0" y="0" className="illustration__label illustration__label--small">双向注意力（每个位置可看左右）</text>
        {Array.from({ length: 6 }).map((_, i) => (
          <circle key={`bn-${i}`} cx={i * 76 + 44} cy="20" r="6" className="illustration__neuron" />
        ))}
        {/* 每个节点和其他所有节点连线 */}
        {Array.from({ length: 6 }).flatMap((_, i) =>
          Array.from({ length: 6 }).map((_, j) =>
            i !== j ? (
              <line
                key={`bl-${i}-${j}`}
                x1={i * 76 + 44}
                y1="20"
                x2={j * 76 + 44}
                y2="20"
                className="illustration__attn-link"
              />
            ) : null,
          ),
        )}
      </g>

      {/* BERT Transformer 编码器堆叠 */}
      <g transform="translate(50, 196)">
        <rect width="450" height="44" rx="8" className="illustration__block illustration__block--alt" />
        <text x="225" y="20" textAnchor="middle" className="illustration__block-label">12 × Transformer Encoder Block</text>
        <text x="225" y="36" textAnchor="middle" className="illustration__label illustration__label--small">d=768 · 12 头 · ~110M 参数</text>
      </g>

      {/* MLM 预测头：在 [MASK] 位置预测原 token */}
      <g transform="translate(50, 256)">
        <text x="0" y="0" className="illustration__label illustration__label--small">MLM 头 · 只在 [MASK] 位置算 softmax</text>
        {/* 预测概率分布 */}
        <g transform="translate(76, 12)">
          <rect width="56" height="22" rx="4" className="illustration__bar illustration__bar--real" />
          <text x="62" y="16" className="illustration__label illustration__label--small">cat 0.87</text>
        </g>
        <g transform="translate(382, 12)">
          <rect width="48" height="22" rx="4" className="illustration__bar illustration__bar--real" />
          <text x="54" y="16" className="illustration__label illustration__label--small">mat 0.73</text>
        </g>
        <text x="0" y="58" className="illustration__label illustration__label--small">
          loss = − log p(cat | The, _, sat, on, the, _) − log p(mat | …)
        </text>
      </g>

      {/* === 右半：GPT-1 === */}
      <text x="600" y="64" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-scale-ink)" }}>
        GPT-1 · Causal (left-to-right) Decoder
      </text>

      {/* GPT 输入：完整序列 */}
      {["The", "cat", "sat", "on", "the"].map((t, i) => (
        <g key={`gt-${i}`} transform={`translate(${610 + i * 80}, 80)`}>
          <rect width="68" height="28" rx="6" className="illustration__block illustration__block--alt" />
          <text x="34" y="18" textAnchor="middle" className="illustration__block-label illustration__block-label--small">{t}</text>
        </g>
      ))}
      {/* 预测目标：next token */}
      <g transform="translate(1010, 80)">
        <rect width="68" height="28" rx="6" className="illustration__token--target" />
        <text x="34" y="18" textAnchor="middle" className="illustration__block-label illustration__block-label--small">?</text>
      </g>

      <text x="600" y="128" className="illustration__label illustration__label--small">
        每个位置只能看到自己 + 左边，预测右邻 token
      </text>

      {/* causal mask 三角矩阵 */}
      <g transform="translate(600, 152)">
        <text x="0" y="0" className="illustration__label illustration__label--small">causal mask（下三角 1，上三角 0）</text>
        {Array.from({ length: 5 }).map((_, r) =>
          Array.from({ length: 5 }).map((_, c) => (
            <rect
              key={`cm-${r}-${c}`}
              x={c * 18}
              y={r * 18 + 8}
              width="16"
              height="16"
              rx="1.5"
              className={c <= r ? "illustration__attn-cell" : "illustration__attn-cell illustration__attn-cell--masked"}
              style={{ opacity: c <= r ? 0.7 : 0.1 }}
            />
          )),
        )}
      </g>

      <g transform="translate(610, 196)">
        <rect width="450" height="44" rx="8" className="illustration__block illustration__block--alt" />
        <text x="225" y="20" textAnchor="middle" className="illustration__block-label">12 × Transformer Decoder Block</text>
        <text x="225" y="36" textAnchor="middle" className="illustration__label illustration__label--small">d=768 · 12 头 · ~117M 参数</text>
      </g>

      {/* GPT 输出：自回归预测下一个 token */}
      <g transform="translate(610, 256)">
        <text x="0" y="0" className="illustration__label illustration__label--small">语言模型头 · 每个位置都算 softmax over |V|</text>
        <g transform="translate(326, 12)">
          <rect width="60" height="22" rx="4" className="illustration__bar illustration__bar--real" />
          <text x="66" y="16" className="illustration__label illustration__label--small">mat 0.42</text>
        </g>
        <text x="0" y="58" className="illustration__label illustration__label--small">
          loss = − Σ log p(t_(i+1) | t_1, …, t_i)
        </text>
      </g>

      {/* === 底部：共同的两阶段范式 === */}
      <g transform="translate(40, 340)">
        <rect width="1020" height="240" rx="14" className="illustration__group" />
        <text x="20" y="28" className="illustration__label illustration__label--strong">
          共同范式 · "大规模无标注预训练 + 小规模任务微调" 自此成为 NLP 标配
        </text>

        {/* 阶段 1：预训练 */}
        <g transform="translate(40, 56)">
          <rect width="440" height="160" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="26" className="illustration__label illustration__label--strong">Stage 1 · 预训练（无监督）</text>
          <text x="20" y="52" className="illustration__label illustration__label--small">数据：BooksCorpus + Wikipedia · ~3.3B tokens</text>
          <text x="20" y="72" className="illustration__label illustration__label--small">目标：MLM (BERT) / 下一 token (GPT)</text>
          <text x="20" y="92" className="illustration__label illustration__label--small">硬件：BERT 16× TPU 4 天 / GPT 8× P600 一个月</text>
          <text x="20" y="120" className="illustration__label">→ 得到通用语义表示（参数 ≈ 100M）</text>
          <text x="20" y="138" className="illustration__label illustration__label--small">这套权重就是后来所有"预训练模型"的祖先</text>
        </g>

        {/* 阶段 2：微调 */}
        <g transform="translate(540, 56)">
          <rect width="440" height="160" rx="10" className="illustration__block" />
          <text x="20" y="26" className="illustration__label illustration__label--strong">Stage 2 · 下游微调（监督）</text>
          <text x="20" y="52" className="illustration__label illustration__label--small">数据：SST-2 / SQuAD / GLUE 等任务标注集</text>
          <text x="20" y="72" className="illustration__label illustration__label--small">改造：加一个小分类/回归 head</text>
          <text x="20" y="92" className="illustration__label illustration__label--small">训练：所有参数一起更新，但只需几个 epoch</text>
          <text x="20" y="120" className="illustration__label">→ 同一权重适配 11+ 个任务，刷榜</text>
          <text x="20" y="138" className="illustration__label illustration__label--small">GLUE 平均分 + 7 pp，重塑 NLP benchmark 格局</text>
        </g>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2020 — Scaling Laws + few-shot prompt 示例
 * ===================================================================== */
function ScalingDiagram() {
  // 4 个 GPT 系列模型规模
  const models = [
    { x: 110, y: 280, params: "117M", name: "GPT-1" },
    { x: 290, y: 220, params: "1.5B", name: "GPT-2" },
    { x: 470, y: 150, params: "13B", name: "GPT-3 medium" },
    { x: 640, y: 60, params: "175B", name: "GPT-3" },
  ];
  // 拟合的幂律曲线点
  const curve = Array.from({ length: 50 }, (_, i) => {
    const x = 80 + i * 12;
    const y = 320 - Math.pow(i / 50, 0.5) * 280;
    return [x, y] as const;
  });
  const path = curve.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x} ${y}`).join(" ");

  return (
    <svg viewBox="0 0 1100 480" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        Scaling Laws · 模型参数、训练数据、算力按幂律共增 → 不只是变好，是出现新能力
      </text>

      {/* 左半：log-log 缩放曲线 */}
      <g>
        {/* 坐标轴 */}
        <line x1="80" y1="320" x2="720" y2="320" className="illustration__rail" />
        <line x1="80" y1="40" x2="80" y2="320" className="illustration__rail" />

        <text x="720" y="340" textAnchor="end" className="illustration__label illustration__label--small">
          参数量 N (log) →
        </text>
        <text x="76" y="40" textAnchor="end" className="illustration__label illustration__label--small">
          能力 ↑
        </text>

        {/* 曲线 */}
        <path d={path} className="illustration__curve" fill="none" />

        {/* 4 个模型点（点 + 标签错开避免压住曲线） */}
        {models.map((m, i) => (
          <g key={m.name} style={{ animationDelay: `${i * 120}ms` }}>
            <circle cx={m.x} cy={m.y} r="7" className="illustration__neuron illustration__neuron--big" />
            <line x1={m.x} y1={m.y - 8} x2={m.x} y2={m.y - 28} className="illustration__arrow" />
            <rect x={m.x - 56} y={m.y - 56} width="112" height="22" rx="4" className="illustration__block illustration__block--alt" />
            <text x={m.x} y={m.y - 40} textAnchor="middle" className="illustration__block-label illustration__block-label--small">
              {m.name} · {m.params}
            </text>
          </g>
        ))}

        {/* 涌现能力标记区域 */}
        <rect x="600" y="50" width="120" height="50" rx="6" className="illustration__group illustration__group--inner" />
        <text x="660" y="74" textAnchor="middle" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
          ≥ 100B 后
        </text>
        <text x="660" y="92" textAnchor="middle" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
          能力出现"涌现"
        </text>

        {/* 幂律公式 */}
        <text x="80" y="370" className="illustration__label">
          L(N) = (N_c / N)^α  · 损失随参数 N 服从幂律下降，跨 8 个数量级仍成立
        </text>
        <text x="80" y="392" className="illustration__label illustration__label--small">
          Kaplan et al. 2020 · Chinchilla 后修正：算力固定时数据应该比参数多扩 2×
        </text>
      </g>

      {/* 右半：few-shot prompt 示例（in-context learning） */}
      <g transform="translate(770, 50)">
        <rect width="300" height="380" rx="10" className="illustration__block" />
        <text x="20" y="26" className="illustration__label illustration__label--strong">
          in-context few-shot 示例
        </text>
        <text x="20" y="46" className="illustration__label illustration__label--small">
          175B 模型不更新参数，仅靠 prompt 完成新任务
        </text>

        {/* 任务描述 */}
        <g transform="translate(20, 60)">
          <rect width="260" height="32" rx="4" className="illustration__block illustration__block--alt" />
          <text x="10" y="14" className="illustration__label illustration__label--small">任务：英语 → 法语翻译</text>
          <text x="10" y="28" className="illustration__label illustration__label--small">（无任何梯度更新）</text>
        </g>

        {/* 3 个 in-context 示例 */}
        {[
          ["sea otter", "loutre de mer"],
          ["plush giraffe", "girafe en peluche"],
          ["cheese", "fromage"],
        ].map(([en, fr], i) => (
          <g key={i} transform={`translate(20, ${108 + i * 50})`}>
            <rect width="260" height="40" rx="4" className="illustration__block illustration__block--alt" />
            <text x="10" y="16" className="illustration__label illustration__label--small">"{en} →"</text>
            <text x="10" y="32" className="illustration__label illustration__label--small" style={{ fill: "var(--phase-alignment-ink)" }}>
              → {fr}
            </text>
          </g>
        ))}

        {/* Query */}
        <g transform="translate(20, 260)">
          <rect width="260" height="48" rx="4" className="illustration__token--target" />
          <text x="10" y="20" className="illustration__label illustration__label--small">prompt 末尾："peppermint →"</text>
          <text x="10" y="40" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)", fontWeight: 700 }}>
            模型续写：menthe poivrée ✓
          </text>
        </g>

        <text x="20" y="334" className="illustration__label illustration__label--small">
          这就是 in-context learning：
        </text>
        <text x="20" y="350" className="illustration__label illustration__label--small">
          模型把"示例 → 模式"装进 prompt 临时记忆
        </text>
        <text x="20" y="366" className="illustration__label illustration__label--small">
          后来 RAG / chain-of-thought 都从这里发展
        </text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2021 — CLIP 双塔 + N×N 对比矩阵
 * ===================================================================== */
function ClipDiagram() {
  const N = 6;
  // N×N 相似度矩阵：对角线高，其余低
  const sim = Array.from({ length: N }, (_, r) =>
    Array.from({ length: N }, (_, c) => (r === c ? 0.95 : 0.05 + Math.random() * 0.25)),
  );
  const captions = [
    "a dog playing",
    "a cat on rug",
    "snowy mountain",
    "red sports car",
    "plate of food",
    "a smiling kid",
  ];

  return (
    <svg viewBox="0 0 1100 580" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        CLIP · 一个 batch 同时处理 N 张图 + N 段文本，N²−N 个负样本撑起对比学习
      </text>

      {/* 上：N 张图 → ViT */}
      <text x="30" y="64" className="illustration__label">① 图像通路</text>
      {Array.from({ length: N }).map((_, i) => (
        <rect
          key={`img-${i}`}
          x={60 + i * 70}
          y="76"
          width="60"
          height="40"
          rx="6"
          className="illustration__pixel illustration__pixel--big"
          style={{ animationDelay: `${i * 60}ms` }}
        />
      ))}
      <text x="60" y="132" className="illustration__label illustration__label--small">N=batch 张图</text>

      <g transform="translate(490, 80)">
        <rect width="120" height="36" rx="6" className="illustration__block illustration__block--alt" />
        <text x="60" y="22" textAnchor="middle" className="illustration__block-label">ViT 编码器</text>
      </g>
      <line x1="480" y1="98" x2="487" y2="98" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      {/* 图像嵌入 I_1..I_N */}
      <g transform="translate(620, 80)">
        <line x1="0" y1="18" x2="14" y2="18" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        {Array.from({ length: N }).map((_, i) => (
          <g key={i} transform={`translate(20, ${i * 8})`}>
            <rect width="36" height="6" rx="1" className="illustration__featuremap" />
            <text x="42" y="6" className="illustration__label illustration__label--small">I_{i + 1}</text>
          </g>
        ))}
      </g>

      {/* 下：N 段文本 → Text Transformer */}
      <text x="30" y="170" className="illustration__label">② 文本通路</text>
      {captions.map((c, i) => (
        <g key={`txt-${i}`} transform={`translate(60, ${184 + i * 24})`}>
          <rect width="380" height="20" rx="3" className="illustration__block" />
          <text x="10" y="14" className="illustration__label illustration__label--small">"{c}"</text>
        </g>
      ))}

      <g transform="translate(490, 230)">
        <rect width="120" height="36" rx="6" className="illustration__block illustration__block--alt" />
        <text x="60" y="22" textAnchor="middle" className="illustration__block-label">Text Transformer</text>
      </g>
      <line x1="450" y1="248" x2="487" y2="248" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g transform="translate(620, 230)">
        <line x1="0" y1="18" x2="14" y2="18" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        {Array.from({ length: N }).map((_, i) => (
          <g key={i} transform={`translate(20, ${i * 8})`}>
            <rect width="36" height="6" rx="1" className="illustration__featuremap illustration__featuremap--ctx" />
            <text x="42" y="6" className="illustration__label illustration__label--small">T_{i + 1}</text>
          </g>
        ))}
      </g>

      {/* N×N 相似度矩阵 */}
      <text x="780" y="64" className="illustration__label illustration__label--strong">
        ③ N×N 余弦相似度矩阵
      </text>
      <g transform="translate(800, 80)">
        {/* 行列标签 */}
        {Array.from({ length: N }).map((_, i) => (
          <text key={`rl-${i}`} x="-6" y={i * 28 + 22} textAnchor="end" className="illustration__label illustration__label--small">I_{i + 1}</text>
        ))}
        {Array.from({ length: N }).map((_, i) => (
          <text key={`cl-${i}`} x={i * 28 + 14} y="-6" textAnchor="middle" className="illustration__label illustration__label--small">T_{i + 1}</text>
        ))}

        {/* 矩阵单元 */}
        {sim.map((row, r) =>
          row.map((v, c) => (
            <g key={`s-${r}-${c}`}>
              <rect
                x={c * 28}
                y={r * 28 + 8}
                width="26"
                height="26"
                rx="3"
                className={r === c ? "illustration__attn-cell" : "illustration__attn-cell illustration__attn-cell--raw"}
                style={{ opacity: r === c ? 0.95 : 0.12 + v * 0.25, animationDelay: `${(r * N + c) * 25}ms` }}
              />
              {r === c && (
                <text x={c * 28 + 13} y={r * 28 + 25} textAnchor="middle" className="illustration__label illustration__label--small" style={{ fill: "white", fontWeight: 700 }}>
                  +
                </text>
              )}
            </g>
          )),
        )}

        {/* 标注对角线为正样本 */}
        <text x={N * 28 + 14} y={N * 14} className="illustration__label illustration__label--small">
          ← 对角线 N 个
        </text>
        <text x={N * 28 + 14} y={N * 14 + 16} className="illustration__label illustration__label--small">
          正样本 (匹配的 I-T 对)
        </text>
        <text x={N * 28 + 14} y={N * 14 + 40} className="illustration__label illustration__label--small" style={{ fill: "var(--ink-muted)" }}>
          其余 N²−N 个为负样本
        </text>
      </g>

      {/* loss 公式 */}
      <g transform="translate(30, 400)">
        <rect width="1040" height="120" rx="10" className="illustration__block illustration__block--alt" />
        <text x="20" y="28" className="illustration__label illustration__label--strong">
          InfoNCE 对比损失（双向）
        </text>
        <text x="20" y="56" className="illustration__label">
          L = ½ · ( CE_row( sim · τ ) + CE_col( sim · τ ) )
        </text>
        <text x="20" y="82" className="illustration__label illustration__label--small">
          每张图 I_i 把对应文本 T_i 的概率最大化（行 softmax）；每段文本 T_i 把对应图 I_i 最大化（列 softmax）
        </text>
        <text x="20" y="102" className="illustration__label illustration__label--small">
          训练数据：400M 网上抓取的"图-文本"对；图像和文本被映射到同一表示空间 → 之后可以 0-shot 分类、检索、生成
        </text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2022 — RLHF 三阶段
 * ===================================================================== */
function RlhfDiagram() {
  return (
    <svg viewBox="0 0 1100 580" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        RLHF · 把"什么回答更好"的人类偏好编码进语言模型
      </text>

      {/* Stage 1: SFT */}
      <g transform="translate(20, 60)">
        <rect width="340" height="430" rx="14" className="illustration__group" />
        <g className="illustration__badge">
          <rect x="12" y="-12" width="56" height="22" rx="11" />
          <text x="40" y="3" textAnchor="middle">Stage 1</text>
        </g>
        <text x="80" y="6" className="illustration__label illustration__label--strong">SFT · 监督微调</text>
        <text x="18" y="36" className="illustration__label illustration__label--small">~13K 高质量人写演示</text>

        {/* 数据样本 */}
        <g transform="translate(20, 56)">
          <rect width="300" height="60" rx="6" className="illustration__block illustration__block--alt" />
          <text x="12" y="20" className="illustration__label illustration__label--small">Prompt: 解释一下 RLHF</text>
          <text x="12" y="40" className="illustration__label illustration__label--small">Demo (人写): RLHF 是一种用人类反馈…</text>
        </g>

        {/* 训练流程 */}
        <g transform="translate(80, 142)">
          <rect width="180" height="50" rx="6" className="illustration__proj illustration__proj--ffn" />
          <text x="90" y="22" textAnchor="middle" className="illustration__block-label">预训练 GPT-3.5</text>
          <text x="90" y="40" textAnchor="middle" className="illustration__label illustration__label--small">作为起点</text>
        </g>
        <line x1="170" y1="194" x2="170" y2="216" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(80, 216)">
          <rect width="180" height="50" rx="6" className="illustration__proj illustration__proj--act" />
          <text x="90" y="22" textAnchor="middle" className="illustration__block-label">监督微调</text>
          <text x="90" y="40" textAnchor="middle" className="illustration__label illustration__label--small">下一 token 损失</text>
        </g>
        <line x1="170" y1="268" x2="170" y2="290" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(60, 290)">
          <rect width="220" height="34" rx="6" className="illustration__block illustration__block--alt" />
          <text x="110" y="22" textAnchor="middle" className="illustration__block-label">SFT Model π_SFT</text>
        </g>

        <text x="18" y="356" className="illustration__label illustration__label--small">优点：学会"听指令"的基本格式</text>
        <text x="18" y="376" className="illustration__label illustration__label--small">局限：人写演示有上限，无法表达</text>
        <text x="18" y="392" className="illustration__label illustration__label--small">          "回答 A 比回答 B 好多少"</text>
        <text x="18" y="416" className="illustration__label illustration__label--small">→ 进入第二阶段</text>
      </g>

      {/* Stage 2: RM */}
      <g transform="translate(382, 60)">
        <rect width="340" height="430" rx="14" className="illustration__group" />
        <g className="illustration__badge">
          <rect x="12" y="-12" width="56" height="22" rx="11" />
          <text x="40" y="3" textAnchor="middle">Stage 2</text>
        </g>
        <text x="80" y="6" className="illustration__label illustration__label--strong">RM · 训练奖励模型</text>
        <text x="18" y="36" className="illustration__label illustration__label--small">~33K 人类排序对比</text>

        {/* 数据样本 */}
        <g transform="translate(20, 56)">
          <rect width="300" height="90" rx="6" className="illustration__block illustration__block--alt" />
          <text x="12" y="20" className="illustration__label illustration__label--small">Prompt: 同一问题，4 个回答</text>
          <text x="12" y="40" className="illustration__label illustration__label--small">人类排序：A &gt; C &gt; B &gt; D</text>
          <text x="12" y="60" className="illustration__label illustration__label--small">每个 pair (A, C) 等都是训练样本</text>
          <text x="12" y="78" className="illustration__label illustration__label--small">total C(4,2)=6 个 pair / prompt</text>
        </g>

        <line x1="190" y1="146" x2="190" y2="166" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        {/* RM 训练 */}
        <g transform="translate(80, 166)">
          <rect width="180" height="50" rx="6" className="illustration__proj illustration__proj--v" />
          <text x="90" y="22" textAnchor="middle" className="illustration__block-label">RM (从 SFT 复制)</text>
          <text x="90" y="40" textAnchor="middle" className="illustration__label illustration__label--small">输出标量 r(prompt,回答)</text>
        </g>

        <line x1="170" y1="216" x2="170" y2="236" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        {/* loss */}
        <g transform="translate(20, 236)">
          <rect width="300" height="76" rx="6" className="illustration__block" />
          <text x="12" y="20" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-alignment-ink)" }}>
            Bradley-Terry 损失
          </text>
          <text x="12" y="42" className="illustration__label illustration__label--small">
            L = − log σ( r(w) − r(l) )
          </text>
          <text x="12" y="62" className="illustration__label illustration__label--small">
            把胜者 w 推高、败者 l 推低
          </text>
        </g>

        <line x1="170" y1="316" x2="170" y2="336" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(60, 336)">
          <rect width="220" height="34" rx="6" className="illustration__block illustration__block--alt" />
          <text x="110" y="22" textAnchor="middle" className="illustration__block-label">Reward Model RM_φ</text>
        </g>

        <text x="18" y="396" className="illustration__label illustration__label--small">RM 学到"人类偏好"的代理</text>
        <text x="18" y="416" className="illustration__label illustration__label--small">→ 第三阶段当作奖励信号</text>
      </g>

      {/* Stage 3: PPO */}
      <g transform="translate(744, 60)">
        <rect width="340" height="430" rx="14" className="illustration__group" />
        <g className="illustration__badge">
          <rect x="12" y="-12" width="56" height="22" rx="11" />
          <text x="40" y="3" textAnchor="middle">Stage 3</text>
        </g>
        <text x="80" y="6" className="illustration__label illustration__label--strong">PPO · RL 优化策略</text>
        <text x="18" y="36" className="illustration__label illustration__label--small">policy = SFT 模型，用 RM 当 reward</text>

        {/* PPO 循环 */}
        <g transform="translate(20, 60)">
          <rect width="300" height="40" rx="6" className="illustration__block illustration__block--alt" />
          <text x="150" y="18" textAnchor="middle" className="illustration__block-label">prompt 从训练池采样</text>
          <text x="150" y="32" textAnchor="middle" className="illustration__label illustration__label--small">~31K 提示</text>
        </g>
        <line x1="170" y1="100" x2="170" y2="120" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(20, 120)">
          <rect width="300" height="40" rx="6" className="illustration__proj illustration__proj--ffn" />
          <text x="150" y="18" textAnchor="middle" className="illustration__block-label">policy π_θ 生成回答 y</text>
          <text x="150" y="32" textAnchor="middle" className="illustration__label illustration__label--small">从 SFT 初始化</text>
        </g>
        <line x1="170" y1="160" x2="170" y2="180" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(20, 180)">
          <rect width="140" height="40" rx="6" className="illustration__proj illustration__proj--v" />
          <text x="70" y="18" textAnchor="middle" className="illustration__block-label">RM 打分</text>
          <text x="70" y="32" textAnchor="middle" className="illustration__label illustration__label--small">r(x, y)</text>

          <g transform="translate(160, 0)">
            <rect width="140" height="40" rx="6" className="illustration__proj illustration__proj--act" />
            <text x="70" y="18" textAnchor="middle" className="illustration__block-label">KL 惩罚</text>
            <text x="70" y="32" textAnchor="middle" className="illustration__label illustration__label--small">vs π_SFT</text>
          </g>
        </g>

        <line x1="170" y1="220" x2="170" y2="240" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(20, 240)">
          <rect width="300" height="60" rx="6" className="illustration__block illustration__block--alt" />
          <text x="12" y="20" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-alignment-ink)" }}>
            Reward = r(x,y) − β · KL(π_θ || π_SFT)
          </text>
          <text x="12" y="42" className="illustration__label illustration__label--small">奖励：人类偏好 RM 评分</text>
          <text x="12" y="56" className="illustration__label illustration__label--small">惩罚：不要漂离 SFT 太远（防 reward hacking）</text>
        </g>

        <line x1="170" y1="300" x2="170" y2="320" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <g transform="translate(20, 320)">
          <rect width="300" height="40" rx="6" className="illustration__proj illustration__proj--ffn" />
          <text x="150" y="18" textAnchor="middle" className="illustration__block-label">PPO 更新 π_θ</text>
          <text x="150" y="32" textAnchor="middle" className="illustration__label illustration__label--small">截断梯度 + 多个 epoch</text>
        </g>

        {/* 循环箭头回到 prompt */}
        <path d="M 320 340 C 340 340, 340 80, 320 80" className="illustration__feedback" fill="none" markerEnd="url(#arrow-head)" />

        <text x="18" y="396" className="illustration__label illustration__label--small">最终 InstructGPT / ChatGPT</text>
        <text x="18" y="416" className="illustration__label illustration__label--small">回答质量超过 175B GPT-3 (1.3B 即可)</text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2024 — Mixture-of-Experts（MoE）稀疏激活
 * ===================================================================== */
function MoeDiagram() {
  const N_EXPERTS = 8;
  const TOP_K = 2;
  // Router 权重（top-2 高，其余低）
  const routeWeights = [0.62, 0.31, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01];
  const lit = new Set([0, 1]); // 被选中的 expert 索引

  return (
    <svg viewBox="0 0 1100 500" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        Mixture-of-Experts · 同样的容量，每步只激活 {TOP_K}/{N_EXPERTS} = 25% 参数
      </text>

      {/* Token 输入 */}
      <g transform="translate(40, 80)">
        <rect width="120" height="40" rx="6" className="illustration__block illustration__block--alt" />
        <text x="60" y="20" textAnchor="middle" className="illustration__block-label">token h</text>
        <text x="60" y="36" textAnchor="middle" className="illustration__label illustration__label--small">前层 hidden 状态</text>
      </g>

      <line x1="160" y1="100" x2="200" y2="100" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      {/* Router (gating network) */}
      <g transform="translate(200, 80)">
        <rect width="160" height="40" rx="6" className="illustration__proj illustration__proj--v" />
        <text x="80" y="20" textAnchor="middle" className="illustration__block-label">Router · Linear+Softmax</text>
        <text x="80" y="36" textAnchor="middle" className="illustration__label illustration__label--small">输出 N 维概率</text>
      </g>

      <line x1="360" y1="100" x2="400" y2="100" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      {/* Router 权重柱状（N 个 bar，top-2 用强色） */}
      <g transform="translate(400, 80)">
        <text x="0" y="-6" className="illustration__label illustration__label--small">Router 权重 g(h)</text>
        {routeWeights.map((w, i) => (
          <g key={i}>
            <rect
              x={i * 26}
              y={36 - w * 36}
              width="22"
              height={w * 36}
              rx="2"
              className={lit.has(i) ? "illustration__bar illustration__bar--real" : "illustration__bar illustration__bar--fake"}
              style={{ animationDelay: `${i * 60}ms`, opacity: lit.has(i) ? 1 : 0.4 }}
            />
            <text x={i * 26 + 11} y="50" textAnchor="middle" className="illustration__label illustration__label--small">
              E{i + 1}
            </text>
          </g>
        ))}
        <text x={N_EXPERTS * 26 + 14} y="22" className="illustration__label illustration__label--small">
          取 top-{TOP_K}
        </text>
        <text x={N_EXPERTS * 26 + 14} y="38" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)", fontWeight: 700 }}>
          → E1, E2 选中
        </text>
      </g>

      {/* N 个 Expert（FFN） */}
      <text x="40" y="180" className="illustration__label illustration__label--strong">
        N = {N_EXPERTS} 个 Expert · 每个就是普通 FFN（Linear → GELU → Linear）
      </text>

      {Array.from({ length: N_EXPERTS }).map((_, i) => {
        const isLit = lit.has(i);
        const x = 40 + i * 130;
        return (
          <g key={`exp-${i}`} transform={`translate(${x}, 200)`} className={isLit ? "" : "illustration__dim"}>
            <rect width="120" height="84" rx="8" className={isLit ? "illustration__proj illustration__proj--ffn" : "illustration__block"} />
            <text x="60" y="20" textAnchor="middle" className="illustration__block-label">Expert {i + 1}</text>
            <text x="60" y="36" textAnchor="middle" className="illustration__label illustration__label--small">FFN d→4d→d</text>
            {isLit && (
              <>
                <rect x="20" y="46" width="80" height="6" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
                <rect x="20" y="58" width="80" height="6" rx="2" className="illustration__featuremap" />
                <rect x="20" y="70" width="80" height="6" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
              </>
            )}
            {!isLit && (
              <text x="60" y="62" textAnchor="middle" className="illustration__label illustration__label--small" style={{ opacity: 0.5 }}>
                未激活
              </text>
            )}
          </g>
        );
      })}

      {/* Router → 被选中的 expert 的强连接，其他弱 */}
      {Array.from({ length: N_EXPERTS }).map((_, i) => {
        const isLit = lit.has(i);
        const x = 40 + i * 130 + 60;
        return (
          <line
            key={`r-${i}`}
            x1="500"
            y1="120"
            x2={x}
            y2="200"
            className={isLit ? "illustration__branch illustration__branch--q" : "illustration__attn-link"}
            style={{ opacity: isLit ? 0.95 : 0.18 }}
          />
        );
      })}

      {/* 输出 = top-k 的加权和 */}
      <text x="40" y="312" className="illustration__label illustration__label--strong">
        输出 = g_1 · Expert_1(h) + g_2 · Expert_2(h)  ——  仅 2 个 FFN 参与计算
      </text>

      {/* 输出向量 */}
      <g transform="translate(40, 332)">
        <rect width="120" height="40" rx="6" className="illustration__block illustration__block--alt" />
        <text x="60" y="20" textAnchor="middle" className="illustration__block-label">h_next</text>
        <text x="60" y="36" textAnchor="middle" className="illustration__label illustration__label--small">送入下一层</text>
      </g>

      {/* 选中 expert 输出汇入 */}
      <path d="M 100 284 C 100 320, 90 320, 100 332" className="illustration__arrow" fill="none" />
      <path d="M 230 284 C 230 308, 130 308, 100 332" className="illustration__arrow" fill="none" markerEnd="url(#arrow-head)" />

      {/* 收益总结 */}
      <g transform="translate(220, 332)">
        <rect width="850" height="120" rx="10" className="illustration__block" />
        <text x="20" y="26" className="illustration__label illustration__label--strong">
          稀疏 MoE 的工程意义
        </text>
        <text x="20" y="54" className="illustration__label">
          总参数：8 × FFN 大小（相当于 8 倍容量）·  激活参数：2 × FFN（推理成本只翻 2 倍）
        </text>
        <text x="20" y="76" className="illustration__label illustration__label--small">
          代表作：Mixtral 8×7B（46.7B 总参数，每次激活 12.9B）· DeepSeek-V2 / GLM-4 / Qwen-MoE
        </text>
        <text x="20" y="96" className="illustration__label illustration__label--small">
          工程挑战：load balancing（防止 router 总选同一专家）· 通信开销（跨 GPU 路由）· capacity factor 调优
        </text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2016 — AlphaGo: Policy + Value + MCTS + Self-Play RL
 * ===================================================================== */
function AlphaGoDiagram() {
  const stones: { x: number; y: number; color: "black" | "white" }[] = [
    { x: 2, y: 3, color: "black" },
    { x: 3, y: 3, color: "white" },
    { x: 4, y: 4, color: "black" },
    { x: 5, y: 4, color: "white" },
    { x: 3, y: 5, color: "black" },
    { x: 6, y: 5, color: "white" },
  ];

  return (
    <svg viewBox="0 0 1100 580" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        AlphaGo · 神经网络 + 树搜索 + 自我对弈 RL 三件套
      </text>

      <text x="30" y="62" className="illustration__label">① 输入：当前棋盘状态（19×19，含历史 8 步）</text>
      <g transform="translate(40, 80)">
        <rect width="160" height="160" rx="6" fill="#e6c98a" stroke="var(--ink-soft)" strokeWidth="1.2" />
        {Array.from({ length: 9 }).map((_, i) => (
          <line key={`h-${i}`} x1={10 + i * 17.5} y1="10" x2={10 + i * 17.5} y2="150" stroke="var(--ink-soft)" strokeWidth="0.5" />
        ))}
        {Array.from({ length: 9 }).map((_, i) => (
          <line key={`v-${i}`} x1="10" y1={10 + i * 17.5} x2="150" y2={10 + i * 17.5} stroke="var(--ink-soft)" strokeWidth="0.5" />
        ))}
        {stones.map((s, i) => (
          <circle key={i} cx={10 + s.x * 17.5} cy={10 + s.y * 17.5} r="7" fill={s.color === "black" ? "#1a1208" : "#fff"} stroke="#1a1208" strokeWidth="0.8" />
        ))}
        <text x="80" y="178" textAnchor="middle" className="illustration__label illustration__label--small">
          state s · 含历史 + 当前走方
        </text>
      </g>

      <path d="M 210 120 C 250 120, 250 130, 290 130" className="illustration__branch illustration__branch--q" fill="none" markerEnd="url(#arrow-head)" />
      <path d="M 210 160 C 250 160, 250 240, 290 240" className="illustration__branch illustration__branch--v" fill="none" markerEnd="url(#arrow-head)" />

      <text x="290" y="68" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-vision-ink)" }}>
        ② 策略网络 Policy p_σ
      </text>
      <g transform="translate(290, 100)">
        <rect width="180" height="60" rx="8" className="illustration__proj illustration__proj--q" />
        <text x="90" y="26" textAnchor="middle" className="illustration__block-label">13 层 CNN</text>
        <text x="90" y="46" textAnchor="middle" className="illustration__label illustration__label--small">19×19 → 361 维 softmax</text>
      </g>
      <line x1="470" y1="130" x2="492" y2="130" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g transform="translate(498, 80)">
        <text x="0" y="-2" className="illustration__label illustration__label--small">落子概率 P(a|s)</text>
        {[
          { a: "D4", v: 0.32 },
          { a: "Q16", v: 0.18 },
          { a: "K10", v: 0.12 },
          { a: "C3", v: 0.08 },
          { a: "其余 357", v: 0.30 },
        ].map((p, i) => (
          <g key={p.a}>
            <rect x="0" y={6 + i * 14} width={p.v * 140} height="10" rx="2" className="illustration__bar" style={{ animationDelay: `${i * 60}ms` }} />
            <text x={p.v * 140 + 6} y={14 + i * 14} className="illustration__label illustration__label--small">{p.a} {(p.v * 100).toFixed(0)}%</text>
          </g>
        ))}
      </g>

      <text x="290" y="200" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-alignment-ink)" }}>
        ③ 价值网络 Value v_θ
      </text>
      <g transform="translate(290, 210)">
        <rect width="180" height="60" rx="8" className="illustration__proj illustration__proj--v" />
        <text x="90" y="26" textAnchor="middle" className="illustration__block-label">13 层 CNN + tanh</text>
        <text x="90" y="46" textAnchor="middle" className="illustration__label illustration__label--small">19×19 → 1 维 v∈[-1,1]</text>
      </g>
      <line x1="470" y1="240" x2="492" y2="240" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g transform="translate(498, 220)">
        <text x="0" y="-2" className="illustration__label illustration__label--small">局面胜率 v(s)</text>
        <rect x="0" y="6" width="180" height="22" rx="4" className="illustration__block" />
        <rect x="0" y="6" width={0.74 * 180} height="22" rx="4" className="illustration__bar illustration__bar--real" style={{ animationDelay: "300ms" }} />
        <text x="170" y="22" textAnchor="end" className="illustration__label illustration__label--small" style={{ fill: "white", fontWeight: 700 }}>0.74</text>
        <text x="0" y="44" className="illustration__label illustration__label--small">黑胜概率 74%（接近 +1 = 必胜）</text>
      </g>

      <text x="30" y="290" className="illustration__label illustration__label--strong">
        ④ MCTS 搜索树 · Policy 缩小分支、Value 评估叶子
      </text>
      <g transform="translate(40, 308)">
        <circle cx="240" cy="20" r="14" className="illustration__neuron illustration__neuron--big" />
        <text x="240" y="25" textAnchor="middle" className="illustration__block-label illustration__block-label--small" style={{ fill: "white" }}>s</text>

        {[
          { x: 100, label: "a₁" },
          { x: 240, label: "a₂" },
          { x: 380, label: "a₃" },
        ].map((n, i) => (
          <g key={n.label}>
            <line x1="240" y1="34" x2={n.x} y2="76" className="illustration__arrow" />
            <circle cx={n.x} cy="90" r="12" className="illustration__neuron" />
            <text x={n.x} y="94" textAnchor="middle" className="illustration__block-label illustration__block-label--small" style={{ fill: "white" }}>{n.label}</text>
            <text x={n.x} y="116" textAnchor="middle" className="illustration__label illustration__label--small">
              P={[0.32, 0.18, 0.12][i].toFixed(2)}
            </text>
          </g>
        ))}

        {[
          { x: 60, parent: 100 },
          { x: 140, parent: 100 },
          { x: 200, parent: 240 },
          { x: 280, parent: 240 },
          { x: 340, parent: 380 },
          { x: 420, parent: 380 },
        ].map((leaf, i) => (
          <g key={i}>
            <line x1={leaf.parent} y1="104" x2={leaf.x} y2="146" className="illustration__arrow" />
            <rect x={leaf.x - 14} y="146" width="28" height="20" rx="3" className="illustration__block illustration__block--alt" />
            <text x={leaf.x} y="160" textAnchor="middle" className="illustration__label illustration__label--small">
              v={[0.62, 0.58, 0.74, 0.71, 0.55, 0.41][i]}
            </text>
          </g>
        ))}

        <g transform="translate(500, 20)">
          <rect width="450" height="156" rx="10" className="illustration__block illustration__block--alt" />
          <text x="20" y="26" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>
            UCB1 选择
          </text>
          <text x="20" y="54" className="illustration__label">
            a* = argmax [ Q(s,a) + c·P(a|s)·√N(s) / (1+N(s,a)) ]
          </text>
          <text x="20" y="80" className="illustration__label illustration__label--small">
            Q(s,a)：该动作的平均叶子 value（exploit）
          </text>
          <text x="20" y="98" className="illustration__label illustration__label--small">
            P(a|s)：策略网络给的先验（exploration prior）
          </text>
          <text x="20" y="116" className="illustration__label illustration__label--small">
            N：访问次数 —— 多访问的节点优先级下降
          </text>
          <text x="20" y="140" className="illustration__label illustration__label--small">
            每步 16,000 次 rollout，全部并行 + 缓存
          </text>
        </g>
      </g>

      <text x="30" y="510" className="illustration__label illustration__label--strong">
        ⑤ Self-Play RL · 用自己对自己生成的对局，训练更强的策略 / 价值网络
      </text>
      <g transform="translate(30, 528)">
        <rect width="1040" height="40" rx="10" className="illustration__block" />
        <text x="20" y="20" className="illustration__label illustration__label--small">
          初始策略（监督学习人类棋谱） → 自我对弈 30M 局 → 用胜负当 reward 反向更新 → 新策略 → 又自我对弈…
        </text>
        <text x="20" y="34" className="illustration__label illustration__label--small">
          最终 AlphaGo Zero 完全摆脱人类棋谱，仅靠 self-play 就能超越 AlphaGo 原始版
        </text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2019 — GPT-2 + T5 text-to-text 统一
 * ===================================================================== */
function Gpt2T5Diagram() {
  const tasks = [
    { prefix: "translate English to German:", input: "Hello, how are you?", output: "Hallo, wie geht's?", color: "q", taskLabel: "翻译 WMT" },
    { prefix: "summarize:", input: "The cat that sat on the mat was…", output: "A cat sat on a mat.", color: "k", taskLabel: "摘要 CNN/DM" },
    { prefix: "cola sentence:", input: "The boy goed home.", output: "unacceptable", color: "v", taskLabel: "语法判断 CoLA" },
    { prefix: "stsb s1: A man speaks. s2:", input: "Someone is talking.", output: "4.2", color: "o", taskLabel: "语义相似度 STS-B" },
  ];

  return (
    <svg viewBox="0 0 1100 620" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        T5 · 把 NLP 所有任务都格式化成「text → text」的统一接口
      </text>

      <text x="30" y="58" className="illustration__label">
        ① 多任务统一为同一字符串接口
      </text>

      {tasks.map((t, i) => (
        <g key={i} transform={`translate(40, ${78 + i * 84})`}>
          <text x="0" y="-4" className="illustration__label illustration__label--small">
            {t.taskLabel}
          </text>
          <rect width="290" height="28" rx="4" className={`illustration__proj illustration__proj--${t.color}`} />
          <text x="14" y="19" className="illustration__block-label illustration__block-label--small">{t.prefix}</text>

          <rect x="300" y="0" width="350" height="28" rx="4" className="illustration__block illustration__block--alt" />
          <text x="314" y="19" className="illustration__label illustration__label--small">{t.input}</text>

          <line x1="660" y1="14" x2="698" y2="14" className="illustration__arrow" markerEnd="url(#arrow-head)" />

          <rect x="702" y="0" width="320" height="28" rx="4" className="illustration__token--target" />
          <text x="716" y="19" className="illustration__label illustration__label--small" style={{ fontWeight: 700, fill: "var(--rose-900)" }}>{t.output}</text>
        </g>
      ))}

      <text x="30" y="430" className="illustration__label illustration__label--strong">
        ② 同一个 11B 参数的 encoder-decoder Transformer 处理所有任务
      </text>
      <g transform="translate(40, 448)">
        <rect width="200" height="60" rx="8" className="illustration__proj illustration__proj--ffn" />
        <text x="100" y="28" textAnchor="middle" className="illustration__block-label">Encoder · 24 层</text>
        <text x="100" y="46" textAnchor="middle" className="illustration__label illustration__label--small">读"task: 输入"</text>

        <line x1="200" y1="30" x2="240" y2="30" className="illustration__arrow" markerEnd="url(#arrow-head)" />

        <rect x="240" y="0" width="200" height="60" rx="8" className="illustration__proj illustration__proj--o" />
        <text x="340" y="28" textAnchor="middle" className="illustration__block-label">Decoder · 24 层</text>
        <text x="340" y="46" textAnchor="middle" className="illustration__label illustration__label--small">自回归输出答案</text>

        <text x="460" y="24" className="illustration__label illustration__label--small">→ 11B 参数</text>
        <text x="460" y="42" className="illustration__label illustration__label--small">→ 750GB C4 语料预训练</text>
      </g>

      <text x="640" y="430" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-scale-ink)" }}>
        ③ 同时 GPT-2 用 1.5B 参数证明 zero-shot
      </text>
      <g transform="translate(640, 448)">
        <rect width="420" height="60" rx="8" className="illustration__block" />
        <text x="14" y="22" className="illustration__label illustration__label--small">prompt：「Translate to French: The cat is on」</text>
        <text x="14" y="44" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)", fontWeight: 700 }}>
          → 续写：「the mat. Le chat est sur le tapis.」
        </text>
      </g>

      <g transform="translate(40, 530)">
        <rect width="1020" height="76" rx="10" className="illustration__block illustration__block--alt" />
        <text x="20" y="26" className="illustration__label">
          意义：「任务 = 一个字符串」消除了下游接口分裂；同年 GPT-2 又证明大模型不微调也能 zero-shot
        </text>
        <text x="20" y="50" className="illustration__label illustration__label--small">
          为 2020 GPT-3 in-context few-shot 铺平道路 —— 接口先统一，规模再起飞
        </text>
        <text x="20" y="68" className="illustration__label illustration__label--small">
          T5 同时探索了"训练数据 vs 模型大小 vs 训练步数"三轴 → Chinchilla scaling 的前传
        </text>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2023 — GPT-4 (闭源 多模态) vs LLaMA (开源生态)
 * ===================================================================== */
function Gpt4LlamaDiagram() {
  return (
    <svg viewBox="0 0 1100 620" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        2023 双轨同年并行：闭源能力极限 vs 开源生态爆发
      </text>

      {/* === 左半：GPT-4 === */}
      <g transform="translate(20, 58)">
        <rect width="500" height="540" rx="14" className="illustration__group" />
        <text x="20" y="28" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>
          GPT-4 · 闭源能力跃迁
        </text>
        <text x="20" y="48" className="illustration__label illustration__label--small">
          参数 / 训练数据 / 训练方法均未公开
        </text>

        <text x="20" y="84" className="illustration__label">① 多模态输入（文本 + 图像）</text>
        <g transform="translate(20, 96)">
          <rect width="460" height="56" rx="6" className="illustration__block illustration__block--alt" />
          <text x="14" y="22" className="illustration__label illustration__label--small">文本："这张图里有什么不寻常？"</text>
          <text x="14" y="42" className="illustration__label illustration__label--small">图像：</text>
          <rect x="84" y="28" width="60" height="22" rx="3" className="illustration__pixel illustration__pixel--big" />
        </g>

        <g transform="translate(20, 168)">
          <rect width="460" height="80" rx="10" className="illustration__proj illustration__proj--o" />
          <text x="230" y="34" textAnchor="middle" className="illustration__block-label">GPT-4（黑盒）</text>
          <text x="230" y="56" textAnchor="middle" className="illustration__label illustration__label--small">
            推测：MoE 架构 ~1.8T 总参 · vision encoder · 32K context
          </text>
          <text x="230" y="72" textAnchor="middle" className="illustration__label illustration__label--small" style={{ opacity: 0.75 }}>
            （以上未官方确认）
          </text>
        </g>

        <text x="20" y="276" className="illustration__label">② 输出：复杂多步推理</text>
        <g transform="translate(20, 288)">
          <rect width="460" height="78" rx="6" className="illustration__token--target" />
          <text x="14" y="22" className="illustration__label illustration__label--small">
            "图中是一个 VGA 接口被插进 iPhone 充电口，
          </text>
          <text x="14" y="40" className="illustration__label illustration__label--small">
            两种接口物理不兼容，所以这个场景不合理…"
          </text>
          <text x="14" y="62" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
            → 不仅识别物体，还推理出场景荒谬性
          </text>
        </g>

        <text x="20" y="392" className="illustration__label">③ 专家级 benchmark 表现</text>
        {[
          { task: "美国律师资格考试", score: 90, prev: 10 },
          { task: "GRE 数学", score: 80, prev: 25 },
          { task: "MMLU 多任务", score: 86, prev: 70 },
        ].map((b, i) => (
          <g key={i} transform={`translate(20, ${408 + i * 36})`}>
            <text x="0" y="16" className="illustration__label illustration__label--small">{b.task}</text>
            <rect x="160" y="6" width="200" height="14" rx="2" className="illustration__block" />
            <rect x="160" y="6" width={b.prev * 2} height="14" rx="2" className="illustration__bar illustration__bar--fake" style={{ opacity: 0.5 }} />
            <rect x="160" y="6" width={b.score * 2} height="14" rx="2" className="illustration__bar illustration__bar--real" style={{ animationDelay: `${i * 100}ms` }} />
            <text x="370" y="18" className="illustration__label illustration__label--small">GPT-3.5 {b.prev}% → GPT-4 {b.score}%</text>
          </g>
        ))}
      </g>

      {/* === 右半：LLaMA === */}
      <g transform="translate(540, 58)">
        <rect width="540" height="540" rx="14" className="illustration__group" />
        <text x="20" y="28" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-alignment-ink)" }}>
          LLaMA · 开源权重 + 社区生态
        </text>
        <text x="20" y="48" className="illustration__label illustration__label--small">
          7B / 13B / 33B / 65B 四档，论文 + 权重全部公开
        </text>

        <g transform="translate(180, 76)">
          <rect width="180" height="60" rx="10" className="illustration__proj illustration__proj--v" />
          <text x="90" y="28" textAnchor="middle" className="illustration__block-label">LLaMA 基座</text>
          <text x="90" y="46" textAnchor="middle" className="illustration__label illustration__label--small">在 1.4T tokens 上预训练</text>
        </g>

        {[
          { col: 0, row: 0, label: "Alpaca", desc: "Stanford 用 GPT-3.5 生成 52K 指令", color: "q" },
          { col: 1, row: 0, label: "Vicuna", desc: "ShareGPT 对话微调，质量接近 GPT-3.5", color: "v" },
          { col: 2, row: 0, label: "Code LLaMA", desc: "代码继续训练，对标 Codex", color: "k" },
          { col: 0, row: 1, label: "Llama 2 Chat", desc: "Meta 官方 RLHF 微调，可商用", color: "ffn" },
          { col: 1, row: 1, label: "中文社区", desc: "Chinese-LLaMA / Baichuan / Qwen", color: "act" },
          { col: 2, row: 1, label: "量化版本", desc: "GGML/GGUF · MacBook 上跑 7B", color: "o" },
        ].map((d, i) => (
          <g key={d.label} transform={`translate(${30 + d.col * 170}, ${180 + d.row * 100})`}>
            <line
              x1="80"
              y1="0"
              x2={250 - 30 - d.col * 170 + 90}
              y2={-180 + 76 + 60 - d.row * 100}
              className="illustration__residual"
            />
            <rect width="160" height="80" rx="8" className={`illustration__proj illustration__proj--${d.color}`} />
            <text x="80" y="22" textAnchor="middle" className="illustration__block-label">{d.label}</text>
            <text x="80" y="42" textAnchor="middle" className="illustration__label illustration__label--small">{d.desc}</text>
          </g>
        ))}

        <g transform="translate(20, 410)">
          <rect width="500" height="118" rx="10" className="illustration__block illustration__block--alt" />
          <text x="14" y="26" className="illustration__label illustration__label--strong">
            社区影响：Hugging Face 月下载从 100k → 千万级
          </text>
          <text x="14" y="50" className="illustration__label illustration__label--small">
            • 学术：首次能复现"大模型 + RLHF"实验
          </text>
          <text x="14" y="68" className="illustration__label illustration__label--small">
            • 行业：垂直公司开始本地部署、规避 API 风险
          </text>
          <text x="14" y="86" className="illustration__label illustration__label--small">
            • QLoRA 让 7B 微调在单卡 24GB 即可
          </text>
          <text x="14" y="104" className="illustration__label illustration__label--small">
            • 评测：Open LLM Leaderboard 成为社区共识
          </text>
        </g>
      </g>
    </svg>
  );
}

/* =====================================================================
 * 2025 — DeepSeek R1: test-time compute + 可验证奖励 RL
 * ===================================================================== */
function DeepSeekR1Diagram() {
  return (
    <svg viewBox="0 0 1100 660" role="img" className="illustration__svg illustration__svg--tall">
      <ArrowDefs />

      <text x="30" y="30" className="illustration__label illustration__label--strong">
        DeepSeek R1 · 推理时长链思考 + 可验证奖励替代 RLHF
      </text>

      <text x="30" y="62" className="illustration__label illustration__label--strong">
        ① 推理时（test-time）：让模型"想更久" 而不是"训更大"
      </text>

      <g transform="translate(40, 80)">
        <rect width="220" height="50" rx="6" className="illustration__block illustration__block--alt" />
        <text x="14" y="22" className="illustration__label illustration__label--small">提示：</text>
        <text x="14" y="38" className="illustration__label illustration__label--small">"24 = ? · ? · 4，求 ? · ?"</text>
      </g>

      <line x1="260" y1="105" x2="290" y2="105" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g transform="translate(290, 80)">
        <rect width="540" height="120" rx="10" className="illustration__group illustration__group--inner" />
        <text x="14" y="22" className="illustration__label illustration__label--small">
          长链 chain-of-thought（每步生成几十到几千 token）：
        </text>
        {[
          "<think> 24 = a·b·4，所以 a·b = 6 …",
          "试 a=2, b=3：2·3·4 = 24 ✓",
          "试 a=1, b=6：1·6·4 = 24 ✓",
          "答案 = (2,3) 或 (1,6) </think>",
        ].map((step, i) => (
          <text key={i} x="20" y={48 + i * 18} className="illustration__label illustration__label--small">
            {step}
          </text>
        ))}
      </g>

      <line x1="840" y1="140" x2="870" y2="140" className="illustration__arrow" markerEnd="url(#arrow-head)" />

      <g transform="translate(870, 110)">
        <rect width="200" height="60" rx="6" className="illustration__token--target" />
        <text x="14" y="24" className="illustration__label illustration__label--small">最终回答：</text>
        <text x="14" y="44" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)", fontWeight: 700 }}>
          (2,3) 或 (1,6)
        </text>
      </g>

      <g transform="translate(40, 220)">
        <text x="0" y="0" className="illustration__label illustration__label--small">
          GPT-3 时代算力主要花在训练；R1 / o1 时代推理期算力同样可观（一次回答 ≈ 几千 token 思考）
        </text>
        <g transform="translate(0, 16)">
          <text x="0" y="14" className="illustration__label illustration__label--small">GPT-3 (2020)</text>
          <rect x="120" y="6" width="240" height="14" rx="2" className="illustration__bar" />
          <rect x="360" y="6" width="20" height="14" rx="2" className="illustration__bar illustration__bar--fake" />
          <text x="388" y="18" className="illustration__label illustration__label--small">训练 95% / 推理 5%</text>
        </g>
        <g transform="translate(0, 36)">
          <text x="0" y="14" className="illustration__label illustration__label--small">R1 / o1 (2025)</text>
          <rect x="120" y="6" width="140" height="14" rx="2" className="illustration__bar" />
          <rect x="260" y="6" width="120" height="14" rx="2" className="illustration__bar illustration__bar--real" />
          <text x="388" y="18" className="illustration__label illustration__label--small">训练 55% / 推理 45%</text>
        </g>
      </g>

      <text x="30" y="320" className="illustration__label illustration__label--strong">
        ② 训练时：用"可验证奖励"替代 RLHF 里需要训练的 reward model
      </text>

      <g transform="translate(40, 342)">
        <rect width="320" height="260" rx="14" className="illustration__group" />
        <text x="14" y="22" className="illustration__label illustration__label--strong">RLHF（2022）—— 难度高</text>
        <g transform="translate(14, 36)">
          <rect width="290" height="36" rx="6" className="illustration__block illustration__block--alt" />
          <text x="145" y="22" textAnchor="middle" className="illustration__block-label illustration__block-label--small">人类排序 → 训 RM_φ</text>
        </g>
        <line x1="160" y1="76" x2="160" y2="96" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(14, 96)">
          <rect width="290" height="36" rx="6" className="illustration__proj illustration__proj--v" />
          <text x="145" y="22" textAnchor="middle" className="illustration__block-label illustration__block-label--small">RM 当奖励（嘈杂可钻空）</text>
        </g>
        <line x1="160" y1="132" x2="160" y2="152" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(14, 152)">
          <rect width="290" height="36" rx="6" className="illustration__proj illustration__proj--ffn" />
          <text x="145" y="22" textAnchor="middle" className="illustration__block-label illustration__block-label--small">PPO 更新（要 critic、复杂）</text>
        </g>
        <text x="14" y="222" className="illustration__label illustration__label--small">优点：通用</text>
        <text x="14" y="240" className="illustration__label illustration__label--small">缺点：RM 不准 → reward hacking</text>
      </g>

      <g transform="translate(370, 470)">
        <line x1="0" y1="0" x2="30" y2="0" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <text x="15" y="-8" textAnchor="middle" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>
          R1
        </text>
      </g>

      <g transform="translate(414, 342)">
        <rect width="320" height="260" rx="14" className="illustration__group" />
        <text x="14" y="22" className="illustration__label illustration__label--strong" style={{ fill: "var(--rose-700)" }}>R1 风格 RL —— 简单暴力</text>
        <g transform="translate(14, 36)">
          <rect width="290" height="36" rx="6" className="illustration__block illustration__block--alt" />
          <text x="145" y="22" textAnchor="middle" className="illustration__block-label illustration__block-label--small">数学 + 代码（有标准答案）</text>
        </g>
        <line x1="160" y1="76" x2="160" y2="96" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(14, 96)">
          <rect width="290" height="36" rx="6" className="illustration__proj illustration__proj--q" />
          <text x="145" y="22" textAnchor="middle" className="illustration__block-label illustration__block-label--small">生成 → 自动 verifier 打分</text>
        </g>
        <line x1="160" y1="132" x2="160" y2="152" className="illustration__arrow" markerEnd="url(#arrow-head)" />
        <g transform="translate(14, 152)">
          <rect width="290" height="36" rx="6" className="illustration__proj illustration__proj--ffn" />
          <text x="145" y="22" textAnchor="middle" className="illustration__block-label illustration__block-label--small">GRPO（无 critic 的 RL）</text>
        </g>
        <text x="14" y="222" className="illustration__label illustration__label--small" style={{ fill: "var(--rose-700)" }}>
          奖励 = 答案对错 / 单测通过率
        </text>
        <text x="14" y="240" className="illustration__label illustration__label--small">无 RM、无 critic、奖励绝对可靠</text>
      </g>

      <g transform="translate(750, 342)">
        <rect width="320" height="260" rx="14" className="illustration__group" />
        <text x="14" y="22" className="illustration__label illustration__label--strong">
          ③ 结果与意义
        </text>
        <text x="14" y="48" className="illustration__label illustration__label--small">
          • DeepSeek R1（开源）数学/代码追平 o1
        </text>
        <text x="14" y="66" className="illustration__label illustration__label--small">
          • 训练数据：纯合成（模型生成 + verifier 标）
        </text>
        <text x="14" y="84" className="illustration__label illustration__label--small">
          • 训练成本 &lt;6M USD（vs GPT-4 ~100M）
        </text>
        <text x="14" y="102" className="illustration__label illustration__label--small">
          • "推理 = 长链思考"成为新维度
        </text>
        <text x="14" y="140" className="illustration__label illustration__label--strong" style={{ fill: "var(--phase-alignment-ink)" }}>
          叙事拐点
        </text>
        <text x="14" y="162" className="illustration__label illustration__label--small">
          闭源不再是性能必然垄断者；
        </text>
        <text x="14" y="180" className="illustration__label illustration__label--small">
          "可验证任务上 RL" 给了开源
        </text>
        <text x="14" y="198" className="illustration__label illustration__label--small">
          一条不需要海量人类标注的捷径。
        </text>
        <text x="14" y="222" className="illustration__label illustration__label--small">
          下一步：通用任务上"可验证奖励"
        </text>
        <text x="14" y="240" className="illustration__label illustration__label--small">
          如何构造，仍是开放问题。
        </text>
      </g>
    </svg>
  );
}
