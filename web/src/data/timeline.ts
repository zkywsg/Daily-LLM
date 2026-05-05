export type TimelineWork = {
  name: string;
  contribution: string;
  modulePath?: string;
};

export type RelatedModule = {
  label: string;
  path: string;
};

export type TimelineNode = {
  year: string;
  title: string;
  shortTitle: string;
  phase: string;
  previousLimit: string;
  whatHappened: string;
  solved: string;
  newProblems: string;
  keyWorks: TimelineWork[];
  relatedModules: RelatedModule[];
};

export const timelineNodes: TimelineNode[] = [
  {
    year: "1948",
    title: "Shannon 信息论：信息的数学基础",
    shortTitle: "Shannon",
    phase: "理论起点",
    previousLimit: "通信工程长期依赖经验设计编码方案，却缺少量化信息与噪声极限的统一数学语言。",
    whatHappened: "Shannon 提出信息熵、互信息和信道容量，让信息可以用比特度量，也让压缩与传输有了理论边界。",
    solved: "信息从工程直觉变成可计算对象，为交叉熵、互信息和表示学习提供了底层语言。",
    newProblems: "理论刻意抽掉语义，只回答信息量，不回答信息是否被理解。",
    keyWorks: [
      {
        name: "信息熵",
        contribution: "用不确定性度量信息量，成为机器学习损失函数的重要基础。",
      },
      {
        name: "信道容量",
        contribution: "定义噪声信道的最大可靠传输速率。",
      },
    ],
    relatedModules: [
      {
        label: "概率与信息论",
        path: "../00-Prerequisites/probability-information-theory/",
      },
    ],
  },
  {
    year: "2012",
    title: "AlexNet：一声炮响，旧世界终结",
    shortTitle: "AlexNet",
    phase: "视觉线",
    previousLimit: "计算机视觉依赖 SIFT、HOG 等手工特征，ImageNet 错误率多年停在 25% 到 26%。",
    whatHappened: "AlexNet 用深度卷积网络、ReLU、GPU 并行、Dropout 和数据增强，把 Top-5 错误率打到 15.3%。",
    solved: "它证明特征可以从数据中学习，深度学习成为视觉识别的主线。",
    newProblems: "大数据、GPU 算力和模型可解释性成为新的门槛。",
    keyWorks: [
      {
        name: "Dropout",
        contribution: "随机失活缓解过拟合，成为深度网络标准正则化手段。",
        modulePath: "../00-Prerequisites/regularization/",
      },
      {
        name: "ReLU",
        contribution: "改善梯度流动，让深层网络训练明显加速。",
        modulePath: "../00-Prerequisites/activation-functions/",
      },
      {
        name: "GPU 训练",
        contribution: "确立 CUDA 加速深度网络训练的基础设施范式。",
      },
    ],
    relatedModules: [
      {
        label: "CNN 架构",
        path: "../01-Visual-Intelligence/cnn-architectures/",
      },
      {
        label: "训练基础",
        path: "../01-Visual-Intelligence/training/",
      },
    ],
  },
  {
    year: "2013",
    title: "Word2Vec：词也能有坐标",
    shortTitle: "Word2Vec",
    phase: "语言线",
    previousLimit: "One-Hot 维度高且没有语义距离，模型不知道猫和狗比猫和飞机更接近。",
    whatHappened: "Word2Vec 用上下文预测训练稠密词向量，让词之间的语义关系进入向量空间。",
    solved: "NLP 获得可迁移的语义表示，下游任务不再完全依赖稀疏人工特征。",
    newProblems: "每个词仍只有一个静态向量，无法区分苹果公司和苹果水果。",
    keyWorks: [
      {
        name: "Skip-gram",
        contribution: "用中心词预测上下文，适合学习稀有词表示。",
      },
      {
        name: "VAE",
        contribution: "用变分推断学习连续潜变量，推动生成模型发展。",
      },
    ],
    relatedModules: [
      {
        label: "嵌入表示",
        path: "../00-Prerequisites/embeddings/",
      },
      {
        label: "语言线总览",
        path: "../02-Language-Transformers/",
      },
    ],
  },
  {
    year: "2014",
    title: "GAN、Seq2Seq、Attention、Adam：一年四响",
    shortTitle: "GAN / Attention",
    phase: "生成与序列",
    previousLimit: "模型会分类却不擅长生成，机器翻译依赖规则对齐，优化学习率需要大量手工调节。",
    whatHappened: "GAN 打开生成式建模，Seq2Seq 改写翻译范式，Attention 缓解固定向量瓶颈，Adam 简化优化。",
    solved: "生成、序列转换、动态对齐和自适应优化四个基础件同时到位。",
    newProblems: "GAN 训练不稳定，Seq2Seq 仍串行，Attention 还没有成为统一架构。",
    keyWorks: [
      {
        name: "Attention",
        contribution: "让解码器按需回看输入位置，突破固定上下文瓶颈。",
        modulePath: "../02-Language-Transformers/attention-mechanisms/",
      },
      {
        name: "Adam",
        contribution: "结合动量和自适应学习率，成为深度学习默认优化器。",
      },
    ],
    relatedModules: [
      {
        label: "注意力机制",
        path: "../02-Language-Transformers/attention-mechanisms/",
      },
      {
        label: "编码器-解码器",
        path: "../00-Prerequisites/encoder-decoder/",
      },
    ],
  },
  {
    year: "2015",
    title: "ResNet 与 Batch Norm：深度的解放",
    shortTitle: "ResNet / BN",
    phase: "视觉线",
    previousLimit: "网络变深后训练误差反而升高，深度超过二十多层就容易退化。",
    whatHappened: "ResNet 用跳跃连接学习残差，Batch Norm 稳定层输入分布，让极深网络可以训练。",
    solved: "网络深度被解放，152 层 ResNet 在 ImageNet 上达到低于人类水平的错误率。",
    newProblems: "更深更宽的模型加剧算力需求，也让归一化和残差设计成为工程必修课。",
    keyWorks: [
      {
        name: "Residual Connection",
        contribution: "给梯度和信息提供直通路径，缓解退化问题。",
        modulePath: "../00-Prerequisites/residual-connections/",
      },
      {
        name: "Batch Normalization",
        contribution: "稳定中间激活分布，提高训练速度和稳定性。",
        modulePath: "../00-Prerequisites/normalization/",
      },
    ],
    relatedModules: [
      {
        label: "残差连接",
        path: "../00-Prerequisites/residual-connections/",
      },
      {
        label: "CNN 架构",
        path: "../01-Visual-Intelligence/cnn-architectures/",
      },
    ],
  },
  {
    year: "2016",
    title: "AlphaGo：强化学习登台",
    shortTitle: "AlphaGo",
    phase: "决策智能",
    previousLimit: "围棋搜索空间极大，传统搜索和人工特征难以覆盖复杂局面。",
    whatHappened: "AlphaGo 结合策略网络、价值网络、蒙特卡洛树搜索和强化学习，击败李世石。",
    solved: "它证明深度学习可以和搜索、规划结合，处理复杂决策问题。",
    newProblems: "系统高度专用，训练成本高，离通用智能仍有明显距离。",
    keyWorks: [
      {
        name: "Policy Network",
        contribution: "缩小搜索动作空间，让搜索更有效率。",
      },
      {
        name: "Value Network",
        contribution: "评估局面胜率，减少深度展开成本。",
      },
    ],
    relatedModules: [
      {
        label: "系统生产总览",
        path: "../05-Systems-Production/",
      },
    ],
  },
  {
    year: "2017",
    title: "Transformer：把 RNN 扔掉",
    shortTitle: "Transformer",
    phase: "语言线",
    previousLimit: "LSTM 天生串行，长句训练慢，远距离依赖仍然难以稳定建模。",
    whatHappened: "Transformer 完全用自注意力和前馈网络替代循环结构，让序列建模可以大规模并行。",
    solved: "它统一了上下文建模方式，成为后续 BERT、GPT 和大语言模型的基础架构。",
    newProblems: "自注意力带来 O(n²) 复杂度，长上下文成本开始成为核心瓶颈。",
    keyWorks: [
      {
        name: "Self-Attention",
        contribution: "每个 token 直接和其他 token 建立依赖。",
        modulePath: "../02-Language-Transformers/attention-mechanisms/",
      },
      {
        name: "Multi-Head Attention",
        contribution: "在多个子空间并行建模不同关系。",
        modulePath: "../02-Language-Transformers/transformer-architecture/",
      },
    ],
    relatedModules: [
      {
        label: "Transformer 架构",
        path: "../02-Language-Transformers/transformer-architecture/",
      },
      {
        label: "注意力机制",
        path: "../02-Language-Transformers/attention-mechanisms/",
      },
    ],
  },
  {
    year: "2018",
    title: "BERT 与 GPT-1：预训练时代",
    shortTitle: "BERT / GPT-1",
    phase: "预训练",
    previousLimit: "静态词向量无法处理一词多义，下游任务通常需要从头训练或重做大量特征工程。",
    whatHappened: "BERT 用双向 Masked LM 学上下文表示，GPT-1 用自回归预训练展示生成式迁移。",
    solved: "预训练加微调成为 NLP 标准范式，词义开始随上下文动态变化。",
    newProblems: "预训练成本上升，模型路线分化为理解式编码器和生成式解码器。",
    keyWorks: [
      {
        name: "BERT",
        contribution: "双向上下文预训练刷新多项 NLP 理解任务。",
        modulePath: "../02-Language-Transformers/pretrained-models/",
      },
      {
        name: "GPT-1",
        contribution: "证明自回归语言模型可以迁移到下游任务。",
      },
    ],
    relatedModules: [
      {
        label: "预训练模型",
        path: "../02-Language-Transformers/pretrained-models/",
      },
    ],
  },
  {
    year: "2019",
    title: "GPT-2 与 T5：规模的野心",
    shortTitle: "GPT-2 / T5",
    phase: "预训练",
    previousLimit: "NLP 任务接口分裂，研究者还不确定单纯扩大语言模型是否会持续带来能力提升。",
    whatHappened: "GPT-2 展示大规模自回归生成能力，T5 把多种 NLP 任务统一为文本到文本格式。",
    solved: "规模化训练和统一任务接口成为新的主线。",
    newProblems: "参数、数据和算力快速膨胀，评估与安全问题开始变得突出。",
    keyWorks: [
      {
        name: "GPT-2",
        contribution: "1.5B 参数模型展示强生成能力和零样本潜力。",
      },
      {
        name: "T5",
        contribution: "把分类、翻译、摘要等任务统一成 text-to-text。",
      },
    ],
    relatedModules: [
      {
        label: "预训练模型",
        path: "../02-Language-Transformers/pretrained-models/",
      },
    ],
  },
  {
    year: "2020",
    title: "GPT-3 与 Scaling Laws：大力出奇迹",
    shortTitle: "GPT-3",
    phase: "规模化",
    previousLimit: "主流观点认为大模型仍需要针对每个任务微调，Few-shot 能力并不可靠。",
    whatHappened: "GPT-3 扩展到 175B 参数，Scaling Laws 系统描述数据、参数和计算之间的关系。",
    solved: "Prompt 和 Few-shot 让模型在不更新参数的情况下完成多种任务。",
    newProblems: "训练成本、推理成本和不可控输出成为规模路线的代价。",
    keyWorks: [
      {
        name: "Scaling Laws",
        contribution: "用经验规律指导模型、数据和算力配比。",
        modulePath: "../03-Scale-Multimodal/pre-training/",
      },
      {
        name: "GPT-3",
        contribution: "展示大模型涌现式 Few-shot 能力。",
      },
    ],
    relatedModules: [
      {
        label: "预训练",
        path: "../03-Scale-Multimodal/pre-training/",
      },
      {
        label: "提示工程",
        path: "../03-Scale-Multimodal/prompt-engineering/",
      },
    ],
  },
  {
    year: "2021",
    title: "CLIP、Codex、LoRA：多模态与效率",
    shortTitle: "CLIP / LoRA",
    phase: "多模态",
    previousLimit: "视觉和语言系统割裂，大模型微调需要复制全部参数，成本只有巨头承担得起。",
    whatHappened: "CLIP 用图文对比学习对齐视觉语言，Codex 把语言模型迁移到代码，LoRA 降低微调成本。",
    solved: "多模态对齐、代码生成和参数高效微调同时进入实用阶段。",
    newProblems: "数据版权、模型偏见和微调后的部署治理成为新问题。",
    keyWorks: [
      {
        name: "CLIP",
        contribution: "用自然语言监督视觉模型，连接图像和文本空间。",
        modulePath: "../03-Scale-Multimodal/multimodal/",
      },
      {
        name: "LoRA",
        contribution: "通过低秩适配器降低大模型微调成本。",
        modulePath: "../04-Alignment-OpenSource/peft/",
      },
    ],
    relatedModules: [
      {
        label: "多模态",
        path: "../03-Scale-Multimodal/multimodal/",
      },
      {
        label: "PEFT",
        path: "../04-Alignment-OpenSource/peft/",
      },
    ],
  },
  {
    year: "2022",
    title: "ChatGPT 与 RLHF：AI 走进大众",
    shortTitle: "ChatGPT",
    phase: "对齐",
    previousLimit: "GPT-3 更像文本补全工具，缺少可靠的指令遵循和安全对齐机制。",
    whatHappened: "ChatGPT 通过指令微调和 RLHF 把语言模型包装成可对话、可遵循指令的产品体验。",
    solved: "大模型从研究系统进入大众产品，交互范式从 prompt 变成自然对话。",
    newProblems: "幻觉、安全、价值对齐和评估标准成为核心挑战。",
    keyWorks: [
      {
        name: "RLHF",
        contribution: "把人类偏好转化为奖励信号，改善回答可用性。",
        modulePath: "../04-Alignment-OpenSource/alignment/",
      },
      {
        name: "Instruction Tuning",
        contribution: "让模型更稳定地理解任务指令。",
      },
    ],
    relatedModules: [
      {
        label: "对齐",
        path: "../04-Alignment-OpenSource/alignment/",
      },
      {
        label: "提示工程",
        path: "../03-Scale-Multimodal/prompt-engineering/",
      },
    ],
  },
  {
    year: "2023",
    title: "GPT-4 与 LLaMA：开源的反击",
    shortTitle: "GPT-4 / LLaMA",
    phase: "开源与多模态",
    previousLimit: "前沿大模型权重掌握在少数闭源公司手里，研究者难以直接复现实验。",
    whatHappened: "GPT-4 展示更强推理和多模态能力，LLaMA 点燃开源模型生态。",
    solved: "闭源能力上限与开源可及性同时提升，社区开始快速复现、微调和部署大模型。",
    newProblems: "模型评测、许可边界、部署成本和安全治理更加复杂。",
    keyWorks: [
      {
        name: "GPT-4",
        contribution: "显著提升复杂推理、多任务和多模态能力。",
      },
      {
        name: "LLaMA",
        contribution: "让高质量基座模型进入开源研究和工程社区。",
        modulePath: "../04-Alignment-OpenSource/",
      },
    ],
    relatedModules: [
      {
        label: "对齐与开源",
        path: "../04-Alignment-OpenSource/",
      },
      {
        label: "模型服务",
        path: "../05-Systems-Production/model-serving/",
      },
    ],
  },
  {
    year: "2024",
    title: "MoE、长上下文、o1：推理时慢思考",
    shortTitle: "MoE / o1",
    phase: "系统生产",
    previousLimit: "大模型推理成本随规模上涨，长上下文和复杂推理一步错就会连锁失败。",
    whatHappened: "MoE 降低单次激活参数比例，长上下文扩展输入窗口，o1 类模型强调推理时计算。",
    solved: "能力提升不再只依赖训练期堆参数，推理期计算成为重要方向。",
    newProblems: "路由、缓存、延迟、成本和评估都变成系统级问题。",
    keyWorks: [
      {
        name: "MoE",
        contribution: "按 token 激活部分专家，在能力和成本之间折中。",
        modulePath: "../05-Systems-Production/model-serving/architecture/",
      },
      {
        name: "Long Context",
        contribution: "让模型处理更长文档和复杂任务上下文。",
      },
    ],
    relatedModules: [
      {
        label: "模型服务架构",
        path: "../05-Systems-Production/model-serving/architecture/",
      },
      {
        label: "训练基础设施",
        path: "../05-Systems-Production/training-infrastructure/",
      },
    ],
  },
  {
    year: "2025",
    title: "DeepSeek R1 与 Test-Time Compute：开源追平",
    shortTitle: "DeepSeek R1",
    phase: "推理模型",
    previousLimit: "推理模型被认为是少数闭源公司的专属能力，社区缺少可验证的开源路径。",
    whatHappened: "DeepSeek R1 展示开源推理模型的强竞争力，Test-Time Compute 成为提升复杂推理的重要杠杆。",
    solved: "开源社区证明推理能力可以被系统化训练和复现。",
    newProblems: "推理轨迹、奖励设计、蒸馏质量和生产成本需要新的工程方法。",
    keyWorks: [
      {
        name: "DeepSeek R1",
        contribution: "开源推理模型展示接近闭源前沿的复杂推理能力。",
      },
      {
        name: "Test-Time Compute",
        contribution: "通过推理时更多计算换取更稳的复杂问题求解。",
      },
    ],
    relatedModules: [
      {
        label: "模型服务",
        path: "../05-Systems-Production/model-serving/",
      },
      {
        label: "评估与监控",
        path: "../05-Systems-Production/monitoring/",
      },
    ],
  },
];

export function getNodeByYear(year: string): TimelineNode | undefined {
  return timelineNodes.find((node) => node.year === year);
}
