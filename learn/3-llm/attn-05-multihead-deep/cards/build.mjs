// 05 多头深入 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const cards = [
 `<div class="tag">注意力 · 多头</div><h1>多头注意力里，<br>“拼接”其实是<span class="a">求和</span></h1><p class="mute">拆开看，每个头在做什么 →</p>`,
 `<div class="tag">等价写法</div><h2>拼接 × W<sub>O</sub><br>= 各头贡献相加</h2><div class="panel" style="text-align:center"><div class="big" style="font-size:50px">[O₁ O₂ … O<sub>h</sub>] · W<sub>O</sub><br>= Σ O<sub>i</sub> · W<sub>O</sub><sup>(i)</sup></div></div><p>把 W<sub>O</sub> 按行切成 h 块，每个头只用自己那一块。这是分块矩阵乘法。</p><p class="mute s">NumPy 验证：两种写法最大差 3.6×10⁻¹⁵。</p>`,
 `<div class="tag">每个头的两件事</div><h2>QK 管“看谁”<br>OV 管“写什么”</h2><div class="panel"><div class="big" style="font-size:50px"><span class="a">W<sub>Q</sub>W<sub>K</sub>ᵀ</span> → 谁看谁<br><span class="a">W<sub>V</sub>W<sub>O</sub></span> → 看到后写什么</div></div><p class="mute s">两者都是低秩矩阵，秩不超过 d<sub>k</sub>。（Elhage et al. 2021 的视角）</p>`,
 `<div class="tag">为什么 d<sub>k</sub> = d / h</div><h2>预算不变，<br>换成多种规则</h2><p>每个头的维度缩小到 d/h，h 个头合起来参数和计算量与一个“全宽”的头基本一样。</p><div class="panel"><div class="big" style="font-size:48px">1 个头 × d 维<br>≈ h 个头 × (d/h) 维</div></div><p class="mute s">换来的是：同时有 h 套匹配规则。</p>`,
 `<div class="tag">去掉一个头？</div><h2>常常没事，<br>但不能说它没用</h2><p>文献里发现，翻译模型可以在测试时剪掉大比例的头而性能几乎不变（Michel 2019）；最重要的头往往有可解释的功能（Voita 2019）。</p><p class="mute s">但编码器-解码器注意力对剪枝更敏感。结论因任务而异，不能推广成“头都是冗余的”。</p>`,
];
build(root, "多头深入", cards);
