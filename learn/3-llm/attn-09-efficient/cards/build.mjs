// 09 高效注意力 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const bar = (t, w, c, v) => `<div class="row" style="grid-template-columns:220px 1fr 160px;font-size:36px"><span>${t}</span><div class="bar"><i style="width:${w}%;background:${c}"></i></div><span class="num">${v}</span></div>`;
const cards = [
 `<div class="tag">注意力 · 提速</div><h1>大模型怎么<br>让注意力<span class="a">又快又省</span>？</h1><p class="mute">四个办法，5 张图 →</p>`,
 `<div class="tag">办法 1 · KV cache</div><h2>旧词的 K、V<br>不会变，存起来</h2><p>生成每个新词时，只算新词那一行，旧词的键和值直接取缓存。</p><div class="panel"><div class="big" style="font-size:48px">生成 100 个词：<br>有缓存 <span class="a">5,050</span> 次<br>无缓存 338,350 次</div></div><p class="mute s">只比较 q·k 点积次数，约 67 倍。代价：占内存。</p>`,
 `<div class="tag">缓存有多大</div><h2>每个词 4.5 MiB</h2><p>GPT-3 175B：96 层，96 个头，每头 128 维，fp16：</p><div class="panel"><div class="big" style="font-size:44px">2 × 96 × 96 × 128 × 2 字节<br>= <span class="a">4.5 MiB / 词</span><br>2048 词 ≈ <span class="a">9 GiB</span></div></div><p class="mute s">一个序列就这么大。（配置已对照公开资料核对。）</p>`,
 `<div class="tag">办法 2 · GQA</div><h2>让多个 q 头<br>共用一份 K/V</h2><div class="panel">${bar("MHA", 100, "#8a8a8a", "8×")}${bar("GQA（8 组）", 12.5, "#3b82f6", "1×")}${bar("MQA", 1.6, "#22c55e", "0.13×")}</div><p class="mute s">以 64 个 q 头为例：KV 头从 64 减到 8，缓存缩小 8 倍；减到 1 就是 MQA。</p>`,
 `<div class="tag">办法 3 · FlashAttention</div><h2>结果不变，<br>只改<span class="a">算的顺序</span></h2><p>分块读入，用“在线 softmax”边读边合并：</p><div class="panel"><div class="big" style="font-size:42px">m′ = max(m, m<sub>块</sub>)<br>ℓ′ = ℓ·e<sup>m−m′</sup> + Σe<sup>s−m′</sup><br>o′ = o·e<sup>m−m′</sup> + Σe<sup>s−m′</sup>v</div></div><p class="mute s">精确结果，不是近似。快在少读写慢速显存。</p>`,
];
build(root, "高效注意力", cards);
