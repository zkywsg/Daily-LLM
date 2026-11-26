// T03 三种架构 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { TFV, build, fig, table, panel } from "../../_shared/tfcards.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const en = ["我", "喜欢", "那只", "猫"], de = ["<s>", "I", "like", "cats"];
const big = (rows, cols, ok, color) => fig(TFV.mask(rows, cols, ok, { cell: 44, color }), 640);
const cards = [
  `<div class="tag">Transformer · 第 3 块积木</div><h1>BERT、GPT、T5<br>差在<span class="a">哪里</span>？</h1><p class="mute">零件一样，看两件事 →</p>`,
  `<div class="tag">编码器 · BERT</div><h2>每个词<br><span class="a">看全句</span></h2>${big(en, en, () => true)}<p class="s mute">行 = 提问的词，列 = 被看的词。适合理解：分类、抽取、检索。</p>`,
  `<div class="tag">解码器 · GPT</div><h2>只能<span class="a">看左边</span></h2>${big(de, de, (i, j) => j <= i)}<p class="s mute">训练时猜下一个词，右边就是答案，所以必须遮住。</p>`,
  `<div class="tag">编码器 + 解码器 · T5</div><h2>交叉注意力：<br>解码时<span class="a" style="color:#e5a50a">回看源句</span></h2>${big(de, en, () => true, TFV.C.amber)}<p class="s mute">Q 来自解码器，K、V 来自编码器。表是 m × n，源句只需编码一次。</p>`,
  `<div class="tag">训练目标 · BERT</div><h2>挖掉 15% 的词，<br>让它猜回来</h2><div class="panel">${[["换成 [MASK]", 80, "#3b82f6"], ["换成随机词", 10, "#e5a50a"], ["保持原样", 10, "#22c55e"]].map(([t, v, c]) => `<div class="row" style="grid-template-columns:260px 1fr 110px;font-size:38px"><span>${t}</span><div class="bar"><i style="width:${v}%;background:${c}"></i></div><span class="num">${v}%</span></div>`).join("")}</div><p class="s mute">后两种，是为了让模型在没有 [MASK] 时也认真处理每个词。</p>`,
  `<div class="tag">一张表记住</div><h2>三种架构</h2>${table(["代表", "能看谁", "猜什么"], [["BERT", "全句", "被遮的词"], ["GPT", "左边", "下一个词"], ["T5", "源句 + 左边", "目标文字"]], [1])}<p class="s mute">现在的大语言模型多是“只有解码器”。编码器仍常用于检索和分类。</p>`,
];
build(root, "三种架构", cards);
