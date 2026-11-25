// 08 真实模型 · 图卡。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { build } from "../../_shared/cardkit.mjs";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const code = (s) => `<pre style="font:32px/1.6 ui-monospace,Menlo,monospace;white-space:pre-wrap;color:#ededed">${s}</pre>`;
const cards = [
 `<div class="tag">注意力 · 实测</div><h1>真实模型的注意力，<br>长什么样？</h1><p class="mute">用 12 行代码取出来看 →</p>`,
 `<div class="tag">怎么取</div><h2>GPT-2 每层每头<br>都能导出来</h2><div class="panel">${code(`model = AutoModel.from_pretrained(\n  "gpt2",\n  attn_implementation="eager",\n  output_attentions=True)\nout = model(**enc)\nout.attentions  # 12 层\n# 每层 [batch, 12 头, n, n]`)}</div><p class="mute s">脚本已过语法检查，尚未在真实模型上运行。</p>`,
 `<div class="tag">该找什么</div><h2>三种文献里的“习惯”</h2><div class="panel"><div class="big" style="font-size:44px;line-height:1.7"><span class="a">上一个词</span>：看对角线下一格<br><span class="a">第一个词</span>：大量权重落在开头<br><span class="a">归纳头</span>：重复模式里看上次的下一个</div></div><p class="mute s">文献描述（Elhage 2021；Olsson 2022；Xiao 2023），本仓库未复现。</p>`,
 `<div class="tag">自己试</div><h2>四个小练习</h2><div class="panel"><p class="s" style="margin:0;line-height:1.8">① 找最像“上一个词”的头<br>② 用重复序列找归纳头候选<br>③ 统计各层对第一个词的权重<br>④ 换同义词，看图变多少</p></div><p class="mute s">答案取决于你的运行结果。</p>`,
];
build(root, "真实模型", cards);
