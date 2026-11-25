// 10 自测 · 图卡 + QUESTIONS.md。用法：node cards/build.mjs
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import { writeFileSync } from "node:fs";
import { build } from "../../_shared/cardkit.mjs";
const require = createRequire(import.meta.url);
const Q = require("../demo/questions.js");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// 在 GitHub 上直接可读的题库（答案折叠）
const L = "ABCD";
let md = "# 注意力系列自测题库\n\n单选题，共 " + Q.length + " 题。点开“答案”查看。也可以用[网页版](demo/index.html)随机出题。\n\n";
Q.forEach((x, i) => { md += `**${i + 1}. [${x.t}]** ${x.q}\n\n` + x.o.map((o, k) => `- ${L[k]}. ${o}`).join("\n") + `\n\n<details><summary>答案</summary>\n\n${L[x.a]}。${x.e}\n\n</details>\n\n`; });
writeFileSync(join(root, "QUESTIONS.md"), md);
const pick = [0, 12, 21];
const qc = (x) => `<div class="panel"><p class="s" style="margin:0 0 14px">${x.q}</p>${x.o.map((o, k) => `<div style="font-size:36px;margin:10px 0;color:#ededed">${L[k]}. ${o}</div>`).join("")}</div>`;
const cards = [
 `<div class="tag">注意力 · 自测</div><h1>学完 9 篇，<br>你<span class="a">真懂了吗</span>？</h1><p class="mute">3 道题试试 →</p>`,
 `<div class="tag">第 1 题</div><h2>${Q[12].t}</h2>${qc(Q[12])}<p class="mute s">答案在下一张</p>`,
 `<div class="tag">第 2 题</div><h2>${Q[21].t}</h2>${qc(Q[21])}<p class="mute s">答案在最后一张</p>`,
 `<div class="tag">答案</div><h2>两题答案</h2><div class="panel"><p class="s" style="margin:0 0 20px"><b class="a">第 1 题：${L[Q[12].a]}</b>　${Q[12].e}</p><p class="s" style="margin:0"><b class="a">第 2 题：${L[Q[21].a]}</b>　${Q[21].e}</p></div><p class="mute s">全部 ${Q.length} 题在仓库里：QUESTIONS.md</p>`,
];
build(root, "自测", cards);
