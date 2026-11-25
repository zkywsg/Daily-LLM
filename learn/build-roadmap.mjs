// 由 roadmap.json 生成：路线图 PNG（.github/assets/roadmap.png）和 learn/ROADMAP.md
// 用法：node learn/build-roadmap.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const R = JSON.parse(readFileSync(join(here, "roadmap.json"), "utf8"));
const assets = join(here, "../.github/assets");
const st = (x) => x.s || "planned";
const all = R.layers.flatMap((l) => l.groups.flatMap((g) => g.nodes));
const cnt = (s) => all.filter((x) => st(x) === s).length;
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

const chip = (x) => `<span class="c ${st(x)}">${esc(x.t)}${x.n ? `<i>${esc(x.n)}</i>` : ""}</span>`;
const rows = R.layers.map((l) => `<div class="row${l.id === 3 ? " on" : ""}">
<div class="lab"><b>${l.id} · ${esc(l.name)}</b><small>${esc(l.goal)}</small></div>
<div class="gs">${l.groups.map((g) => `<div class="g"><div class="gn">${esc(g.name)}</div><div class="chips">${g.nodes.map(chip).join("")}</div></div>`).join("")}</div></div>`).join("");
const html = `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0}
body{width:1600px;background:#000;color:#e8eaed;font:16px/1.4 "PingFang SC","SF Pro Text",-apple-system,sans-serif;padding:48px 56px 40px}
.k{font:700 15px Menlo,monospace;letter-spacing:.14em;color:#3b82f6}
h1{font-size:42px;margin:8px 0 6px}.sub{color:#8b929c;font-size:18px}
.top{display:flex;justify-content:space-between;align-items:flex-end}
.leg{display:flex;gap:20px;color:#8b929c;font-size:15px;align-items:center}
.dot{display:inline-block;width:14px;height:14px;border-radius:4px;margin-right:8px;vertical-align:-2px}
.rows{margin-top:26px;display:flex;flex-direction:column;gap:12px}
.row{display:flex;border:1px solid #1f2430;border-radius:14px;padding:18px 22px;gap:20px}
.row.on{border-color:#3b82f6;background:#050b18}
.lab{width:210px;flex:none;padding-top:4px}.lab b{display:block;font-size:21px}.lab small{color:#8b929c;font-size:14px}
.row.on .lab small{color:#7aa7f0}
.gs{display:flex;flex-direction:column;gap:12px;flex:1}
.g{display:flex;gap:14px;align-items:flex-start}
.gn{width:86px;flex:none;color:#5b6472;font-size:13px;padding-top:9px}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.c{padding:7px 14px;border-radius:9px;border:1.5px dashed #3a4150;color:#8b929c;font-size:15px}
.c i{font-style:normal;margin-left:8px;font-size:12px;opacity:.8}
.c.done{background:#3b82f6;border:0;color:#fff;font-weight:700;padding:8.5px 15.5px}
.c.partial{border:1.5px solid #3b82f6;color:#9db8ea}
</style><body>
<div class="top"><div><div class="k">DAILY-LLM · ROADMAP</div><h1>${esc(R.title)}</h1>
<div class="sub">${R.layers.length} 层 · ${all.length} 个知识块。已完成 ${cnt("done")}，部分覆盖 ${cnt("partial")}，其余待写。</div></div>
<div class="leg"><span><span class="dot" style="background:#3b82f6"></span>已完成</span><span><span class="dot" style="border:1.5px solid #3b82f6"></span>部分覆盖</span><span><span class="dot" style="border:1.5px dashed #4b5563"></span>待写</span></div></div>
<div class="rows">${rows}</div></body></html>`;
const hp = join(assets, "roadmap.html");
writeFileSync(hp, html);
const edge = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge";
const H = 1100;
execFileSync(edge, ["--headless", "--disable-gpu", "--hide-scrollbars", `--window-size=1600,${H}`, `--screenshot=${join(assets, "roadmap.png")}`, "file://" + hp], { stdio: "ignore" });

const label = { done: "已完成", partial: "部分覆盖", planned: "待写" };
let md = `# ${R.title}\n\n> 由 [roadmap.json](roadmap.json) 生成（\`node learn/build-roadmap.mjs\`）。${R.layers.length} 层、${all.length} 个知识块：已完成 ${cnt("done")}，部分覆盖 ${cnt("partial")}，待写 ${cnt("planned")}。\n> “旧版素材”指 \`_archive/v1/\` 里可取材的旧内容（质量未重新核对，写新版时需重写）。\n\n![](../.github/assets/roadmap.png)\n`;
for (const l of R.layers) {
  md += `\n## ${l.id} · ${l.name}\n\n${l.goal}\n\n| 分组 | 知识块 | 状态 | 说明 / 旧版素材 |\n|---|---|---|---|\n`;
  for (const g of l.groups) for (const x of g.nodes) {
    const t = x.link ? `[${x.t}](${x.link})` : x.t;
    const note = [x.n, x.old ? `\`${x.old}\`` : ""].filter(Boolean).join("；");
    md += `| ${g.name} | ${t} | ${label[st(x)]} | ${note} |\n`;
  }
}
writeFileSync(join(here, "ROADMAP.md"), md);
console.log("roadmap:", all.length, "nodes");
