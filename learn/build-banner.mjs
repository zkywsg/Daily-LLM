// 生成 README 顶部横幅 .github/assets/banner.png：整体骨架（6 层）+ 已完成部分的真实图卡
// 用法：node learn/build-banner.mjs（需要 Edge）
import { writeFileSync, readFileSync, mkdtempSync, readdirSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const assets = join(here, "../.github/assets");
const edge = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge";
const R = JSON.parse(readFileSync(join(here, "roadmap.json"), "utf8"));
const tmp = mkdtempSync(join(tmpdir(), "banner-"));
const nodes = R.layers.flatMap((l) => l.groups.flatMap((g) => g.nodes));
const cnt = (xs) => xs.reduce((a, x) => a + x.items.length, 0);
const total = cnt(nodes), done = cnt(nodes.filter((x) => x.s === "done"));
const doneNames = R.layers.flatMap((l) => l.groups.flatMap((g) => g.nodes)).filter((x) => x.s === "done").map((x) => x.t.replace(" 完整模型", "").replace("系列", "")).join("、");
const dirs = readdirSync(here).filter((l) => /^\d+-/.test(l)).flatMap((l) => readdirSync(join(here, l)).filter((d) => !d.startsWith("_")).map((d) => `${l}/${d}`));
const nCards = dirs.reduce((a, d) => { const o = join(here, d, "cards/out"); return a + (existsSync(o) ? readdirSync(o).filter((f) => f.endsWith(".png")).length : 0); }, 0);
const nDemos = dirs.filter((d) => existsSync(join(here, d, "demo/index.html"))).length;
const nDocs = dirs.filter((d) => existsSync(join(here, d, "README.md"))).length;
const card = (t, n) => "file://" + join(here, `${t}/cards/out/${n}.png`);
const rows = R.layers.map((l) => {
  const ns = l.groups.flatMap((g) => g.nodes), n = cnt(ns), d = cnt(ns.filter((x) => x.s === "done"));
  const ex = ns.map((x) => x.t).slice(0, 6).join(" · ");
  return `<div class="r${d ? " on" : ""}"><div class="n">${l.id}</div><div class="m"><b>${l.name}</b><span>${ex}</span></div><div class="c"><b>${d ? d + " / " : ""}${n}</b><span>知识点</span><i><u style="width:${Math.round((d / n) * 100)}%"></u></i></div></div>`;
}).join("");
const thumbs = [["0-math/la-04-svd", "03"], ["3-llm/01-attention", "06"], ["3-llm/attn-06-position", "05"], ["3-llm/tf-01-block", "05"], ["3-llm/tf-04-mini-gpt", "04"]]
  .map(([t, n]) => `<div class="th"><img src="${card(t, n)}"></div>`).join("");
const html = `<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0}
body{width:1600px;height:720px;background:#000;color:#e8eaed;font:16px/1.4 "PingFang SC","SF Pro Text",-apple-system,sans-serif;position:relative;overflow:hidden}
.glow{position:absolute;left:-200px;top:-200px;width:900px;height:900px;background:radial-gradient(closest-side,#0b2a66 0,transparent 70%);opacity:.5}
.l{position:absolute;left:64px;top:52px;width:640px}
.k{font:700 15px Menlo,monospace;letter-spacing:.14em;color:#3b82f6}
h1{font-size:64px;line-height:1.1;margin:14px 0 14px;letter-spacing:.01em}
h2{font-size:40px;line-height:1.2;color:#3b82f6;margin-bottom:18px}
.t{font-size:20px;color:#a3aab5;line-height:1.6;width:600px}
.st{display:flex;gap:30px;margin-top:24px}.st b{display:block;font-size:32px;color:#fff}.st span{color:#8b929c;font-size:14px}
.dn{margin-top:26px;color:#7aa7f0;font-size:14px}
.ths{display:flex;gap:12px;margin-top:10px}
.th{width:94px;height:167px;border-radius:10px;border:1px solid #2a3140;overflow:hidden;background:#000}.th img{width:100%;display:block}
.R{position:absolute;left:760px;top:44px;width:780px;display:flex;flex-direction:column;gap:9px}
.r{display:flex;align-items:center;gap:18px;height:98px;border:1px solid #1f2430;border-radius:14px;padding:0 22px;background:#06070a}
.r.on{border-color:#3b82f6;background:#050b18}
.n{font:700 38px Menlo,monospace;color:#2a3140;width:38px}.r.on .n{color:#3b82f6}
.m{flex:1}.m b{display:block;font-size:22px}.m span{color:#8b929c;font-size:14px;line-height:1.5;display:block;margin-top:3px}
.c{width:150px;text-align:right;white-space:nowrap}.c b{font-size:24px}.c span{color:#8b929c;font-size:12px;margin-left:5px}
.c i{display:block;height:5px;border-radius:3px;background:#161b26;margin-top:8px}.c u{display:block;height:5px;border-radius:3px;background:#3b82f6}
</style><body><div class="glow"></div>
<div class="l"><div class="k">DAILY-LLM</div>
<h1>从零学</h1><h2>机器学习 · 深度学习 · 大模型</h2>
<div class="t">把所有知识自己学一遍，并且全部呈现在这个仓库里：每个知识点一篇讲解，配互动页和 TikTok 图卡。</div>
<div class="st"><div><b>${R.layers.length}</b><span>层</span></div><div><b>${nodes.length}</b><span>个知识块</span></div><div><b>${total}</b><span>个知识点</span></div><div><b>${done}</b><span>已完成</span></div></div>
<div class="dn">已完成：${doneNames}（${nDocs} 篇，${nCards} 张图卡，${nDemos} 个互动页）</div>
<div class="ths">${thumbs}</div></div>
<div class="R">${rows}</div></body>`;
const hp = join(tmp, "banner.html");
writeFileSync(hp, html);
execFileSync(edge, ["--headless", "--disable-gpu", "--hide-scrollbars", "--window-size=1600,720", `--screenshot=${join(assets, "banner.png")}`, "file://" + hp], { stdio: "ignore" });
console.log("banner ok");
