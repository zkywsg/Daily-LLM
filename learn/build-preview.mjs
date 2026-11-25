// 生成 learn/preview.html：把每个知识点的讲解、图卡、互动版按顺序放在一页里。
// 用法：node learn/build-preview.mjs && open learn/preview.html
// 自动扫描 learn/<层>/<知识点>/README.md，新增知识点后重跑即可。
import { createRequire } from "node:module";
import { readdirSync, readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, "./package.json"));
const { marked } = require("marked");
const katex = require("katex");

// KaTeX 样式：字体内联为 base64，页面完全自包含、可离线
const kdist = join(dirname(require.resolve("katex")), "..", "dist");
let kcss = readFileSync(join(kdist, "katex.min.css"), "utf8");
kcss = kcss.replace(/url\(fonts\/([^)]+?\.woff2)\)\s*format\("woff2"\)(?:,\s*url\([^)]*\)[^,;}]*)*/g, (_, f) =>
  `url(data:font/woff2;base64,${readFileSync(join(kdist, "fonts", f)).toString("base64")}) format("woff2")`);

function renderMath(md) {
  // 跳过围栏代码块，只处理正文里的 $$..$$ 与 $..$
  return md.split(/(```[\s\S]*?```)/g).map((part, i) => {
    if (i % 2) return part;
    part = part.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => katex.renderToString(t.trim(), { displayMode: true, throwOnError: false }));
    return part.replace(/\$([^$\n]+?)\$/g, (_, t) => katex.renderToString(t, { throwOnError: false }));
  }).join("");
}

const topics = [];
for (const layer of readdirSync(here).sort()) {
  const lp = join(here, layer);
  if (!statSync(lp).isDirectory()) continue;
  for (const t of readdirSync(lp).sort()) {
    const dir = join(lp, t);
    if (!existsSync(join(dir, "README.md"))) continue;
    const rel = `${layer}/${t}`;
    const md = readFileSync(join(dir, "README.md"), "utf8");
    const title = (md.match(/^#\s+(.+)$/m) || [, t])[1];
    let html = marked.parse(renderMath(md));
    // 指向兄弟知识点目录的链接（../xxx/）改成页内锚点，这样在线站点里也能跳转
    html = html.replace(/href="\.\.\/([^/"]+)\/?"/g, (_, d) => `href="#${d}"`);
    html = html.replace(/(src|href)="(?!https?:|#|data:)([^"]+)"/g, (_, a, u) => `${a}="${rel}/${u}"`);
    const cardsDir = join(dir, "cards/out");
    const cards = existsSync(cardsDir) ? readdirSync(cardsDir).filter((f) => f.endsWith(".png")).sort() : [];
    const demo = existsSync(join(dir, "demo/index.html")) ? `${rel}/demo/index.html` : null;
    topics.push({ rel, title, html, cards: cards.map((c) => `${rel}/cards/out/${c}`), demo });
  }
}

const short = (t) => t.title.replace(/^注意力 (\d+)：/, "$1 · ").replace(/^基础 ([ABC])：/, "基础 $1 · ");
const side = topics.map((t, i) => `<div class="g"><a class="t" href="#t${i}" data-t="t${i}">${short(t)}</a>
  <div class="sub"><a href="#t${i}-doc">讲解</a><a href="#t${i}-cards">图卡 ${t.cards.length}</a>${t.demo ? `<a href="#t${i}-demo">互动版</a>` : ""}</div></div>`).join("");
const body = topics.map((t, i) => `
<article id="t${i}"><a id="${t.rel.split("/")[1]}"></a>
  <div class="path">${t.rel}</div>
  <section id="t${i}-doc" class="doc">${t.html}</section>
  <section id="t${i}-cards" class="blk"><div class="bh"><h2>图卡</h2><span>TikTok 竖版 · ${t.cards.length} 张 · 横向滑动，点图放大</span></div>
    <div class="strip">${t.cards.map((c, k) => `<a class="card" href="${c}" target="_blank"><img loading="lazy" src="${c}" alt="第 ${k + 1} 张"><span>${String(k + 1).padStart(2, "0")}</span></a>`).join("")}</div></section>
  ${t.demo ? `<section id="t${i}-demo" class="blk"><div class="bh"><h2>互动版</h2><span><a href="${t.demo}" target="_blank">新标签页打开 ↗</a></span></div>
    <iframe src="${t.demo}" title="互动版" loading="lazy"></iframe></section>` : ""}
</article>`).join("");

writeFileSync(join(here, "preview.html"), `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily-LLM · 从零学大模型</title>
<style>${kcss}</style>
<style>
:root{color-scheme:dark;--bg:#000;--card:#0a0a0a;--raised:#111;--line:#1f1f1f;--line2:#2e2e2e;--ink:#ededed;--mute:#8a8a8a;--accent:#3b82f6;--accent-bg:rgba(59,130,246,.12);
 --mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,"PingFang SC","Noto Sans SC",sans-serif}
*{box-sizing:border-box}html{background:#000;scroll-behavior:smooth;scroll-padding-top:20px}
body{margin:0;background:#000;color:var(--ink);font:15px/1.75 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
aside{position:fixed;inset:0 auto 0 0;width:248px;border-right:1px solid var(--line);padding:22px 18px;overflow-y:auto;background:#000}
aside h1{font-size:14px;margin:0;font-weight:700;letter-spacing:-.01em}aside .meta{font:11.5px var(--mono);color:var(--mute);margin:4px 0 22px}
aside .grp{font:600 11px var(--mono);color:var(--mute);letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px}
aside .g{margin-bottom:6px;padding-left:10px;border-left:1px solid var(--line)}aside .g.on{border-left-color:var(--accent)}
aside .t{display:block;color:var(--ink);font-size:13.5px;line-height:1.5;padding:3px 0}aside .g.on .t{color:var(--accent)}
aside .sub{display:flex;gap:12px;padding-bottom:6px}aside .sub a{font-size:12px;color:var(--mute)}aside .sub a:hover{color:var(--ink);text-decoration:none}
main{margin-left:248px;padding:36px 40px 120px;max-width:1040px}
article{margin-bottom:96px}.path{font:11.5px var(--mono);color:var(--mute);margin-bottom:10px;letter-spacing:.02em}
section{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:28px 32px;margin:14px 0}
.bh{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px}.bh h2{margin:0;font-size:15px;font-weight:600}.bh span{font-size:12px;color:var(--mute)}
.doc h1{font-size:28px;line-height:1.3;margin:0 0 14px;font-weight:700;letter-spacing:-.025em}
.doc h2{font-size:19px;margin:2em 0 .6em;padding-bottom:8px;border-bottom:1px solid var(--line);font-weight:650;letter-spacing:-.015em}
.doc h3{font-size:16px;margin:1.6em 0 .4em;font-weight:600}.doc hr{border:0;border-top:1px solid var(--line);margin:2em 0}
.doc p,.doc li{color:#d4d4d4}.doc strong{color:#fff}.doc img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.doc table{border-collapse:collapse;width:100%;font-size:14px;margin:1em 0}.doc th,.doc td{border:1px solid var(--line);padding:7px 12px;text-align:left}.doc th{background:var(--raised);font-weight:600;color:#fff}
.doc pre{background:#000;border:1px solid var(--line);border-radius:6px;padding:14px 16px;overflow-x:auto;font:13px/1.65 var(--mono)}
.doc code{font-family:var(--mono);font-size:.9em;background:var(--raised);padding:1px 5px;border-radius:3px}.doc pre code{background:none;padding:0}
.doc blockquote{margin:1em 0;padding:2px 16px;border-left:2px solid var(--accent);background:var(--accent-bg);border-radius:0 4px 4px 0}.doc blockquote p{color:#bcd}
.doc em{color:var(--mute)}.katex{color:var(--ink);font-size:1.06em}
.strip{display:flex;gap:14px;overflow-x:auto;scroll-snap-type:x proximity;padding:2px 2px 12px}
.card{flex:0 0 216px;scroll-snap-align:start;color:var(--mute);font:11.5px var(--mono)}.card:hover{text-decoration:none}
.card img{width:216px;height:384px;object-fit:cover;display:block;border:1px solid var(--line);border-radius:6px;margin-bottom:6px;transition:border-color .15s}.card:hover img{border-color:var(--accent)}
iframe{width:100%;height:1100px;border:1px solid var(--line);border-radius:6px;background:#000;display:block}
@media(max-width:900px){aside{display:none}main{margin:0;padding:20px 14px 80px}section{padding:18px}}
</style></head><body>
<aside><h1>Daily-LLM · 学习资料</h1><div class="meta">${topics.length} 个知识点 · 离线可看</div><div class="grp">3 · 大模型 / 注意力</div>${side}</aside>
<main>${body}</main>
<script>
(function(){var g=[].slice.call(document.querySelectorAll("aside .g"));var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){var id=e.target.id;g.forEach(function(x){x.classList.toggle("on",x.querySelector(".t").dataset.t===id)})}})},{rootMargin:"-10% 0px -80% 0px"});
document.querySelectorAll("article").forEach(function(a){io.observe(a)});if(g[0])g[0].classList.add("on")})();
</script></body></html>`);
console.log("topics:", topics.map((t) => t.rel).join(", "));
