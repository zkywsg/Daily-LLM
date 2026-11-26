// 生成可发布到 GitHub Pages 的静态站点：node learn/build-site.mjs  →  learn/_site/
// 站点 = 预览总页(index.html) + 各知识点的互动页、图卡、共享脚本。需要先在 learn/ 下 npm ci。
import { execFileSync } from "node:child_process";
import { cpSync, rmSync, readdirSync, readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url)), out = join(here, "_site");
execFileSync("node", [join(here, "build-preview.mjs")], { stdio: "inherit" });
rmSync(out, { recursive: true, force: true });
// 复制内容，跳过图卡源 HTML（只发布 PNG）、Python 缓存
cpSync(join(here, "3-llm"), join(out, "3-llm"), {
  recursive: true,
  filter: (src) => !/[\\/]cards[\\/]src(?:[\\/]|$)|__pycache__|\.DS_Store/.test(src),
});
cpSync(join(here, "preview.html"), join(out, "index.html"));
writeFileSync(join(out, ".nojekyll"), "");
// 互动页里的“讲解”链接指向 README.md，在站点里改成指向总页对应位置
for (const t of readdirSync(join(out, "3-llm"))) {
  const f = join(out, "3-llm", t, "demo/index.html");
  if (!existsSync(f) || !statSync(f).isFile()) continue;
  writeFileSync(f, readFileSync(f, "utf8").replace(/href="\.\.\/README\.md"/g, `href="../../../index.html#${t}" target="_top"`));
}
console.log("site ->", out);
