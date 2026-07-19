import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";

// 仓库根目录:vitest 从 web/ 下执行,上一级就是仓库根
const REPO_ROOT = path.resolve(process.cwd(), "..");

// 找出所有 `NN-family/assets/*.svg`(全仓库,不局限于金标本节点用到的)。
function findSvgFiles(root: string): string[] {
  const families = fs
    .readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^\d{2}-/.test(d.name));

  const files: string[] = [];
  for (const fam of families) {
    const assetsDir = path.join(root, fam.name, "assets");
    if (!fs.existsSync(assetsDir)) continue;
    for (const f of fs.readdirSync(assetsDir)) {
      if (f.endsWith(".svg")) files.push(path.join(assetsDir, f));
    }
  }
  return files;
}

// 之前手工用 python3 xml.etree 扫过一次仓库,抓到 4 个非法 XML 的 SVG
// (字面 `<`/`<<` 没转义、重复 fill 属性),都会导致浏览器渲染成 0×0
// 空白图但不报任何 console error,肉眼很难发现。这里把那次手工检查
// 沉淀成永久回归测试,覆盖全仓库而不是抽查。
describe("SVG 资源合法性检查", () => {
  const svgFiles = findSvgFiles(REPO_ROOT);

  it("扫描到足够多的 SVG 文件", () => {
    expect(svgFiles.length).toBeGreaterThan(100);
  });

  it.each(svgFiles)("%s 是合法的 XML", (file) => {
    const content = fs.readFileSync(file, "utf-8");
    const doc = new DOMParser().parseFromString(content, "image/svg+xml");
    const parserError = doc.querySelector("parsererror");
    expect(
      parserError,
      `${path.relative(REPO_ROOT, file)} XML 解析失败: ${parserError?.textContent?.trim()}`
    ).toBeNull();
  });
});
