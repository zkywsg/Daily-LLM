import { describe, it, expect } from "vitest";

// 每个金标本节点的 lib/prose.ts 都 export 一个 extractProse() 和一个
// `XXX_SOURCE_PATH` 常量(指向对应的 markdown 正文)。这里把两者配对,
// 对真实 markdown 跑一遍 extractProse,断言提取出的每个字段都非空——
// 专门抓"H2/H3 标题正则没匹配上导致某段 prose 静默变成空字符串"这类
// 冒烟测试(只看有没有 h1)测不出来的问题。
//
// dit-vit/ 没有自己的 lib/prose.ts(复用 dit/ 的组件),不在扫描范围内,
// 属已知例外——它的两个 footer 小节(DiT vs U-Net 对比 / DiT 的衍生)
// 在源 markdown 里本就不存在,是有意留空,不是 bug。

type ProseModule = Record<string, unknown>;

const proseModules = import.meta.glob<ProseModule>("./*/lib/prose.ts", {
  eager: true,
});

const markdownModules = import.meta.glob("../../../../../[0-9][0-9]-*/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

function findSourcePath(mod: ProseModule): string | undefined {
  const entry = Object.entries(mod).find(
    ([key, value]) => key.endsWith("_SOURCE_PATH") && typeof value === "string"
  );
  return entry?.[1] as string | undefined;
}

function findMarkdown(sourcePath: string): string | undefined {
  const key = Object.keys(markdownModules).find((k) => k.endsWith(`/${sourcePath}`));
  return key ? markdownModules[key] : undefined;
}

describe("金标本 prose 提取完整性检查", () => {
  const entries = Object.entries(proseModules);

  it("扫描到足够多的 prose.ts 模块(每个金标本节点一个,dit-vit 复用 dit 除外)", () => {
    expect(entries.length).toBeGreaterThan(60);
  });

  it.each(entries)("%s 提取出的所有 prose 段落均非空", (path, mod) => {
    const extractProse = mod.extractProse as ((markdown: string) => Record<string, string>) | undefined;
    expect(extractProse, `${path} 未 export extractProse`).toBeTypeOf("function");

    const sourcePath = findSourcePath(mod);
    expect(sourcePath, `${path} 未找到 XXX_SOURCE_PATH 导出`).toBeDefined();

    const markdown = findMarkdown(sourcePath!);
    expect(markdown, `找不到 ${sourcePath} 对应的 markdown 文件`).toBeDefined();

    const sections = extractProse!(markdown!);
    const emptyKeys = Object.entries(sections)
      .filter(([, value]) => !value || !value.trim())
      .map(([key]) => key);

    expect(emptyKeys, `${path}(${sourcePath})以下段落提取为空: ${emptyKeys.join(", ")}`).toEqual([]);
  });
});
