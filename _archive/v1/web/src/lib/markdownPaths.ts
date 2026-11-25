// 把 markdown 正文里的仓库相对路径(链接/图片)解析成 app 路由或外部 URL。
// sourcePath 形如 "01-cnn/05-resnet.md"(repo-root-relative)。

const GITHUB_BASE = "https://github.com/zkywsg/Daily-LLM";

export type ResolvedLink =
  | { kind: "internal"; to: string }
  | { kind: "external"; href: string }
  | { kind: "passthrough"; href: string };

/** 以 sourcePath 所在目录为基准,把相对路径规范化为 repo-root-relative 路径 */
export function resolveRepoPath(href: string, sourcePath: string): string {
  const baseDir = sourcePath.split("/").slice(0, -1);
  const segments = [...baseDir];
  for (const part of href.split("/")) {
    if (part === "" || part === ".") continue;
    if (part === "..") segments.pop();
    else segments.push(part);
  }
  return segments.join("/");
}

export function resolveMarkdownLink(
  href: string,
  sourcePath: string
): ResolvedLink {
  if (/^(https?:|mailto:|#)/.test(href) || href.startsWith("/")) {
    return { kind: "passthrough", href };
  }
  const repoPath = resolveRepoPath(href, sourcePath);

  // 家族节点: "01-cnn/03-vgg.md" → /families/01-cnn/03-vgg
  const nodeMatch = repoPath.match(/^(\d{2}-[^/]+)\/([^/]+)\.md$/);
  if (nodeMatch) {
    return { kind: "internal", to: `/families/${nodeMatch[1]}/${nodeMatch[2]}` };
  }
  // 家族目录: "08-vit" → /families/08-vit
  const familyMatch = repoPath.match(/^(\d{2}-[^/]+)$/);
  if (familyMatch) {
    return { kind: "internal", to: `/families/${familyMatch[1]}` };
  }
  // 其余(foundations/、projects/ 等站内没有路由的)→ 指向 GitHub
  const isFile = /\.[a-z0-9]+$/i.test(repoPath);
  return {
    kind: "external",
    href: `${GITHUB_BASE}/${isFile ? "blob" : "tree"}/master/${repoPath}`,
  };
}
