import ReactMarkdown from "react-markdown";
import { Link } from "react-router";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github.css";
import { MermaidBlock } from "./MermaidBlock";
import { resolveMarkdownLink, resolveRepoPath } from "../../lib/markdownPaths";

// 家族目录下的静态资源(图片)在构建期收集,markdown 里的相对 src 据此解析
const assetUrls = import.meta.glob("../../../../[0-9][0-9]-*/assets/*", {
  query: "?url",
  import: "default",
  eager: true,
}) as Record<string, string>;

function assetUrlFor(src: string, sourcePath: string): string | null {
  const repoPath = resolveRepoPath(src, sourcePath);
  return assetUrls[`../../../../${repoPath}`] ?? null;
}

interface MarkdownRendererProps {
  markdown: string;
  /** markdown 的 repo-root-relative 路径,如 "01-cnn/05-resnet.md"。提供后相对链接/图片会被解析 */
  sourcePath?: string;
}

export function MarkdownRenderer({ markdown, sourcePath }: MarkdownRendererProps) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
        components={{
          p({ node, children, ...props }) {
            // 用 hast node 的子节点 tagName(不是 React element 的 type,
            // 后者经过 components 映射后是函数而非 "img"/"em" 字符串)
            const childTags = (
              (node as { children?: Array<{ type: string; tagName?: string }> })
                ?.children ?? []
            )
              .filter((c) => c.type === "element")
              .map((c) => c.tagName);
            // 段内含 img → 拆 p。原因:img handler 把 img 改写成 <figure>,
            // 而 <figure> 不允许出现在 <p> 里;HTML 解析器会自动关 p,造成 em caption 孤立
            if (childTags.includes("img")) {
              return <>{children}</>;
            }
            // 单一 em 节点 → figure caption(`*图 N: ...*` 老式写法,独立成段时)
            const isCaption =
              childTags.length === 1 && childTags[0] === "em";
            return (
              <p className={isCaption ? "figureCaption" : undefined} {...props}>
                {children}
              </p>
            );
          },
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const lang = match?.[1];
            const codeStr = String(children).replace(/\n$/, "");
            if (lang === "mermaid") {
              return <MermaidBlock code={codeStr} />;
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          a({ href, children, ...props }) {
            if (!href || !sourcePath) {
              return (
                <a href={href} {...props}>
                  {children}
                </a>
              );
            }
            const resolved = resolveMarkdownLink(href, sourcePath);
            if (resolved.kind === "internal") {
              return (
                <Link to={resolved.to} {...props}>
                  {children}
                </Link>
              );
            }
            const external = resolved.kind === "external";
            return (
              <a
                href={resolved.href}
                {...(external
                  ? { target: "_blank", rel: "noreferrer" }
                  : {})}
                {...props}
              >
                {children}
              </a>
            );
          },
          img({ src, alt, ...props }) {
            const original = typeof src === "string" ? src : "";
            const resolved =
              sourcePath && original && !/^(https?:)?\/\//.test(original)
                ? assetUrlFor(original, sourcePath)
                : null;
            // 把 alt 作为 figcaption 显示 —— 但跳过两类情况:
            //   1. alt 空(占位图)
            //   2. alt 以 "图 N" 开头(老式样本通常紧跟一个 *图 N: ...* 的 em 段落作详注,避免重复)
            const showCaption =
              !!alt && alt.trim().length > 0 && !/^图\s*\d+/.test(alt.trim());
            return (
              <figure className="markdownFigure">
                <img
                  src={resolved ?? original}
                  alt={alt}
                  style={{ maxWidth: "100%" }}
                  {...props}
                />
                {showCaption && <figcaption>{alt}</figcaption>}
              </figure>
            );
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
