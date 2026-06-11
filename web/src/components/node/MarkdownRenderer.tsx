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
            return (
              <img
                src={resolved ?? original}
                alt={alt}
                style={{ maxWidth: "100%" }}
                {...props}
              />
            );
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
