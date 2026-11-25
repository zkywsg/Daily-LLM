import { Link } from "react-router";

export function NotFoundPage() {
  return (
    <div style={{ padding: "var(--space-16)", textAlign: "center" }}>
      <h1 style={{ fontSize: "var(--fs-4xl)", marginBottom: "var(--space-4)" }}>
        404
      </h1>
      <p
        style={{
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        页面不存在
      </p>
      <Link to="/">返回主页</Link>
    </div>
  );
}
