import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import { fileURLToPath, URL } from "node:url";

// 仓库根目录（web/ 的上一级）
// 需要允许 vite 读 ../tracks/, ../foundations/ 下的 markdown
const REPO_ROOT = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      // 默认 Vite 限制只能读取项目根下的文件，扩到仓库根，
      // 这样 TrackView 才能 `import cnnMd from "../../../tracks/.../README.md?raw"`
      allow: [REPO_ROOT],
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    server: {
      // vitest 也走同一套 fs allow
      deps: {
        inline: ["react-markdown"],
      },
    },
  },
});
