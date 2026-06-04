/// <reference types="vite/client" />

// Vite ?raw import: 把任意文件作为字符串内联
declare module "*?raw" {
  const content: string;
  export default content;
}

// Markdown 文件作为字符串导入（配 ?raw 使用）
declare module "*.md?raw" {
  const content: string;
  export default content;
}
