import type { FamilyId } from "../types/family";

/** 把家族 id 映射到 CSS 变量引用，用于 inline style */
export function familyColorVar(id: FamilyId): string {
  const num = id.split("-")[0];
  return `var(--family-${num})`;
}

/** 取得家族 colorToken 的 CSS 变量名（不含 var() 包装） */
export function familyColorToken(id: FamilyId): string {
  const num = id.split("-")[0];
  return `--family-${num}`;
}
