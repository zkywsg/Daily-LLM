// 轻量 runtime 校验 families.json 形状(无新依赖).
// 不通过的字段在 dev 抛 console.error + 阻止启动;prod build 也会立刻打印,定位 generator 输出漂移。

import type { FamiliesData, FamilyData, NodeData } from "../types/family";
import rawJson from "./families.json";

function isString(x: unknown): x is string {
  return typeof x === "string";
}
function isNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}
function isArray<T>(x: unknown, item: (e: unknown) => e is T): x is T[] {
  return Array.isArray(x) && x.every(item);
}

function assertNode(n: unknown, ctx: string): asserts n is NodeData {
  if (!n || typeof n !== "object") {
    throw new Error(`${ctx}: node 不是 object`);
  }
  const o = n as Record<string, unknown>;
  for (const key of ["name", "family", "paper", "key_idea", "path"] as const) {
    if (!isString(o[key])) throw new Error(`${ctx}.${key} 不是 string`);
  }
  if (!isNumber(o.year)) throw new Error(`${ctx}.year 不是 number`);
  if (!isNumber(o.order)) throw new Error(`${ctx}.order 不是 number`);
  if (!isArray(o.authors, isString))
    throw new Error(`${ctx}.authors 不是 string[]`);
  if (!isArray(o.assets, isString))
    throw new Error(`${ctx}.assets 不是 string[]`);
}

function assertFamily(f: unknown, idx: number): asserts f is FamilyData {
  const ctx = `families[${idx}]`;
  if (!f || typeof f !== "object") throw new Error(`${ctx} 不是 object`);
  const o = f as Record<string, unknown>;
  for (const key of ["id", "label", "blurb", "colorToken"] as const) {
    if (!isString(o[key])) throw new Error(`${ctx}.${key} 不是 string`);
  }
  if (o.yearRange !== null && !isArray(o.yearRange, isNumber)) {
    throw new Error(`${ctx}.yearRange 应是 [number, number] | null`);
  }
  if (!Array.isArray(o.nodes)) throw new Error(`${ctx}.nodes 不是 array`);
  o.nodes.forEach((n, i) => assertNode(n, `${ctx}.nodes[${i}]`));
}

function assertFamiliesData(d: unknown): asserts d is FamiliesData {
  if (!d || typeof d !== "object") throw new Error("families.json 根不是 object");
  const o = d as Record<string, unknown>;
  if (!isString(o.generatedAt))
    throw new Error("families.json.generatedAt 不是 string");
  if (!Array.isArray(o.families))
    throw new Error("families.json.families 不是 array");
  o.families.forEach((f, i) => assertFamily(f, i));
}

try {
  assertFamiliesData(rawJson);
} catch (e) {
  // 立刻报到 console,且重抛阻止模块导出畸形数据
  // eslint-disable-next-line no-console
  console.error("[families.json schema error]", e);
  throw e;
}

export const familiesData: FamiliesData = rawJson as FamiliesData;
