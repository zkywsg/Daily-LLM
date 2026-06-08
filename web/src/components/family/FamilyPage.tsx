import { useParams, Link, Navigate } from "react-router";
import type { CSSProperties } from "react";
import type { FamiliesData, FamilyId } from "../../types/family";
import familiesJson from "../../data/families.json";
import { familyColorVar } from "../../lib/colors";
import { getMiniArch } from "../mini-arches/getMiniArch";
import styles from "./FamilyPage.module.css";

const data = familiesJson as unknown as FamiliesData;

export function FamilyPage() {
  const { familyId } = useParams<{ familyId: FamilyId }>();
  const family = data.families.find((f) => f.id === familyId);
  if (!family) {
    return <Navigate to="/404" replace />;
  }
  const accent = familyColorVar(family.id);

  return (
    <div className={styles.container}>
      <Link to="/" className={styles.back}>
        ← 返回主页
      </Link>
      <div className={styles.familyId}>{family.id}</div>
      <h1 className={styles.title} style={{ color: accent }}>
        {family.label}
      </h1>
      <p
        className={styles.blurb}
        style={{ ["--accent" as string]: accent } as CSSProperties}
      >
        {family.blurb}
      </p>

      {family.nodes.length === 0 ? (
        <p style={{ color: "var(--ink-muted)" }}>本家族内容待补充。</p>
      ) : (
        <>
          <h2 style={{ fontSize: "var(--fs-xl)" }}>子时间线</h2>
          <div className={styles.subtimeline}>
            {family.nodes.map((n) => {
              const slug = n.path.split("/").pop()!.replace(/\.md$/, "");
              const MiniArch = getMiniArch(n.path);
              return (
                <Link
                  key={n.path}
                  to={`/families/${family.id}/${slug}`}
                  className={styles.nodeCard}
                  style={{ borderTopColor: accent, borderTopWidth: 3 }}
                >
                  <div className={styles.nodeYear}>{n.year}</div>
                  <div className={styles.nodeName}>{n.name}</div>
                  {MiniArch && (
                    <div style={{ margin: "var(--space-2) 0" }}>
                      <MiniArch width={160} height={50} />
                    </div>
                  )}
                  <p className={styles.nodeIdea}>{n.key_idea}</p>
                </Link>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
