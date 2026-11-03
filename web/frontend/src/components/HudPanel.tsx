import type { ReactNode } from "react";
import styles from "./HudPanel.module.css";

export function HudPanel({
  title,
  subtitle,
  children,
  accent = false,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  accent?: boolean;
}) {
  return (
    <section className={`${styles.panel} ${accent ? styles.accent : ""}`}>
      <header className={styles.header}>
        <span className={styles.title}>{title}</span>
        {subtitle && <span className={styles.subtitle}>{subtitle}</span>}
      </header>
      <div className={styles.body}>{children}</div>
    </section>
  );
}
