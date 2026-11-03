import { useEffect, useMemo, useState, type ReactNode } from "react";
import katex from "katex";
import styles from "./Article.module.css";

export function PageHeader({
  kicker, title, lede, children,
}: { kicker: string; title: ReactNode; lede: ReactNode; children?: ReactNode }) {
  return (
    <header className={styles.pageHeader}>
      <div className={styles.kicker}>{kicker}</div>
      <h1 className={styles.title}>{title}</h1>
      <p className={styles.lede}>{lede}</p>
      {children}
    </header>
  );
}

export type TocEntry = { id: string; num: string; title: string };

/** Two-column article: sticky table of contents with scroll-spy + body. */
export function Article({ toc, children }: { toc: TocEntry[]; children: ReactNode }) {
  const [active, setActive] = useState(toc[0]?.id);
  useEffect(() => {
    const els = toc.map((t) => document.getElementById(t.id)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (vis[0]) setActive(vis[0].target.id);
      },
      { rootMargin: "-80px 0px -65% 0px" },
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, [toc]);

  return (
    <div className={styles.articleGrid}>
      <nav className={styles.toc} aria-label="Contents">
        <div className={styles.tocLabel}>Contents</div>
        <ol>
          {toc.map((t) => (
            <li key={t.id}>
              <a href={`#${t.id}`} className={active === t.id ? styles.tocActive : undefined}>
                <span className={styles.tocNum}>{t.num}</span>
                {t.title}
              </a>
            </li>
          ))}
        </ol>
      </nav>
      <div className={styles.body}>{children}</div>
    </div>
  );
}

export function Section({
  id, num, title, children,
}: { id: string; num: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className={styles.section}>
      <h2 className={styles.h2}>
        <span className={styles.secNum}>{num}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

export function Figure({
  num, caption, children, wide = true,
}: { num: number | string; caption: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <figure className={`${styles.figure} ${wide ? styles.wide : ""}`}>
      <div className={styles.figBody}>{children}</div>
      <figcaption className={styles.caption}>
        <span className={styles.figNum}>Figure {num}.</span> {caption}
      </figcaption>
    </figure>
  );
}

export function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: block, throwOnError: false }),
    [children, block],
  );
  return block
    ? <div className={styles.eq} dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Note({ label = "Note", children }: { label?: string; children: ReactNode }) {
  return (
    <aside className={styles.note}>
      <div className={styles.noteLabel}>{label}</div>
      <div>{children}</div>
    </aside>
  );
}

export function Stat({ value, unit, label }: { value: string; unit?: string; label: ReactNode }) {
  return (
    <div className={styles.stat}>
      <div className={styles.statValue}>
        {value}
        {unit && <span className={styles.statUnit}>{unit}</span>}
      </div>
      <div className={styles.statLabel}>{label}</div>
    </div>
  );
}

export function StatRow({ children }: { children: ReactNode }) {
  return <div className={styles.statRow}>{children}</div>;
}

/** Labelled numeric read-out used beside figures. */
export function Readout({
  label, value, unit, tone,
}: { label: ReactNode; value: string; unit?: string; tone?: "blue" | "red" | "green" }) {
  return (
    <div className={styles.readout}>
      <span className={styles.readoutLabel}>{label}</span>
      <span className={`${styles.readoutValue} ${tone ? styles[tone] : ""}`}>
        {value}
        {unit && <span className={styles.readoutUnit}> {unit}</span>}
      </span>
    </div>
  );
}

export function Panel({ title, children, className }: { title?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`${styles.panel} ${className ?? ""}`}>
      {title && <div className={styles.panelTitle}>{title}</div>}
      {children}
    </div>
  );
}

export function Segmented<T extends string | number>({
  options, value, onChange, label,
}: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void; label?: string }) {
  return (
    <div className={styles.segWrap}>
      {label && <span className={styles.segLabel}>{label}</span>}
      <div className={styles.segmented} role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button
            key={String(o.value)}
            role="radio"
            aria-checked={o.value === value}
            className={o.value === value ? styles.segOn : undefined}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export const articleStyles = styles;
