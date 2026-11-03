import { useEffect, useState } from "react";
import { HudPanel } from "../components/HudPanel";
import { Histogram } from "../components/Histogram";
import { GalleryCard } from "../components/GalleryCard";
import { getGallery, getStats, type CaseOut, type StatsResponse } from "../lib/api";
import styles from "./Gallery.module.css";

const LIMIT = 24;
type ModeFilter = "all" | "simulate" | "diagnose";

export function Gallery() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [items, setItems] = useState<CaseOut[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [mode, setMode] = useState<ModeFilter>("all");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    getGallery({ mode: mode === "all" ? undefined : mode, limit: LIMIT, offset })
      .then((page) => {
        setItems(page.items);
        setTotal(page.total);
      })
      .finally(() => setLoading(false));
  }, [mode, offset]);

  return (
    <div className={styles.wrap}>
      <div className={styles.statsRow}>
        <HudPanel title="Cases">
          <div className="readout" style={{ fontSize: 28, fontWeight: 700 }}>
            {stats?.total_cases ?? "..."}
          </div>
          <div className="label-caps" style={{ marginTop: 4 }}>
            {stats?.simulate_count ?? 0} simulate, {stats?.diagnose_count ?? 0} diagnose
          </div>
        </HudPanel>
        <HudPanel title="Avg. diagnose confidence">
          <div className="readout" style={{ fontSize: 28, fontWeight: 700 }}>
            {stats?.avg_diagnose_confidence != null
              ? `${stats.avg_diagnose_confidence.toFixed(0)}%`
              : "..."}
          </div>
        </HudPanel>
        <HudPanel title="M distribution" subtitle="spherical equivalent, dioptres" accent>
          {stats && <Histogram data={stats.m_histogram} />}
        </HudPanel>
      </div>

      <HudPanel title="Public cases" subtitle={`${total} total`}>
        <div className={styles.filters}>
          {(["all", "simulate", "diagnose"] as const).map((m) => (
            <button
              key={m}
              className={mode === m ? "primary" : ""}
              onClick={() => {
                setMode(m);
                setOffset(0);
              }}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>

        {loading ? (
          <div className="label-caps">loading…</div>
        ) : items.length === 0 ? (
          <div className="label-caps">
            no public cases yet. Run a Simulate or Diagnose to add one
          </div>
        ) : (
          <div className={styles.grid}>
            {items.map((c) => (
              <GalleryCard key={c.id} c={c} />
            ))}
          </div>
        )}

        {total > LIMIT && (
          <div className={styles.pager}>
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>
              ◀ PREV
            </button>
            <span className="label-caps">
              {offset + 1} to {Math.min(offset + LIMIT, total)} of {total}
            </span>
            <button disabled={offset + LIMIT >= total} onClick={() => setOffset(offset + LIMIT)}>
              NEXT ▶
            </button>
          </div>
        )}
      </HudPanel>
    </div>
  );
}
