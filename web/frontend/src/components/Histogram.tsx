import styles from "./Histogram.module.css";

export function Histogram({ data }: { data: { bucket: string; count: number }[] }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  return (
    <div className={styles.chart}>
      {data.map((d) => (
        <div key={d.bucket} className={styles.col} title={`${d.bucket} D: ${d.count}`}>
          <div className={styles.barTrack}>
            <div className={styles.bar} style={{ height: `${(d.count / max) * 100}%` }} />
          </div>
          <div className={styles.tick}>{d.bucket.split(" to ")[0]}</div>
        </div>
      ))}
    </div>
  );
}
