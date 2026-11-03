import styles from "./ConfidenceGauge.module.css";

function qualify(score: number): string {
  if (score >= 75) return "plausible crescent structure";
  if (score >= 40) return "some structure detected";
  if (score >= 15) return "far from the training distribution";
  return "not recognizable as a photorefraction crescent";
}

function colorFor(score: number): string {
  if (score >= 60) return "var(--accent)";
  if (score >= 25) return "var(--accent-2)";
  return "var(--danger)";
}

export function ConfidenceGauge({
  score,
  detail,
}: {
  score: number;
  detail: Record<string, number>;
}) {
  const color = colorFor(score);
  return (
    <div>
      <div className={styles.row}>
        <span className="label-caps">CONFIDENCE (EXPERIMENTAL)</span>
        <span className="readout" style={{ color, fontSize: 18 }}>
          {score.toFixed(0)}%
        </span>
      </div>
      <div className={styles.track}>
        <div className={styles.fill} style={{ width: `${score}%`, background: color }} />
      </div>
      <div className={styles.note}>{qualify(score)}</div>
      <details className={styles.details}>
        <summary className="label-caps" style={{ cursor: "pointer" }}>
          why? z-scores vs. training distribution
        </summary>
        <ul className={styles.zlist}>
          {Object.entries(detail).map(([k, v]) => (
            <li key={k}>
              <span className="mono">{k}</span>
              <span className="readout">
                {v > 0 ? "+" : ""}
                {v.toFixed(2)}σ
              </span>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
