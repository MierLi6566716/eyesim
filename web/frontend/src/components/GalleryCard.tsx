import { Link } from "react-router-dom";
import type { CaseOut } from "../lib/api";
import styles from "./GalleryCard.module.css";

export function GalleryCard({ c }: { c: CaseOut }) {
  return (
    <Link to={`/share/${c.share_slug}`} className={styles.card}>
      <img src={c.blurred_image_url ?? c.source_image_url} alt="" className={styles.thumb} />
      <div className={styles.meta}>
        <span className={`${styles.badge} ${c.mode === "diagnose" ? styles.diag : styles.sim}`}>
          {c.mode}
        </span>
        <span className="readout" style={{ fontSize: 11 }}>
          M {c.M.toFixed(2)}
        </span>
      </div>
    </Link>
  );
}
