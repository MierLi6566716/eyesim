import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { HudPanel } from "../components/HudPanel";
import { ResultViewer } from "../components/ResultViewer";
import { getCaseBySlug, shareCardUrl, type CaseOut } from "../lib/api";
import styles from "./Share.module.css";

export function Share() {
  const { slug } = useParams<{ slug: string }>();
  const [caseData, setCaseData] = useState<CaseOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!slug) return;
    getCaseBySlug(slug)
      .then(setCaseData)
      .catch(() => setError("This case doesn't exist, or isn't public."));
  }, [slug]);

  if (error) {
    return (
      <HudPanel title="Not found">
        <p style={{ fontSize: 13 }}>
          {error} <Link to="/tools/gallery">Back to the gallery.</Link>
        </p>
      </HudPanel>
    );
  }
  if (!caseData) {
    return (
      <HudPanel title="Loading">
        <span className="label-caps">fetching case…</span>
      </HudPanel>
    );
  }

  const rx =
    caseData.S != null
      ? `S ${caseData.S.toFixed(2)} D   C ${caseData.C?.toFixed(2)} D   AXIS ${caseData.axis?.toFixed(0)}°`
      : `M ${caseData.M.toFixed(2)}   J0 ${caseData.J0.toFixed(2)}   J45 ${caseData.J45.toFixed(2)}`;

  return (
    <div className={styles.wrap}>
      <HudPanel
        title={caseData.mode === "diagnose" ? "DIAGNOSE RESULT" : "SIMULATE RESULT"}
        subtitle={rx}
        accent
      >
        <ResultViewer case={caseData} shaderKernel={[]} />
        {caseData.confidence != null && (
          <div className={styles.confidence}>
            confidence {caseData.confidence.toFixed(0)}% (experimental estimate)
          </div>
        )}
        <a
          href={shareCardUrl(caseData.share_slug)}
          target="_blank"
          rel="noreferrer"
          className={styles.cardLink}
        >
          <button style={{ marginTop: 16 }}>View or download share card</button>
        </a>
      </HudPanel>
    </div>
  );
}
