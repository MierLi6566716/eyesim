import { useState } from "react";
import { HudPanel } from "../components/HudPanel";
import { Dropzone } from "../components/Dropzone";
import { CameraCapture } from "../components/CameraCapture";
import { ConfidenceGauge } from "../components/ConfidenceGauge";
import { ResultViewer } from "../components/ResultViewer";
import { postDiagnose, type DiagnoseResponse } from "../lib/api";
import styles from "./Diagnose.module.css";

export function Diagnose() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DiagnoseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFile(f: File) {
    setFile(f);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function handleRun() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await postDiagnose(file));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.grid}>
      <div className={styles.sidebar}>
        <HudPanel title="Diagnose" subtitle="experimental" accent>
          <p className={styles.disclaimer}>
            PupilNet was trained only on synthetic eccentric-photorefraction
            images, not real photos. Treat any estimate below as a rough demo,
            not a measurement, and check the confidence score before
            trusting it at all.
          </p>
        </HudPanel>

        <HudPanel title="Capture">
          <CameraCapture onCapture={handleFile} />
          <div className={styles.or}>or</div>
          <Dropzone onFile={handleFile} />
          {previewUrl && <img src={previewUrl} alt="capture preview" className={styles.thumb} />}
          <button
            className="primary"
            disabled={!file || loading}
            onClick={handleRun}
            style={{ width: "100%", marginTop: 12 }}
          >
            {loading ? "Analyzing…" : "Run diagnosis"}
          </button>
          {error && <div style={{ color: "var(--danger)", fontSize: 12, marginTop: 8 }}>{error}</div>}
        </HudPanel>
      </div>

      <HudPanel
        title="Result"
        subtitle={
          result
            ? `S ${result.case.S?.toFixed(2)}  C ${result.case.C?.toFixed(2)}  axis ${result.case.axis?.toFixed(0)}°`
            : undefined
        }
        accent
      >
        {!result && (
          <p style={{ color: "var(--fg-dim)", fontSize: 13 }}>
            Capture or upload a photo to get an experimental PupilNet
            estimate, with a computed confidence score showing how far the
            input sits from the model's synthetic training distribution.
          </p>
        )}
        {result && (
          <>
            <ConfidenceGauge score={result.case.confidence ?? 0} detail={result.confidence_detail} />
            <div style={{ height: 20 }} />
            <ResultViewer case={result.case} shaderKernel={result.shader_kernel} />
          </>
        )}
      </HudPanel>
    </div>
  );
}
