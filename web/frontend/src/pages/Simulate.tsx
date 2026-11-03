import { useState } from "react";
import { HudPanel } from "../components/HudPanel";
import { Slider } from "../components/Slider";
import { Dropzone } from "../components/Dropzone";
import { ResultViewer } from "../components/ResultViewer";
import { postSimulate, type SimulateResponse } from "../lib/api";
import styles from "./Simulate.module.css";

const PRESETS: { label: string; S: number; C: number; axis: number }[] = [
  { label: "20/20, no error", S: 0, C: 0, axis: 0 },
  { label: "Mild myopia", S: -1.5, C: 0, axis: 0 },
  { label: "Myopia + astigmatism", S: -2.25, C: -0.75, axis: 30 },
  { label: "High myopia", S: -5.5, C: 0, axis: 0 },
  { label: "Hyperopia", S: 2.0, C: 0, axis: 0 },
];

export function Simulate() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [S, setS] = useState(-2.25);
  const [C, setC] = useState(-0.75);
  const [axis, setAxis] = useState(30);
  const [result, setResult] = useState<SimulateResponse | null>(null);
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
      setResult(await postSimulate(file, S, C, axis));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.grid}>
      <div className={styles.sidebar}>
        <HudPanel title="Source image">
          <Dropzone onFile={handleFile} />
          {previewUrl && <img src={previewUrl} alt="upload preview" className={styles.thumb} />}
        </HudPanel>

        <HudPanel title="Prescription" subtitle="pick a preset or dial in">
          <div className={styles.presets}>
            {PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => {
                  setS(p.S);
                  setC(p.C);
                  setAxis(p.axis);
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
          <Slider label="Sphere S" value={S} min={-8} max={6} step={0.25} unit=" D" onChange={setS} />
          <Slider label="Cylinder C" value={C} min={-4} max={0} step={0.25} unit=" D" onChange={setC} />
          <Slider label="Axis" value={axis} min={0} max={180} step={5} unit="°" onChange={setAxis} accent="amber" />
          <button
            className="primary"
            disabled={!file || loading}
            onClick={handleRun}
            style={{ width: "100%", marginTop: 8 }}
          >
            {loading ? "Rendering…" : "Run simulation"}
          </button>
          {error && <div style={{ color: "var(--danger)", fontSize: 12, marginTop: 8 }}>{error}</div>}
        </HudPanel>
      </div>

      <HudPanel
        title="Result"
        subtitle={
          result
            ? `M ${result.case.M.toFixed(2)}  J0 ${result.case.J0.toFixed(2)}  J45 ${result.case.J45.toFixed(2)}`
            : undefined
        }
        accent
      >
        {!result && (
          <p style={{ color: "var(--fg-dim)", fontSize: 13 }}>
            Upload a photo, choose a prescription, and run the simulation to
            see exactly what that eye sees, and the pre-distorted version it
            would perceive as sharp.
          </p>
        )}
        {result && <ResultViewer case={result.case} shaderKernel={result.shader_kernel} />}
      </HudPanel>
    </div>
  );
}
