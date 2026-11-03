import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { useDarkMode } from "../lib/useDarkMode";
import { FieldCanvas } from "../components/FieldCanvas";
import { LineChart } from "../components/LineChart";
import { Readout, Segmented } from "../components/Article";
import { Slider } from "../components/Slider";
import { DEFAULT_GEOMETRY, renderPhotorefraction, toPowerVector } from "../lib/optics";
import { gray } from "../lib/images";
import styles from "./figures.module.css";
import local from "./BenchFigures.module.css";

const BenchScene = lazy(() => import("../three/BenchScene").then((m) => ({ default: m.BenchScene })));

/** Screen offset (mm) that realises sphere S with a lens of focal length f0 (mm): S = 1/(f0+Δx) − 1/f0. */
export function screenOffsetMm(S: number, f0mm: number) {
  const f = f0mm / 1000;
  return (1 / (S + 1 / f) - f) * 1000;
}

// Example drive train (proposal §4B): T8 lead screw (8 mm/rev), 200 steps × 16 microsteps.
const UM_PER_USTEP = 8000 / (200 * 16);
// Rotation stage: GT2 belt, 20T → 120T (6:1), 200 × 16 microsteps.
const DEG_PER_USTEP = 360 / (200 * 16 * 6);

export function BenchFigure() {
  const [S, setS] = useState(-3);
  const [C, setC] = useState(-1);
  const [axis, setAxis] = useState(30);
  const [pupil, setPupil] = useState(3);
  const [f0, setF0] = useState(50);
  const [labels, setLabels] = useState(true);
  const [capturing, setCapturing] = useState(false);
  const [firing, setFiring] = useState<"h" | "v" | null>(null);
  const dark = useDarkMode();

  useEffect(() => {
    if (!capturing) { setFiring(null); return; }
    let i = 0;
    const seq: ("h" | "v" | null)[] = ["h", null, "v", null];
    setFiring(seq[0]);
    const id = setInterval(() => { i = (i + 1) % seq.length; setFiring(seq[i]); }, 450);
    return () => clearInterval(id);
  }, [capturing]);

  const dx = screenOffsetMm(S, f0);
  const pv = toPowerVector(S, C, axis);
  const g = { ...DEFAULT_GEOMETRY, Dmm: 300, pupilRmm: pupil };
  const frameH = useMemo(() => renderPhotorefraction(pv, 10, 0, g, 72), [S, C, axis, pupil]); // eslint-disable-line react-hooks/exhaustive-deps
  const frameV = useMemo(() => renderPhotorefraction(pv, 0, 10, g, 72), [S, C, axis, pupil]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className={styles.split}>
      <div className={styles.controls}>
        <div className={styles.controlsGroup}>
          <div className={styles.groupTitle}>Commanded prescription</div>
          <Slider label="Sphere (screen travel)" value={S} min={-6} max={4} step={0.25} unit=" D" onChange={setS} />
          <Slider label="Cylinder trial lens" value={C} min={-3} max={0} step={0.25} unit=" D" onChange={setC} />
          <Slider label="Axis (rotation stage)" value={axis} min={0} max={180} step={5} unit="°" digits={0} signed={false} onChange={setAxis} accent="amber" />
          <Slider label="Iris radius" value={pupil} min={1.5} max={4} step={0.25} unit=" mm" signed={false} onChange={setPupil} />
        </div>
        <div className={styles.controlsGroup}>
          <Segmented
            label="Cornea lens focal length"
            options={[{ value: 40, label: "40 mm" }, { value: 50, label: "50 mm" }, { value: 75, label: "75 mm" }]}
            value={f0}
            onChange={setF0}
          />
          <div style={{ display: "flex", gap: 14, marginTop: 14, flexWrap: "wrap" }}>
            <label className={styles.check}><input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} />Labels</label>
            <label className={styles.check}><input type="checkbox" checked={capturing} onChange={(e) => setCapturing(e.target.checked)} />Fire LEDs</label>
          </div>
        </div>
        <div className={styles.readouts}>
          <Readout label="Screen offset Δx" value={(dx >= 0 ? "+" : "") + dx.toFixed(2)} unit="mm" tone="blue" />
          <Readout label="Lead-screw µsteps" value={Math.round((dx * 1000) / UM_PER_USTEP).toLocaleString()} />
          <Readout label="Rotation µsteps" value={Math.round(axis / DEG_PER_USTEP).toLocaleString()} />
          <Readout label="Label (M, J₀, J₄₅)" value={`${pv.M.toFixed(2)}, ${pv.J0.toFixed(2)}, ${pv.J45.toFixed(2)}`} />
        </div>
      </div>
      <div>
        <div className={local.stage}>
          <Suspense fallback={<div className={local.loading}>Loading 3D bench…</div>}>
            <BenchScene f0mm={f0} dxMm={dx} axisDeg={axis} hasCyl={C !== 0} pupilMm={pupil} firing={firing} labels={labels} dark={dark} />
          </Suspense>
          <div className={local.hint}>Drag to orbit, scroll to zoom, right-drag to pan</div>
          <div className={local.frames}>
            <div className={local.frame}>
              <div className={`${styles.canvasFrame} ${styles.canvasDark}`} style={{ outline: firing === "h" ? "2px solid var(--red)" : undefined }}>
                <FieldCanvas data={frameH} n={72} map={gray} label="Expected camera frame, horizontal LED" smooth />
              </div>
              <span>LED 0°</span>
            </div>
            <div className={local.frame}>
              <div className={`${styles.canvasFrame} ${styles.canvasDark}`} style={{ outline: firing === "v" ? "2px solid var(--red)" : undefined }}>
                <FieldCanvas data={frameV} n={72} map={gray} label="Expected camera frame, vertical LED" smooth />
              </div>
              <span>LED 90°</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** 2-D vergence diagram + travel-per-dioptre curves for different cornea focal lengths. */
export function VergenceFigure() {
  const [S, setS] = useState(-4);
  const [f0, setF0] = useState(50);
  const dx = screenOffsetMm(S, f0);
  const linear = -S * (f0 / 1000) ** 2 * 1000;

  // SVG geometry (mm → px)
  const W = 640, H = 200, lensX = 70, ay = H / 2;
  const scale = 520 / (f0 + screenOffsetMm(-6, f0)); // px per mm: the longest travel fills the drawing
  const bar = [1, 2, 5, 10, 20].reduce((b, v) => (Math.abs(v * scale - 90) < Math.abs(b * scale - 90) ? v : b), 1);
  const fx = lensX + f0 * scale;
  const sx = lensX + (f0 + dx) * scale;
  const a = 34; // drawn half-aperture (px)
  const rays = [-1, -0.5, 0.5, 1].map((t) => {
    const y = ay + t * a;
    const yScreen = ay + t * a * (1 - (sx - lensX) / (fx - lensX));
    return { y, yScreen };
  });

  const curves = [17, 40, 50, 75].map((f, i) => ({
    name: `f₀ = ${f} mm${f === 17 ? " (eye scale)" : ""}`,
    color: ["var(--ink-3)", "var(--green)", "var(--blue)", "var(--red)"][i],
    dashed: f === 17,
    points: Array.from({ length: 41 }, (_, k) => {
      const s = -6 + k * 0.25;
      return [s, screenOffsetMm(s, f)] as [number, number];
    }),
  }));

  return (
    <div>
      <div className={styles.toolbar}>
        <div style={{ width: 240 }}>
          <Slider label="Target sphere S" value={S} min={-6} max={4} step={0.25} unit=" D" onChange={setS} />
        </div>
        <Segmented
          options={[17, 40, 50, 75].map((v) => ({ value: v, label: `${v} mm` }))}
          value={f0}
          onChange={setF0}
          label="f₀"
        />
        <div className={styles.spacer} />
        <div style={{ minWidth: 220 }}>
          <Readout label="Exact Δx" value={dx.toFixed(2)} unit="mm" tone="blue" />
          <Readout label="Linear −S·f₀²" value={linear.toFixed(2)} unit="mm" />
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className={styles.svgFig} role="img" aria-label="Ray diagram of the model eye">
        <line x1={10} x2={W - 10} y1={ay} y2={ay} stroke="var(--rule-strong)" strokeDasharray="4 4" />
        {rays.map((r, i) => (
          <polyline key={i} points={`10,${r.y} ${lensX},${r.y} ${sx},${r.yScreen}`} fill="none" stroke="var(--blue)" strokeWidth={1.6} opacity={0.85} />
        ))}
        {/* lens */}
        <path d={`M${lensX},${ay - 52} Q${lensX + 14},${ay} ${lensX},${ay + 52} Q${lensX - 14},${ay} ${lensX},${ay - 52}Z`} fill="var(--blue-soft)" stroke="var(--blue)" />
        {/* focus */}
        {fx < W - 10 && (
          <g>
            <circle cx={fx} cy={ay} r={3.5} fill="var(--ink)" />
            <text x={fx} y={ay + 62} textAnchor="middle">focus (f₀)</text>
            <line x1={fx} x2={fx} y1={ay + 8} y2={ay + 48} stroke="var(--ink-3)" strokeDasharray="2 3" />
          </g>
        )}
        {/* screen */}
        <rect x={sx - 3} y={ay - 60} width={6} height={120} fill="var(--surface-2)" stroke="var(--ink-2)" />
        <line x1={sx - 3} x2={sx - 3} y1={Math.min(rays[0].yScreen, rays[3].yScreen)} y2={Math.max(rays[0].yScreen, rays[3].yScreen)} stroke="var(--red)" strokeWidth={4} />
        <text x={sx} y={ay - 68} textAnchor="middle">screen</text>
        {/* Δx dimension */}
        {Math.abs(sx - fx) > 2 && (
          <g>
            <line x1={fx} x2={sx} y1={ay - 80} y2={ay - 80} stroke="var(--red)" />
            <line x1={fx} x2={fx} y1={ay - 86} y2={ay - 74} stroke="var(--red)" />
            <line x1={sx} x2={sx} y1={ay - 86} y2={ay - 74} stroke="var(--red)" />
            <text x={(fx + sx) / 2} y={ay - 86} textAnchor="middle" style={{ fill: "var(--red)" }}>Δx = {dx.toFixed(1)} mm</text>
          </g>
        )}
        <text x={lensX} y={H - 8} textAnchor="middle">cornea lens</text>
        <text x={12} y={ay - a - 8}>from camera</text>
        {/* scale bar */}
        <line x1={W - 20 - bar * scale} x2={W - 20} y1={H - 12} y2={H - 12} stroke="var(--ink-2)" strokeWidth={2} />
        <text x={W - 20 - (bar * scale) / 2} y={H - 18} textAnchor="middle">{bar} mm</text>
      </svg>
      <div style={{ marginTop: 20 }}>
        <LineChart
          title="Screen travel needed for each target sphere"
          series={curves}
          xLabel="target sphere S (D)"
          yLabel="Δx (mm)"
          formatX={(v) => v.toFixed(0)}
          formatY={(v) => v.toFixed(1)}
          height={240}
        />
      </div>
    </div>
  );
}

const STEPS = [
  { t: "Move", d: "Step the rail to Δx(S) and the rotation stage to θ" },
  { t: "Settle", d: "Wait for mechanical ringing to decay" },
  { t: "LED 0°", d: "Fire horizontal LED, expose frame" },
  { t: "LED 90°", d: "Fire vertical LED, expose frame" },
  { t: "Label", d: "prescription_to_power_vector(S, C, θ)" },
  { t: "Append", d: "images (N,64,64,2), labels (N,3) → bench.npz" },
];

export function CaptureFigure() {
  const [i, setI] = useState(0);
  const [play, setPlay] = useState(true);
  useEffect(() => {
    if (!play) return;
    const id = setInterval(() => setI((k) => (k + 1) % STEPS.length), 900);
    return () => clearInterval(id);
  }, [play]);
  return (
    <div>
      <div className={local.pipeline}>
        {STEPS.map((s, k) => (
          <button
            key={s.t}
            className={`${local.stepCard} ${k === i ? local.stepOn : ""}`}
            onClick={() => { setPlay(false); setI(k); }}
          >
            <span className={local.stepNum}>{k + 1}</span>
            <span className={local.stepTitle}>{s.t}</span>
            <span className={local.stepDesc}>{s.d}</span>
          </button>
        ))}
      </div>
      <div className={styles.toolbar} style={{ marginTop: 14, marginBottom: 0 }}>
        <button onClick={() => setPlay((p) => !p)}>{play ? "Pause" : "Play"}</button>
        <span className={styles.paneFoot} style={{ margin: 0 }}>
          The same schema as <code>generate_dataset.py</code>, so <code>PrescriptionDataset</code> reads bench data with no adapter.
        </span>
      </div>
    </div>
  );
}
