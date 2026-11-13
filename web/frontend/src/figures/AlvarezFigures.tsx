import { Suspense, lazy, useMemo, useState } from "react";
import { useDarkMode } from "../lib/useDarkMode";
import { LineChart } from "../components/LineChart";
import { Readout, Segmented } from "../components/Article";
import { Slider } from "../components/Slider";
import { ACT, AMPS_PER_DIOPTRE, GAIN_D_PER_MM, displacementMm, simulateAxis } from "../lib/alvarez";
import styles from "./figures.module.css";
import local from "./BenchFigures.module.css";

const AlvarezScene = lazy(() => import("../three/AlvarezScene").then((m) => ({ default: m.AlvarezScene })));

const signed = (v: number, digits = 2) => (v > 0 ? "+" : "") + v.toFixed(digits).replace("-", "−");

/** 3-D plate pair: commanded power → per-plate shift → ray bundle. */
export function AlvarezLensFigure() {
  const [kind, setKind] = useState<"sphere" | "astig">("sphere");
  const [M, setM] = useState(-3);
  const [J0, setJ0] = useState(0.75);
  const [J45, setJ45] = useState(-0.5);
  const dark = useDarkMode();

  const sphere = kind === "sphere";
  const dx = displacementMm(sphere ? M : J0);
  const dy = sphere ? 0 : displacementMm(J45);

  return (
    <div className={styles.split}>
      <div className={styles.controls}>
        <div className={styles.controlsGroup}>
          <Segmented
            label="Plate pair"
            options={[{ value: "sphere", label: "Spherical" }, { value: "astig", label: "Astigmatic" }]}
            value={kind}
            onChange={setKind}
          />
        </div>
        <div className={styles.controlsGroup}>
          <div className={styles.groupTitle}>Commanded lens power</div>
          {sphere ? (
            <Slider label="M (slide along x)" value={M} min={-6} max={6} step={0.25} unit=" D" onChange={setM} />
          ) : (
            <>
              <Slider label="J₀ (slide along x)" value={J0} min={-1.5} max={1.5} step={0.25} unit=" D" onChange={setJ0} />
              <Slider label="J₄₅ (slide along y)" value={J45} min={-1.5} max={1.5} step={0.25} unit=" D" onChange={setJ45} accent="amber" />
            </>
          )}
        </div>
        <div className={styles.readouts}>
          <Readout label="Per-plate shift, x" value={signed(dx, 3)} unit="mm" tone="blue" />
          {!sphere && <Readout label="Per-plate shift, y" value={signed(dy, 3)} unit="mm" tone="blue" />}
          <Readout label="Gain" value={GAIN_D_PER_MM.toFixed(1)} unit="D/mm" />
          <Readout label="0.25 D step" value={(displacementMm(0.25) * 1000).toFixed(1)} unit="µm" />
        </div>
      </div>
      <div>
        <div className={local.stage}>
          <Suspense fallback={<div className={local.loading}>Loading 3D lens…</div>}>
            <AlvarezScene
              kind={kind}
              dx={dx}
              dy={dy}
              M={sphere ? M : 0}
              J0={sphere ? 0 : J0}
              J45={sphere ? 0 : J45}
              dark={dark}
            />
          </Suspense>
          <div className={local.hint}>Interactive view: orbit, zoom and pan</div>
        </div>
      </div>
    </div>
  );
}

/** Closed-loop step response of one plate axis under the illustrative voice-coil model. */
export function ActuatorFigure() {
  const [from, setFrom] = useState(0);
  const [to, setTo] = useState(-3);

  const x0 = displacementMm(from) / 1000;
  const x1 = displacementMm(to) / 1000;
  const trace = useMemo(() => simulateAxis(x0, x1), [x0, x1]);

  // Settled once the plate stays within half a clinical step (0.125 D) of the target.
  const bandUm = displacementMm(0.125) * 1000;
  const targetUm = x1 * 1e6;
  let last = -1;
  trace.x.forEach((x, i) => { if (Math.abs(x - targetUm) > bandUm) last = i; });
  const settled = last < trace.t.length - 1;
  const settleMs = last < 0 ? 0 : Math.max(0, trace.t[last + 1] - 2);

  const holdA = Math.abs(to) * AMPS_PER_DIOPTRE;
  const peakMa = Math.max(...trace.i.map(Math.abs));

  return (
    <div>
      <div className={styles.toolbar}>
        <div style={{ flex: "1 1 220px" }}>
          <Slider label="From" value={from} min={-6} max={6} step={0.25} unit=" D" onChange={setFrom} />
        </div>
        <div style={{ flex: "1 1 220px" }}>
          <Slider label="To" value={to} min={-6} max={6} step={0.25} unit=" D" onChange={setTo} accent="amber" />
        </div>
      </div>
      <div className={styles.duo}>
        <LineChart
          title="Plate position"
          series={[{ name: "Position", color: "var(--blue)", points: trace.t.map((t, i) => [t, trace.x[i]]) }]}
          xLabel="time (ms)"
          yLabel="position (µm)"
          refLines={[{ y: targetUm, label: "target" }]}
          formatX={(v) => v.toFixed(1)}
          formatY={(v) => v.toFixed(0)}
          hideLegend
        />
        <LineChart
          title="Coil current"
          series={[{ name: "Current", color: "var(--red)", points: trace.t.map((t, i) => [t, trace.i[i]]) }]}
          xLabel="time (ms)"
          yLabel="current (mA)"
          formatX={(v) => v.toFixed(1)}
          formatY={(v) => v.toFixed(0)}
          hideLegend
        />
      </div>
      <div className={styles.readouts}>
        <Readout label="Plate travel" value={signed((x1 - x0) * 1000, 3)} unit="mm" tone="blue" />
        <Readout label="Settles within ±0.125 D" value={settled ? settleMs.toFixed(1) : "> 28"} unit="ms" />
        <Readout label="Peak current" value={peakMa.toFixed(0)} unit="mA" />
        <Readout label="Holding current" value={(holdA * 1000).toFixed(0)} unit="mA" tone="red" />
        <Readout label="Holding power" value={(holdA * holdA * ACT.coilOhm * 1000).toFixed(0)} unit="mW" tone="red" />
      </div>
    </div>
  );
}
