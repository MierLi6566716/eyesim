import { useEffect, useMemo, useRef, useState } from "react";
import { FieldCanvas } from "../components/FieldCanvas";
import { LineChart } from "../components/LineChart";
import { Readout, Segmented } from "../components/Article";
import {
  closedLoop, convolve, fastPsf, gaussian, mulberry32, pvNorm, toPowerVector,
  type LoopEval, type PowerVector,
} from "../lib/optics";
import { gray, siemensStar } from "../lib/images";
import type { Eye } from "./EyeControls";
import styles from "./figures.module.css";

const N = 112;
let star: Float64Array | null = null;
const target = () => (star ??= siemensStar(N, 20));

/** Test-set MAE of PupilNet (checkpoints/best.pt) — used to draw a realistic CNN estimate. */
export const CNN_MAE = { M: 0.184, J0: 0.046, J45: 0.05 };

export function simulatedCnn(eye: PowerVector, seed: number): PowerVector {
  const r = mulberry32(seed);
  // For a Gaussian error, σ = MAE·√(π/2).
  const s = Math.sqrt(Math.PI / 2);
  return {
    M: eye.M + gaussian(r) * CNN_MAE.M * s,
    J0: eye.J0 + gaussian(r) * CNN_MAE.J0 * s,
    J45: eye.J45 + gaussian(r) * CNN_MAE.J45 * s,
  };
}

const fmt = (v: number) => (v > 0 ? "+" : "") + v.toFixed(3).replace("-", "−");

export function ClosedLoopFigure({ eye }: { eye: Eye }) {
  const [mode, setMode] = useState<"scan" | "cnn">("scan");
  const [hist, setHist] = useState<LoopEval[]>([]);
  const [running, setRunning] = useState(false);
  const [seed, setSeed] = useState(7);
  const raf = useRef<number | null>(null);
  const pv = useMemo(() => toPowerVector(eye.S, eye.C, eye.axis), [eye.S, eye.C, eye.axis]);
  const cnn = useMemo(() => simulatedCnn(pv, seed), [pv, seed]);

  const stop = () => {
    if (raf.current) cancelAnimationFrame(raf.current);
    raf.current = null;
    setRunning(false);
  };

  useEffect(() => { stop(); setHist([]); }, [pv, mode]);
  useEffect(() => stop, []);

  const run = () => {
    stop();
    setHist([]);
    const warm = mode === "cnn" ? { M: -cnn.M, J0: -cnn.J0, J45: -cnn.J45 } : undefined;
    const gen = closedLoop(pv, target(), N, 2, warm);
    const acc: LoopEval[] = [];
    setRunning(true);
    const tick = () => {
      const t0 = performance.now();
      while (performance.now() - t0 < 14) {
        const r = gen.next();
        if (r.done) { setHist([...acc]); stop(); return; }
        acc.push(r.value);
      }
      setHist([...acc]);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  };

  const best = hist.reduce<LoopEval | null>((b, e) => (!b || e.score > b.score ? e : b), null);
  const current = hist[hist.length - 1];
  const shownResidual = best?.residual ?? pv;
  const perceived = useMemo(() => {
    const { kernel, K } = fastPsf(shownResidual, 2);
    return convolve(target(), N, kernel, K);
  }, [shownResidual]);

  let bestSoFar = Infinity;
  const residualSeries = hist.map((e, i) => {
    bestSoFar = Math.min(bestSoFar, pvNorm(e.residual));
    return [i + 1, Math.max(bestSoFar, 1e-3)] as [number, number];
  });
  const scanPts = hist.map((e, i) => [i + 1, Math.max(pvNorm(e.residual), 1e-3)] as [number, number]);

  return (
    <div>
      <div className={styles.toolbar}>
        <Segmented
          options={[{ value: "scan", label: "Coarse scan, then Nelder-Mead" }, { value: "cnn", label: "CNN estimate, then Nelder-Mead" }]}
          value={mode}
          onChange={setMode}
        />
        <button className="primary" onClick={run} disabled={running}>{running ? "Running…" : "Run loop"}</button>
        {mode === "cnn" && <button onClick={() => { setSeed((s) => s + 1); setHist([]); }}>New CNN draw</button>}
        <div className={styles.spacer} />
        <span className={styles.badge}>Eye M {pv.M.toFixed(2)}, J₀ {pv.J0.toFixed(2)}, J₄₅ {pv.J45.toFixed(2)} D</span>
      </div>
      <div className={styles.split} style={{ gridTemplateColumns: "240px minmax(0,1fr)" }}>
        <div className={styles.pane}>
          <div className={styles.paneHead}>
            <span className={styles.paneTitle}>Best retinal image so far</span>
          </div>
          <div className={styles.canvasFrame}>
            <FieldCanvas data={perceived} n={N} map={gray} label="Siemens star through current best correction" smooth />
          </div>
          <div className={styles.readouts}>
            <Readout label="Evaluations" value={String(hist.length)} />
            <Readout label="Phase" value={current ? (current.phase === "scan" ? "coarse scan" : "Nelder-Mead") : "..."} />
            <Readout label="Residual |Δ|" value={best ? pvNorm(best.residual).toFixed(3) : pvNorm(pv).toFixed(3)} unit="D" tone={best && pvNorm(best.residual) < 0.25 ? "green" : "red"} />
            <Readout label="Lens M" value={best ? fmt(best.correction.M) : "..."} unit="D" />
            <Readout label="Lens J₀" value={best ? fmt(best.correction.J0) : "..."} unit="D" />
            <Readout label="Lens J₄₅" value={best ? fmt(best.correction.J45) : "..."} unit="D" />
          </div>
        </div>
        <div className={styles.pane}>
          <LineChart
            title="Residual refractive error vs. objective evaluations"
            series={[
              { name: "Each evaluation", color: "var(--ink-3)", points: scanPts, dots: true },
              { name: "Best so far", color: "var(--blue)", points: residualSeries },
            ]}
            xLabel="evaluation"
            yLabel="|ΔM, ΔJ₀, ΔJ₄₅| (D)"
            logY
            xDomain={[0, Math.max(40, hist.length + 2)]}
            yDomain={[1e-3, 10]}
            refLines={[{ y: 0.25, label: "0.25 D clinical step" }]}
            formatX={(v) => String(Math.round(v))}
            formatY={(v) => (v >= 1 ? String(v) : v.toString())}
            height={330}
          />
          {mode === "cnn" && (
            <div className={styles.paneFoot}>
              Simulated CNN estimate: M {fmt(cnn.M)}, J₀ {fmt(cnn.J0)}, J₄₅ {fmt(cnn.J45)} D. Drawn as truth + Gaussian
              noise at PupilNet’s measured test MAE. The lens is first commanded to the negated estimate, so
              the coarse scan is skipped.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
