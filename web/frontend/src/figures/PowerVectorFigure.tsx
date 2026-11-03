import { Readout } from "../components/Article";
import { toPowerVector } from "../lib/optics";
import { EyeControls, type Eye } from "./EyeControls";
import styles from "./figures.module.css";

const fmt = (v: number) => (v > 0 ? "+" : "") + v.toFixed(2).replace("-", "−");

/** Principal meridians of the eye and the same prescription as a point in the J-plane. */
export function PowerVectorFigure({ eye, onChange }: { eye: Eye; onChange: (e: Eye) => void }) {
  const pv = toPowerVector(eye.S, eye.C, eye.axis);
  const a = (eye.axis * Math.PI) / 180;
  const R = 78;
  const cx = 100, cy = 100;
  const line = (ang: number) => ({
    x1: cx - R * Math.cos(ang), y1: cy + R * Math.sin(ang),
    x2: cx + R * Math.cos(ang), y2: cy - R * Math.sin(ang),
  });
  const m1 = line(a), m2 = line(a + Math.PI / 2);
  const lab = (ang: number, r: number) => ({ x: cx + r * Math.cos(ang), y: cy - r * Math.sin(ang) });
  const l1 = lab(a, R + 14), l2 = lab(a + Math.PI / 2, R + 14);

  // J-plane
  const J = 2, P = 80, jx = 100, jy = 100;
  const px = jx + (pv.J0 / J) * P, py = jy - (pv.J45 / J) * P;

  return (
    <div className={styles.split}>
      <EyeControls eye={eye} onChange={onChange} pupil={false} />
      <div>
        <div className={styles.duo}>
          <div className={styles.pane}>
            <div className={styles.paneHead}>
              <span className={styles.paneTitle}>Principal meridians</span>
              <span className={styles.paneMeta}>front view, TABO</span>
            </div>
            <svg viewBox="-10 -6 220 212" className={styles.svgFig} role="img" aria-label="Principal meridians diagram">
              <circle cx={cx} cy={cy} r={R} fill="var(--surface-2)" stroke="var(--rule-strong)" />
              {[0, 30, 60, 90, 120, 150].map((d) => {
                const t = (d * Math.PI) / 180;
                return <line key={d} {...line(t)} stroke="var(--rule)" strokeWidth={1} />;
              })}
              <circle cx={cx} cy={cy} r={R * 0.34} fill="var(--surface)" stroke="var(--rule-strong)" />
              <line {...m1} stroke="var(--blue)" strokeWidth={2.5} strokeLinecap="round" />
              <line {...m2} stroke="var(--red)" strokeWidth={2.5} strokeLinecap="round" strokeDasharray={eye.C === 0 ? "4 4" : undefined} />
              <text x={l1.x} y={l1.y} textAnchor="middle" dominantBaseline="middle" style={{ fill: "var(--blue)", fontWeight: 600 }}>
                {fmt(eye.S)}
              </text>
              <text x={l2.x} y={l2.y} textAnchor="middle" dominantBaseline="middle" style={{ fill: "var(--red)", fontWeight: 600 }}>
                {fmt(eye.S + eye.C)}
              </text>
              <text x={cx + R + 2} y={cy + 14} style={{ fontSize: 10.5, fill: "var(--ink-3)" }}>0°</text>
              <text x={cx + 4} y={cy - R - 4} style={{ fontSize: 10.5, fill: "var(--ink-3)" }}>90°</text>
            </svg>
            <div className={styles.paneFoot}>
              <span style={{ color: "var(--blue)" }}>━</span> axis meridian (power S),{" "}
              <span style={{ color: "var(--red)" }}>━</span> perpendicular meridian (S + C)
            </div>
          </div>

          <div className={styles.pane}>
            <div className={styles.paneHead}>
              <span className={styles.paneTitle}>Astigmatism in the J-plane</span>
              <span className={styles.paneMeta}>±2 D</span>
            </div>
            <svg viewBox="0 0 200 200" className={styles.svgFig} role="img" aria-label="J0 J45 plane">
              {[-1, -0.5, 0.5, 1].map((t) => (
                <g key={t}>
                  <line x1={jx + t * P} x2={jx + t * P} y1={jy - P} y2={jy + P} stroke="var(--rule)" />
                  <line y1={jy + t * P} y2={jy + t * P} x1={jx - P} x2={jx + P} stroke="var(--rule)" />
                </g>
              ))}
              <circle cx={jx} cy={jy} r={P / 2} fill="none" stroke="var(--rule-strong)" strokeDasharray="2 3" />
              <line x1={jx - P} x2={jx + P} y1={jy} y2={jy} stroke="var(--ink-3)" />
              <line y1={jy - P} y2={jy + P} x1={jx} x2={jx} stroke="var(--ink-3)" />
              <line x1={jx} y1={jy} x2={px} y2={py} stroke="var(--ink)" strokeWidth={1.5} />
              <circle cx={px} cy={py} r={6} fill="var(--red)" stroke="var(--surface)" strokeWidth={2} />
              <text x={jx + P - 2} y={jy - 6} textAnchor="end">J₀</text>
              <text x={jx + 6} y={jy - P + 10}>J₄₅</text>
              <text x={jx + P / 2 + 3} y={jy + 13} style={{ fontSize: 10, fill: "var(--ink-3)" }}>1 D</text>
            </svg>
            <div className={styles.paneFoot}>
              Vector length = |C|/2; angle = 2θ. Axis wrap-around (0° ≡ 180°) disappears here, which is why
              the CNN regresses (M, J₀, J₄₅) and not (S, C, θ).
            </div>
          </div>
        </div>
        <div className={styles.readouts}>
          <Readout label="Spherical equivalent M = S + C/2" value={fmt(pv.M)} unit="D" tone="blue" />
          <Readout label="J₀ = −(C/2)·cos 2θ" value={fmt(pv.J0)} unit="D" />
          <Readout label="J₄₅ = −(C/2)·sin 2θ" value={fmt(pv.J45)} unit="D" />
        </div>
      </div>
    </div>
  );
}
