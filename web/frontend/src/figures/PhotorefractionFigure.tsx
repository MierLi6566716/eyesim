import { useMemo, useState } from "react";
import { FieldCanvas } from "../components/FieldCanvas";
import { Readout } from "../components/Article";
import { Slider } from "../components/Slider";
import { DEFAULT_GEOMETRY, renderPhotorefraction, toPowerVector, type PowerVector } from "../lib/optics";
import { gray } from "../lib/images";
import type { Eye } from "./EyeControls";
import styles from "./figures.module.css";

const N = 96;

function useFrames({ pv, D, ecc, tag }: { pv: PowerVector; D: number; ecc: number; tag: string }) {
  const g = { ...DEFAULT_GEOMETRY, Dmm: D, eccMm: ecc };
  const h = useMemo(() => renderPhotorefraction(pv, ecc, 0, g, N), [pv, D, ecc]); // eslint-disable-line react-hooks/exhaustive-deps
  const v = useMemo(() => renderPhotorefraction(pv, 0, ecc, g, N), [pv, D, ecc]); // eslint-disable-line react-hooks/exhaustive-deps
  return { h, v, tag };
}

export function PhotorefractionFigure({ eye }: { eye: Eye }) {
  const [D, setD] = useState(1000);
  const [ecc, setEcc] = useState(10);
  const [DB, setDB] = useState(800);
  const pvA = useMemo(() => toPowerVector(eye.S, eye.C, eye.axis), [eye.S, eye.C, eye.axis]);
  const k = D / DB;
  const pvB = useMemo(() => ({ M: pvA.M * k, J0: pvA.J0 * k, J45: pvA.J45 * k }), [pvA, k]);
  const A = useFrames({ pv: pvA, D, ecc, tag: "A" });
  const B = useFrames({ pv: pvB, D: DB, ecc, tag: "B" });
  let diff = 0;
  for (let i = 0; i < A.h.length; i++) diff = Math.max(diff, Math.abs(A.h[i] - B.h[i]), Math.abs(A.v[i] - B.v[i]));

  const frame = (data: Float64Array, label: string, meta: string, step: string) => (
    <div className={styles.pane}>
      <div className={styles.paneHead}>
        <span className={styles.paneTitle}><span className={styles.step}>{step}</span>{label}</span>
        <span className={styles.paneMeta}>{meta}</span>
      </div>
      <div className={`${styles.canvasFrame} ${styles.canvasDark}`}>
        <FieldCanvas data={data} n={N} map={gray} label={label} smooth />
      </div>
    </div>
  );

  return (
    <div className={styles.split}>
      <div className={styles.controls}>
        <div className={styles.controlsGroup}>
          <div className={styles.groupTitle}>Camera geometry</div>
          <Slider label="Working distance D" value={D} min={300} max={1500} step={10} unit=" mm" digits={0} signed={false} onChange={setD} />
          <Slider label="LED eccentricity" value={ecc} min={4} max={20} step={0.5} unit=" mm" digits={1} signed={false} onChange={setEcc} />
        </div>
        <div className={styles.controlsGroup}>
          <div className={styles.groupTitle}>Eye B: same M·D, different D</div>
          <Slider label="Distance of eye B" value={DB} min={300} max={1500} step={10} unit=" mm" digits={0} signed={false} onChange={setDB} accent="amber" />
          <Readout label="Eye A: M" value={pvA.M.toFixed(2)} unit="D" />
          <Readout label="Eye B: M" value={pvB.M.toFixed(2)} unit="D" tone="red" />
          <Readout label="M·D (both)" value={(pvA.M * D / 1000).toFixed(3)} unit="D·m" />
          <Readout label="max |A − B| pixel" value={diff.toExponential(1)} tone="green" />
        </div>
      </div>
      <div>
        <div className={styles.quad}>
          {frame(A.h, "A, LED 0°", `ex=${ecc}`, "a")}
          {frame(A.v, "A, LED 90°", `ey=${ecc}`, "b")}
          {frame(B.h, "B, LED 0°", `D=${DB}`, "c")}
          {frame(B.v, "B, LED 90°", `D=${DB}`, "d")}
        </div>
        <div className={styles.paneFoot} style={{ marginTop: 12 }}>
          Eye A is the prescription from Figure 2 at distance D. Eye B has a different refractive error
          at a different distance, chosen so that M·D matches. The four frames are pixel-identical, so no
          network, however large, can tell A from B without knowing D.
        </div>
      </div>
    </div>
  );
}
