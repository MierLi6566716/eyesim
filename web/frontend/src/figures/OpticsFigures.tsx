import { useDeferredValue, useMemo, useState } from "react";
import { FieldCanvas } from "../components/FieldCanvas";
import { Readout } from "../components/Article";
import { Slider } from "../components/Slider";
import {
  ARCMIN_PER_PX, computePsf, convolve, predistort, sharpness, toPowerVector, wavefrontWaves,
} from "../lib/optics";
import { diverging, graySrgb, letterChart, viridis } from "../lib/images";
import { EyeControls, type Eye } from "./EyeControls";
import styles from "./figures.module.css";

const N_IMG = 256;
/** Accommodation amplitude of a young adult (Hofstetter minimum at age 25 is 8.75 D). */
const ACCOMMODATION_D = 8;
/** CSS pixels per arcminute at a 60 cm viewing distance (1 CSS px = 0.2646 mm). */
const CSS_PX_PER_ARCMIN = (600 * Math.tan(Math.PI / 10800)) / 0.2646;
/** Width at which the chart subtends its true angle (1 image px = 1 arcmin) at 60 cm. */
const TRUE_SCALE_W = Math.round(N_IMG * ARCMIN_PER_PX * CSS_PX_PER_ARCMIN);
let chartCache: Float64Array | null = null;
function chart() {
  if (!chartCache) chartCache = letterChart(N_IMG);
  return chartCache;
}

const logT = (v: number) => Math.log10(Math.max(v, 1e-3));

function useOptics(eye: Eye, distM: number) {
  const d = useDeferredValue(eye);
  const dd = useDeferredValue(distM);
  return useMemo(() => {
    const pv = toPowerVector(d.S, d.C, d.axis);
    const wf = wavefrontWaves(pv, d.pupil, 128);
    let wmax = 0;
    for (const v of wf) if (!Number.isNaN(v)) wmax = Math.max(wmax, Math.abs(v));
    // Defocus at the retina for a chart at distance dd: the chart adds a vergence of 1/dd,
    // and accommodation cancels the positive part up to its amplitude. Accommodation cannot
    // relax below zero, so a myopic eye is only in focus when the chart sits at (or nearer
    // than its accommodation allows from) its far point at 1/|M| metres. Astigmatism is
    // unaffected by accommodation.
    const demand = pv.M + 1 / dd;
    const accommodated = Math.min(Math.max(demand, 0), ACCOMMODATION_D);
    const residualM = demand - accommodated;
    const pvSeen = { ...pv, M: residualM };
    const psf = computePsf(pvSeen, d.pupil);
    const img = chart();
    const retina = convolve(img, N_IMG, psf.kernel, psf.K);
    return { pv, wf, wmax, psf, img, retina, accommodated, residualM };
  }, [d, dd]);
}

function ColorScale({ map, lo, hi }: { map: (t: number) => [number, number, number]; lo: string; hi: string }) {
  const stops = Array.from({ length: 9 }, (_, i) => {
    const [r, g, b] = map(i / 8);
    return `rgb(${r},${g},${b}) ${(i / 8) * 100}%`;
  }).join(",");
  return (
    <div className={styles.scale}>
      <span>{lo}</span>
      <span className={styles.scaleBar} style={{ background: `linear-gradient(90deg, ${stops})` }} />
      <span>{hi}</span>
    </div>
  );
}

const divMap = (t: number) => diverging(t * 2 - 1);

/** Figure: prescription → wavefront → PSF → retinal image. */
export function OpticsChainFigure({
  eye, onChange, dist, onDist,
}: { eye: Eye; onChange: (e: Eye) => void; dist: number; onDist: (d: number) => void }) {
  const o = useOptics(eye, dist);
  const w = Math.max(o.wmax, 0.25);
  const acuityLine = snellenLimit(o.psf.blurArcmin);
  const farPoint = o.pv.M < -0.01 ? -1 / o.pv.M : null;
  return (
    <div className={styles.split}>
      <div>
        <EyeControls eye={eye} onChange={onChange} />
        <div className={styles.controlsGroup} style={{ marginTop: 22 }}>
          <div className={styles.groupTitle}>Letter chart</div>
          <Slider
            label="Chart distance" value={dist} min={0.25} max={6} step={0.05} unit=" m"
            digits={2} signed={false} onChange={onDist}
          />
        </div>
        <div className={styles.readouts}>
          <Readout label="Defocus at the retina" value={o.residualM.toFixed(2)} unit="D" />
          {o.accommodated > 0.005 && <Readout label="Cancelled by accommodation" value={o.accommodated.toFixed(2)} unit="D" />}
          {farPoint != null && <Readout label="Far point of this eye" value={farPoint.toFixed(2)} unit="m" />}
          <Readout label="Peak wavefront error" value={o.wmax.toFixed(1)} unit="λ" />
          <Readout label="Geometric blur ⌀" value={o.psf.blurArcmin.toFixed(1)} unit="arcmin" tone="red" />
          <Readout label="Smallest legible row ≈" value={acuityLine} />
        </div>
      </div>
      <div>
        <div className={styles.triptych}>
          <div className={styles.pane}>
            <div className={styles.paneHead}>
              <span className={styles.paneTitle}><span className={styles.step}>a</span>Pupil wavefront W(x,y)</span>
            </div>
            <div className={styles.canvasFrame} style={{ background: "transparent", outline: "none" }}>
              <FieldCanvas data={o.wf} n={128} map={divMap} lo={-w} hi={w} label="Wavefront error map" smooth />
            </div>
            <ColorScale map={divMap} lo={`−${w.toFixed(1)}λ`} hi={`+${w.toFixed(1)}λ`} />
          </div>
          <div className={styles.pane}>
            <div className={styles.paneHead}>
              <span className={styles.paneTitle}><span className={styles.step}>b</span>PSF |𝓕{"{P}"}|²</span>
              <span className={styles.paneMeta}>{o.psf.displayFieldArcmin.toFixed(0)}′ field</span>
            </div>
            <div className={styles.canvasFrame}>
              <FieldCanvas data={o.psf.display} n={o.psf.displayN} map={viridis} lo={-3} hi={0} transform={logT} label="Point spread function, log scale" />
            </div>
            <ColorScale map={viridis} lo="10⁻³" hi="1" />
          </div>
          <div className={styles.pane}>
            <div className={styles.paneHead}>
              <span className={styles.paneTitle}><span className={styles.step}>c</span>Retinal image</span>
              <span className={styles.paneMeta}>{(N_IMG * ARCMIN_PER_PX / 60).toFixed(1)}° field</span>
            </div>
            <div className={styles.canvasFrame} style={{ display: "flex", justifyContent: "center", padding: "12px 0" }}>
              <div style={{ width: TRUE_SCALE_W }}>
                <FieldCanvas data={o.retina} n={N_IMG} map={graySrgb} label="Letter chart as seen by this eye" smooth />
              </div>
            </div>
            <div className={styles.paneFoot}>
              <span className={styles.badge}>
                <span className={styles.badgeDot} style={o.psf.mode === "geometric" ? { background: "var(--red)" } : undefined} />
                {o.psf.mode === "wave" ? "Fourier (wave) optics" : "Ray optics (blur ≫ diffraction)"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function snellenLimit(blurArcmin: number) {
  // A letter stays legible while its height is at least about 1.2 times the blur-disc diameter.
  // This reproduces published uncorrected acuities for myopia at a 3 mm pupil:
  // -0.5 D about 20/25, -1 D about 20/50, -2 D about 20/100.
  const rows = [20, 25, 30, 40, 50, 70, 100, 200];
  for (const r of rows) if (blurArcmin <= (5 * r) / 20 / 1.2 + 0.6) return `20/${r}`;
  return "worse than 20/200";
}

/** Figure: Wiener pre-distortion (Step 2). */
export function PredistortFigure({ eye, dist }: { eye: Eye; dist: number }) {
  const o = useOptics(eye, dist);
  const [k, setK] = useState(-3);
  const dk = useDeferredValue(k);
  const pre = useMemo(() => predistort(o.img, N_IMG, o.psf.kernel, o.psf.K, 10 ** dk), [o, dk]);
  const sOrig = sharpness(o.img, N_IMG);
  const pct = (s: number) => `${((100 * s) / sOrig).toFixed(0)}%`;
  return (
    <div>
      <div className={styles.toolbar}>
        <div style={{ width: 260 }}>
          <Slider label="Wiener regulariser log₁₀ k" value={k} min={-4} max={-0.5} step={0.25} onChange={setK} signed={false} />
        </div>
        <div className={styles.spacer} />
        <span className={styles.badge}>
          Eye: S {eye.S.toFixed(2)} / C {eye.C.toFixed(2)} × {eye.axis}°, pupil {eye.pupil} mm, screen at {dist.toFixed(2)} m (set in Fig. 2)
        </span>
      </div>
      <div className={styles.triptych}>
        <div className={styles.pane}>
          <div className={styles.paneHead}>
            <span className={styles.paneTitle}><span className={styles.step}>a</span>Normal screen, seen by the eye</span>
          </div>
          <div className={styles.canvasFrame} style={{ display: "flex", justifyContent: "center", padding: "12px 0" }}><div style={{ width: TRUE_SCALE_W }}><FieldCanvas data={o.retina} n={N_IMG} map={graySrgb} label="Blurred chart" smooth /></div></div>
          <div className={styles.paneFoot}>Edge energy {pct(sharpness(o.retina, N_IMG))} of original</div>
        </div>
        <div className={styles.pane}>
          <div className={styles.paneHead}>
            <span className={styles.paneTitle}><span className={styles.step}>b</span>Pre-distorted screen</span>
          </div>
          <div className={styles.canvasFrame} style={{ display: "flex", justifyContent: "center", padding: "12px 0" }}><div style={{ width: TRUE_SCALE_W }}><FieldCanvas data={pre.display} n={N_IMG} map={graySrgb} label="Pre-distorted chart" smooth /></div></div>
          <div className={styles.paneFoot}>What the display shows, clipped to [0, 1]</div>
        </div>
        <div className={styles.pane}>
          <div className={styles.paneHead}>
            <span className={styles.paneTitle}><span className={styles.step}>c</span>Pre-distorted, seen by the eye</span>
          </div>
          <div className={styles.canvasFrame} style={{ display: "flex", justifyContent: "center", padding: "12px 0" }}><div style={{ width: TRUE_SCALE_W }}><FieldCanvas data={pre.perceived} n={N_IMG} map={graySrgb} label="Pre-distorted chart as perceived" smooth /></div></div>
          <div className={styles.paneFoot}>Edge energy {pct(sharpness(pre.perceived, N_IMG))} of original</div>
        </div>
      </div>
    </div>
  );
}
