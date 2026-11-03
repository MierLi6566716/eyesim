import { Slider } from "../components/Slider";
import styles from "./figures.module.css";

export type Eye = { S: number; C: number; axis: number; pupil: number };

export const PRESETS: { label: string; eye: Omit<Eye, "pupil"> }[] = [
  { label: "Emmetropic", eye: { S: 0, C: 0, axis: 0 } },
  { label: "Mild myopia", eye: { S: -1, C: 0, axis: 0 } },
  { label: "Myopic astig.", eye: { S: -2.25, C: -0.75, axis: 30 } },
  { label: "Hyperopia", eye: { S: 1.5, C: 0, axis: 0 } },
  { label: "Pure astig.", eye: { S: 0, C: -1.5, axis: 90 } },
];

export function EyeControls({
  eye, onChange, pupil = true, presets = true,
}: { eye: Eye; onChange: (e: Eye) => void; pupil?: boolean; presets?: boolean }) {
  const set = (k: keyof Eye) => (v: number) => onChange({ ...eye, [k]: v });
  return (
    <div className={styles.controls}>
      {presets && (
        <div className={styles.controlsGroup}>
          <div className={styles.groupTitle}>Presets</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {PRESETS.map((p) => (
              <button
                key={p.label}
                style={{ padding: "4px 9px", fontSize: 12.5 }}
                onClick={() => onChange({ ...eye, ...p.eye })}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className={styles.controlsGroup}>
        <div className={styles.groupTitle}>Prescription</div>
        <Slider label="Sphere S" value={eye.S} min={-6} max={4} step={0.25} unit=" D" onChange={set("S")} />
        <Slider label="Cylinder C" value={eye.C} min={-3} max={0} step={0.25} unit=" D" onChange={set("C")} />
        <Slider label="Axis θ" value={eye.axis} min={0} max={180} step={5} unit="°" digits={0} signed={false} onChange={set("axis")} accent="amber" />
        {pupil && (
          <Slider label="Pupil radius" value={eye.pupil} min={1} max={3} step={0.25} unit=" mm" signed={false} onChange={set("pupil")} />
        )}
      </div>
    </div>
  );
}
