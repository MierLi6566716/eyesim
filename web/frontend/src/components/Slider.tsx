import type { ReactNode } from "react";
import styles from "./Slider.module.css";

export function Slider({
  label,
  value,
  min,
  max,
  step,
  unit = "",
  onChange,
  accent = "cyan",
  digits = 2,
  signed = true,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
  accent?: "cyan" | "amber";
  digits?: number;
  signed?: boolean;
}) {
  const shown = value.toFixed(digits).replace("-", "−");
  return (
    <label className={styles.row}>
      <div className={styles.labelRow}>
        <span className="label-caps">{label}</span>
        <span className={`readout ${styles.value} ${accent === "amber" ? styles.amber : ""}`}>
          {signed && value > 0 ? "+" : ""}
          {shown}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={`${styles.slider} ${accent === "amber" ? styles.amberSlider : ""}`}
      />
    </label>
  );
}
