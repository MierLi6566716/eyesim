import { useState } from "react";
import { ScreenFilterCanvas } from "./ScreenFilterCanvas";
import type { CaseOut } from "../lib/api";
import styles from "./ResultViewer.module.css";

type View = "blurred" | "predistorted" | "live-filter";

export function ResultViewer({
  case: c,
  shaderKernel,
}: {
  case: CaseOut;
  shaderKernel: number[][];
}) {
  const [view, setView] = useState<View>("blurred");
  return (
    <>
      <div className={styles.tabs}>
        <button className={view === "blurred" ? "primary" : ""} onClick={() => setView("blurred")}>
          What the eye sees
        </button>
        <button
          className={view === "predistorted" ? "primary" : ""}
          onClick={() => setView("predistorted")}
        >
          Pre-distorted (corrected)
        </button>
        {shaderKernel.length > 0 && (
          <button
            className={view === "live-filter" ? "primary" : ""}
            onClick={() => setView("live-filter")}
          >
            Live GPU filter
          </button>
        )}
      </div>
      <div className={styles.resultImage}>
        {view === "blurred" && c.blurred_image_url && (
          <img src={c.blurred_image_url} alt="blurred" className={styles.full} />
        )}
        {view === "predistorted" && c.predistorted_image_url && (
          <img src={c.predistorted_image_url} alt="predistorted" className={styles.full} />
        )}
        {view === "live-filter" && (
          <ScreenFilterCanvas imageUrl={c.source_image_url} kernel={shaderKernel} />
        )}
      </div>
    </>
  );
}
