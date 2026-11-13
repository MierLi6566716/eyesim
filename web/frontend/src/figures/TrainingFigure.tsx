import { useMemo } from "react";
import { LineChart } from "../components/LineChart";
import log from "../data/train_log.csv?raw";
import styles from "./figures.module.css";

type Row = { epoch: number; train: number; val: number; M: number; J0: number; J45: number; Mvis: number };

function parse(): Row[] {
  const [, ...lines] = log.trim().split("\n");
  return lines.map((l) => {
    const c = l.split(",").map(Number);
    return { epoch: c[0], train: c[1], val: c[2], M: c[3], J0: c[4], J45: c[5], Mvis: c[6] };
  });
}

export function TrainingFigure() {
  const rows = useMemo(parse, []);
  return (
    <div className={styles.duo}>
      <LineChart
        title="Loss (MSE on M, J₀, J₄₅)"
        series={[
          { name: "Train", color: "var(--blue)", points: rows.map((r) => [r.epoch, r.train]) },
          { name: "Validation", color: "var(--red)", points: rows.map((r) => [r.epoch, r.val]), dashed: true },
        ]}
        xLabel="epoch"
        yLabel="loss"
        formatX={(v) => String(Math.round(v))}
        formatY={(v) => v.toFixed(3)}
      />
      <LineChart
        title="Validation mean absolute error"
        series={[
          { name: "M, all frames", color: "var(--ink-3)", points: rows.map((r) => [r.epoch, r.M]), dashed: true },
          { name: "M, crescent visible", color: "var(--blue)", points: rows.map((r) => [r.epoch, r.Mvis]) },
          { name: "J₀", color: "var(--red)", points: rows.map((r) => [r.epoch, r.J0]) },
          { name: "J₄₅", color: "var(--green)", points: rows.map((r) => [r.epoch, r.J45]) },
        ]}
        xLabel="epoch"
        yLabel="MAE (D)"
        yDomain={[0, 2.1]}
        refLines={[{ y: 0.125, label: "½ clinical step" }]}
        formatX={(v) => String(Math.round(v))}
        formatY={(v) => v.toFixed(3)}
      />
    </div>
  );
}
