import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import styles from "./LineChart.module.css";

export type Series = {
  name: string;
  color: string;
  points: [number, number][];
  dashed?: boolean;
  /** Draw dots instead of a line. */
  dots?: boolean;
};

type Props = {
  series: Series[];
  xLabel: string;
  yLabel: string;
  height?: number;
  xDomain?: [number, number];
  yDomain?: [number, number];
  logY?: boolean;
  refLines?: { y: number; label: string }[];
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  title?: ReactNode;
  hideLegend?: boolean;
};

function niceTicks(lo: number, hi: number, count = 5) {
  const span = hi - lo || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return out;
}

export function LineChart({
  series, xLabel, yLabel, height = 260, xDomain, yDomain, logY = false, refLines = [],
  formatX = (v) => String(+v.toFixed(2)), formatY = (v) => String(+v.toPrecision(3)), title, hideLegend = false,
}: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(260, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const m = { l: 56, r: 16, t: 12, b: 42 };
  const W = width, H = height;
  const iw = W - m.l - m.r, ih = H - m.t - m.b;

  const all = series.flatMap((s) => s.points);
  const [x0, x1] = xDomain ?? [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
  const ty = (v: number) => (logY ? Math.log10(Math.max(v, 1e-12)) : v);
  const [y0, y1] = useMemo(() => {
    if (yDomain) return [ty(yDomain[0]), ty(yDomain[1])];
    const ys = all.map((p) => ty(p[1])).concat(refLines.map((r) => ty(r.y)));
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (!logY) { const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad; }
    else { lo = Math.floor(lo); hi = Math.ceil(hi); }
    return [lo, hi];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, yDomain, logY, refLines]);

  const sx = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
  const sy = (v: number) => m.t + ih - ((ty(v) - y0) / (y1 - y0 || 1)) * ih;

  const xt = niceTicks(x0, x1, Math.max(3, Math.floor(iw / 90)));
  const yt = logY
    ? Array.from({ length: y1 - y0 + 1 }, (_, i) => 10 ** (y0 + i))
    : niceTicks(y0, y1, 5);

  const hoverX = hover == null ? null : x0 + ((hover - m.l) / iw) * (x1 - x0);
  const nearest = hoverX == null ? [] : series.filter((s) => s.name).map((s) => {
    let best = s.points[0];
    for (const p of s.points) if (Math.abs(p[0] - hoverX) < Math.abs(best[0] - hoverX)) best = p;
    return { s, p: best };
  }).filter((d) => d.p);

  return (
    <div className={styles.wrap} ref={wrap}>
      {(title || (series.length > 1 && !hideLegend)) && (
        <div className={styles.head}>
          {title && <div className={styles.title}>{title}</div>}
          {series.length > 1 && !hideLegend && (
            <div className={styles.legend}>
              {series.map((s) => (
                <span key={s.name} className={styles.legendItem}>
                  <svg width="18" height="10" aria-hidden>
                    {s.dots
                      ? <circle cx="9" cy="5" r="4" fill={s.color} />
                      : <line x1="0" x2="18" y1="5" y2="5" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "4 3" : undefined} />}
                  </svg>
                  {s.name}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
      <svg
        width={W}
        height={H}
        className={styles.svg}
        onPointerMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const x = e.clientX - r.left;
          setHover(x >= m.l && x <= m.l + iw ? x : null);
        }}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={`${yLabel} versus ${xLabel}`}
      >
        {yt.map((v) => (
          <g key={`y${v}`}>
            <line x1={m.l} x2={m.l + iw} y1={sy(v)} y2={sy(v)} className={styles.grid} />
            <text x={m.l - 8} y={sy(v)} className={styles.tick} textAnchor="end" dominantBaseline="middle">
              {formatY(v)}
            </text>
          </g>
        ))}
        {xt.map((v) => (
          <text key={`x${v}`} x={sx(v)} y={m.t + ih + 18} className={styles.tick} textAnchor="middle">
            {formatX(v)}
          </text>
        ))}
        <line x1={m.l} x2={m.l + iw} y1={m.t + ih} y2={m.t + ih} className={styles.axis} />
        <text x={m.l + iw / 2} y={H - 4} className={styles.axisLabel} textAnchor="middle">{xLabel}</text>
        <text transform={`translate(13 ${m.t + ih / 2}) rotate(-90)`} className={styles.axisLabel} textAnchor="middle">
          {yLabel}
        </text>

        {refLines.map((r) => (
          <g key={r.label}>
            <line x1={m.l} x2={m.l + iw} y1={sy(r.y)} y2={sy(r.y)} className={styles.ref} />
            <text x={m.l + iw - 4} y={sy(r.y) - 6} className={styles.refLabel} textAnchor="end">{r.label}</text>
          </g>
        ))}

        {series.map((s) =>
          s.dots ? (
            <g key={s.name}>
              {s.points.map((p, i) => (
                <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={3.5} fill={s.color} className={styles.dot} />
              ))}
            </g>
          ) : (
            <path
              key={s.name}
              d={s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeDasharray={s.dashed ? "5 4" : undefined}
            />
          ),
        )}

        {hover != null && nearest.length > 0 && (
          <g pointerEvents="none">
            <line x1={sx(nearest[0].p[0])} x2={sx(nearest[0].p[0])} y1={m.t} y2={m.t + ih} className={styles.cross} />
            {nearest.map(({ s, p }) => (
              <circle key={s.name} cx={sx(p[0])} cy={sy(p[1])} r={4.5} fill={s.color} className={styles.hoverDot} />
            ))}
          </g>
        )}
      </svg>
      {hover != null && nearest.length > 0 && (
        <div
          className={styles.tooltip}
          style={{
            left: Math.min(sx(nearest[0].p[0]) + 12, W - 180),
            top: m.t + 4,
          }}
        >
          <div className={styles.ttHead}>{xLabel}: {formatX(nearest[0].p[0])}</div>
          {nearest.map(({ s, p }) => (
            <div key={s.name} className={styles.ttRow}>
              <span className={styles.ttSwatch} style={{ background: s.color }} />
              <span>{s.name}</span>
              <span className={styles.ttVal}>{formatY(p[1])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
