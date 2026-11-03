import { useEffect, useRef } from "react";
import type { RGB } from "../lib/images";

/**
 * Paints an n×n scalar field through a colour map. NaN samples are left
 * transparent (used for "outside the pupil").
 */
export function FieldCanvas({
  data, n, map, lo = 0, hi = 1, transform, label, className, smooth = false,
}: {
  data: Float64Array;
  n: number;
  map: (t: number) => RGB;
  lo?: number;
  hi?: number;
  transform?: (v: number) => number;
  label: string;
  className?: string;
  smooth?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = n;
    c.height = n;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(n, n);
    const span = hi - lo || 1;
    for (let i = 0; i < n * n; i++) {
      const raw = data[i];
      if (Number.isNaN(raw)) { img.data[i * 4 + 3] = 0; continue; }
      const v = transform ? transform(raw) : raw;
      const [r, g, b] = map((v - lo) / span);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, n, map, lo, hi, transform]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      className={className}
      style={{ width: "100%", aspectRatio: "1", imageRendering: smooth ? "auto" : "pixelated", display: "block" }}
    />
  );
}
