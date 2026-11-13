/** Synthetic test targets and colour maps for the in-browser figures. */

function rasterise(n: number, draw: (ctx: CanvasRenderingContext2D, s: number) => void): Float64Array {
  const ss = 4; // supersample, then box-filter down for clean edges
  const S = n * ss;
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, S, S);
  ctx.fillStyle = "#000";
  draw(ctx, S);
  const data = ctx.getImageData(0, 0, S, S).data;
  const out = new Float64Array(n * n);
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      out[((y / ss) | 0) * n + ((x / ss) | 0)] += data[(y * S + x) * 4] / 255;
  const k = 1 / (ss * ss);
  for (let i = 0; i < out.length; i++) out[i] *= k;
  return out;
}

/**
 * Letter chart at 1 arcmin per pixel. Rows are true Snellen sizes: a 20/20 letter
 * subtends 5 arcmin, a 20/N letter 5·N/20 arcmin.
 */
export function letterChart(n = 256): Float64Array {
  const rows: { denom: number; letters: string }[] = [
    { denom: 200, letters: "E" },
    { denom: 100, letters: "F P" },
    { denom: 70, letters: "T O Z" },
    { denom: 50, letters: "L P E D" },
    { denom: 40, letters: "P E C F D" },
    { denom: 30, letters: "E D F C Z P" },
    { denom: 20, letters: "F E L O P Z D" },
  ];
  return rasterise(n, (ctx, S) => {
    const px = S / n;
    let y = 10 * px;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const r of rows) {
      const h = ((5 * r.denom) / 20) * px;
      ctx.font = `700 ${h * 1.36}px "IBM Plex Sans", Helvetica, Arial, sans-serif`;
      ctx.fillText(r.letters, S / 2, y - h * 0.2);
      y += h * 1.55;
    }
  });
}

export function siemensStar(n = 128, spokes = 24): Float64Array {
  return rasterise(n, (ctx, S) => {
    const c = S / 2, R = S * 0.47;
    for (let i = 0; i < spokes; i++) {
      const a0 = (i * 2 * Math.PI) / spokes;
      const a1 = a0 + Math.PI / spokes;
      ctx.beginPath();
      ctx.moveTo(c, c);
      ctx.arc(c, c, R, a0, a1);
      ctx.closePath();
      ctx.fill();
    }
  });
}

// ---------------------------------------------------------------------------
// Colour maps
// ---------------------------------------------------------------------------

export type RGB = [number, number, number];

/** Polynomial fit to matplotlib's viridis (perceptually uniform, CVD-safe). */
export function viridis(t: number): RGB {
  t = Math.min(1, Math.max(0, t));
  const c0 = [0.2777273272, 0.0054073445, 0.3340998053];
  const c1 = [0.1050930431, 1.4046135299, 1.3845901626];
  const c2 = [-0.3308618288, 0.2148475595, 0.095095163];
  const c3 = [-4.634230499, -5.7991009734, -19.3324409563];
  const c4 = [6.2282699363, 14.1799333668, 56.6905526007];
  const c5 = [4.7763849977, -13.7451453777, -65.3530326334];
  const c6 = [-5.4354558559, 4.6458526122, 26.3124352496];
  const out = [0, 1, 2].map((i) =>
    c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i]))))),
  );
  return out.map((v) => Math.round(255 * Math.min(1, Math.max(0, v)))) as RGB;
}

const DIV_NEG: RGB = [42, 91, 215];
const DIV_MID: RGB = [236, 235, 231];
const DIV_POS: RGB = [209, 82, 46];

/** Diverging blue – neutral grey – vermilion, t ∈ [−1, 1]. */
export function diverging(t: number): RGB {
  t = Math.min(1, Math.max(-1, t));
  const [a, b] = t < 0 ? [DIV_MID, DIV_NEG] : [DIV_MID, DIV_POS];
  const s = Math.abs(t);
  return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * s)) as RGB;
}

export function gray(t: number): RGB {
  const v = Math.round(255 * Math.min(1, Math.max(0, t)));
  return [v, v, v];
}

/** Gray map for images computed in linear light (blurred charts): encodes to sRGB so that
 *  the light emitted by the screen is proportional to the simulated retinal intensity. */
export function graySrgb(t: number): RGB {
  const x = Math.min(1, Math.max(0, t));
  const s = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  const v = Math.round(255 * s);
  return [v, v, v];
}
