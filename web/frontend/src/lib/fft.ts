/**
 * Minimal radix-2 complex FFT (in place) plus 2-D helpers. Sizes must be
 * powers of two. Used by the in-browser optics so the figures run without
 * the Python backend.
 */

const twiddleCache = new Map<number, { cos: Float64Array; sin: Float64Array; rev: Uint32Array }>();

function tables(n: number) {
  let t = twiddleCache.get(n);
  if (t) return t;
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = Math.sin((2 * Math.PI * i) / n);
  }
  const rev = new Uint32Array(n);
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  t = { cos, sin, rev };
  twiddleCache.set(n, t);
  return t;
}

/** 1-D FFT over a strided view of (re, im). inverse=true omits the 1/n scale. */
function fft1d(
  re: Float64Array, im: Float64Array, n: number, offset: number, stride: number, inverse: boolean,
) {
  const { cos, sin, rev } = tables(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      const a = offset + i * stride, b = offset + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  const sgn = inverse ? 1 : -1;
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sgn * sin[k * step];
        const a = offset + (start + k) * stride;
        const b = offset + (start + k + half) * stride;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

/** In-place 2-D FFT of an n×n complex field (row-major). */
export function fft2d(re: Float64Array, im: Float64Array, n: number, inverse = false) {
  for (let r = 0; r < n; r++) fft1d(re, im, n, r * n, 1, inverse);
  for (let c = 0; c < n; c++) fft1d(re, im, n, c, n, inverse);
  if (inverse) {
    const s = 1 / (n * n);
    for (let i = 0; i < n * n; i++) { re[i] *= s; im[i] *= s; }
  }
}

/** Swap quadrants so the zero-frequency sample sits at the array centre. */
export function fftshift(a: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  const h = n >> 1;
  for (let y = 0; y < n; y++) {
    const yy = (y + h) % n;
    for (let x = 0; x < n; x++) out[yy * n + ((x + h) % n)] = a[y * n + x];
  }
  return out;
}
