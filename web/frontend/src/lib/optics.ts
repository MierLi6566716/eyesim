/**
 * Browser port of the eyesim physics (eyesim/optics.py, eyesim/photorefraction.py,
 * eyesim/predistort.py, eyesim/optimize.py). The conventions match the Python code:
 * minus-cylinder prescriptions, Thibos power vectors, and a Zernike wavefront in metres.
 */
import { fft2d, fftshift } from "./fft";

export const WAVELENGTH_M = 550e-9;
/** Angular size of one pixel in the simulated retinal images. */
export const ARCMIN_PER_PX = 1;
const RAD_PER_ARCMIN = Math.PI / (180 * 60);
const RAD_PER_PX = ARCMIN_PER_PX * RAD_PER_ARCMIN;

export type PowerVector = { M: number; J0: number; J45: number };

export function toPowerVector(S: number, C: number, axisDeg: number): PowerVector {
  const t = (axisDeg * Math.PI) / 180;
  return { M: S + C / 2, J0: -(C / 2) * Math.cos(2 * t), J45: -(C / 2) * Math.sin(2 * t) };
}

export function toPrescription({ M, J0, J45 }: PowerVector) {
  const astig = Math.hypot(J0, J45);
  const C = -2 * astig;
  const axis = astig > 1e-6 ? ((0.5 * (Math.atan2(J45, J0) * 180)) / Math.PI + 180) % 180 : 0;
  return { S: M + astig, C, axis };
}

export function pvAdd(a: PowerVector, b: PowerVector): PowerVector {
  return { M: a.M + b.M, J0: a.J0 + b.J0, J45: a.J45 + b.J45 };
}

export function pvNorm(p: PowerVector) {
  return Math.hypot(p.M, p.J0, p.J45);
}

/** Largest principal-meridian power magnitude (dioptres). */
function maxMeridian(p: PowerVector) {
  return Math.abs(p.M) + Math.hypot(p.J0, p.J45);
}

// ---------------------------------------------------------------------------
// Wavefront and PSF
// ---------------------------------------------------------------------------

/** Wavefront error across the pupil, in waves, on an n×n grid (NaN outside). */
export function wavefrontWaves(p: PowerVector, pupilRmm: number, n = 128): Float64Array {
  const r = pupilRmm * 1e-3;
  const out = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = ((i + 0.5) / n) * 2 - 1;
      const v = ((j + 0.5) / n) * 2 - 1;
      const rho2 = u * u + v * v;
      if (rho2 > 1) { out[j * n + i] = NaN; continue; }
      const x = u * r, y = -v * r;
      // W = -(M ρ² r²)/2 - J0 (x²−y²)/2 - J45 xy  (piston removed)
      const W = -0.5 * p.M * (x * x + y * y) - 0.5 * p.J0 * (x * x - y * y) - p.J45 * x * y;
      out[j * n + i] = W / WAVELENGTH_M;
    }
  }
  return out;
}

export type PsfResult = {
  /** Kernel sampled at ARCMIN_PER_PX, odd size K×K, sums to 1. */
  kernel: Float64Array;
  K: number;
  mode: "wave" | "geometric";
  /** Display-resolution PSF (normalised to peak 1) and its field width in arcmin. */
  display: Float64Array;
  displayN: number;
  displayFieldArcmin: number;
  /** Geometric blur-disc diameter (arcmin) of the widest meridian. */
  blurArcmin: number;
};

const MAX_WAVE_N = 512;

export function computePsf(p: PowerVector, pupilRmm: number): PsfResult {
  const r = pupilRmm * 1e-3;
  const mm = maxMeridian(p);
  const blurRad = mm * 2 * r;
  // PSF sample spacing with the pupil filling half the grid: δ = λ/(4r).
  const delta = WAVELENGTH_M / (4 * r);
  const needN = (1.35 * blurRad) / delta + 48;
  if (needN <= MAX_WAVE_N) {
    const N = needN <= 256 ? 256 : 512;
    return wavePsf(p, r, N, delta, blurRad);
  }
  return geometricPsf(p, r, blurRad);
}

function wavePsf(p: PowerVector, r: number, N: number, delta: number, blurRad: number): PsfResult {
  const P = N / 2;
  const re = new Float64Array(N * N);
  const im = new Float64Array(N * N);
  const k = (2 * Math.PI) / WAVELENGTH_M;
  const c = N / 2;
  for (let j = 0; j < N; j++) {
    const v = (j - c + 0.5) / (P / 2);
    if (Math.abs(v) > 1) continue;
    for (let i = 0; i < N; i++) {
      const u = (i - c + 0.5) / (P / 2);
      if (u * u + v * v > 1) continue;
      const x = u * r, y = -v * r;
      const W = -0.5 * p.M * (x * x + y * y) - 0.5 * p.J0 * (x * x - y * y) - p.J45 * x * y;
      const ph = k * W;
      re[j * N + i] = Math.cos(ph);
      im[j * N + i] = Math.sin(ph);
    }
  }
  fft2d(re, im, N);
  const inten = new Float64Array(N * N);
  let sum = 0;
  for (let i = 0; i < N * N; i++) { inten[i] = re[i] * re[i] + im[i] * im[i]; sum += inten[i]; }
  const psf = fftshift(inten, N);
  for (let i = 0; i < N * N; i++) psf[i] /= sum;

  // Resample to image pixels (bilinear splat).
  const scale = delta / RAD_PER_PX;
  const half = Math.min(127, Math.ceil((N / 2) * scale) + 1);
  const K = 2 * half + 1;
  const kernel = new Float64Array(K * K);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const w = psf[j * N + i];
      if (w < 1e-12) continue;
      splat(kernel, K, half + (i - c) * scale, half + (j - c) * scale, w);
    }
  }
  normalise(kernel);

  // Display crop: power-of-two window around the blur, box-downsampled to ≤128.
  const blurPx = blurRad / delta;
  let crop = 64;
  while (crop < blurPx * 1.5 + 24 && crop < N) crop *= 2;
  const displayN = Math.min(crop, 128);
  const f = crop / displayN;
  const display = new Float64Array(displayN * displayN);
  const o = c - crop / 2;
  for (let j = 0; j < crop; j++)
    for (let i = 0; i < crop; i++)
      display[Math.floor(j / f) * displayN + Math.floor(i / f)] += psf[(o + j) * N + (o + i)];
  peakNormalise(display);

  return {
    kernel, K, mode: "wave", display, displayN,
    displayFieldArcmin: (crop * delta) / RAD_PER_ARCMIN,
    blurArcmin: blurRad / RAD_PER_ARCMIN,
  };
}

/** Ray-optics PSF: each pupil point is deflected by the local wavefront slope. */
function geometricPsf(p: PowerVector, r: number, blurRad: number, samples = 150): PsfResult {
  const halfPx = Math.min(127, Math.ceil((0.55 * blurRad) / RAD_PER_PX) + 2);
  const K = 2 * halfPx + 1;
  const kernel = new Float64Array(K * K);
  for (let j = 0; j < samples; j++) {
    const v = ((j + 0.5) / samples) * 2 - 1;
    for (let i = 0; i < samples; i++) {
      const u = ((i + 0.5) / samples) * 2 - 1;
      if (u * u + v * v > 1) continue;
      const x = u * r, y = v * r;
      const tx = -((p.M + p.J0) * x + p.J45 * y);
      const ty = -((p.M - p.J0) * y + p.J45 * x);
      splat(kernel, K, halfPx + tx / RAD_PER_PX, halfPx - ty / RAD_PER_PX, 1);
    }
  }
  normalise(kernel);
  const display = kernel.slice();
  peakNormalise(display);
  return {
    kernel, K, mode: "geometric", display, displayN: K,
    displayFieldArcmin: K * ARCMIN_PER_PX,
    blurArcmin: blurRad / RAD_PER_ARCMIN,
  };
}

/** Fast geometric PSF used inside optimisation loops. */
export function fastPsf(p: PowerVector, pupilRmm: number): { kernel: Float64Array; K: number } {
  const r = pupilRmm * 1e-3;
  const blurRad = Math.max(maxMeridian(p) * 2 * r, 1.2 * RAD_PER_PX);
  const res = geometricPsf(p, r, blurRad, 64);
  return { kernel: res.kernel, K: res.K };
}

function splat(k: Float64Array, K: number, x: number, y: number, w: number) {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  if (x0 < 0 || y0 < 0 || x0 + 1 >= K || y0 + 1 >= K) return;
  k[y0 * K + x0] += w * (1 - fx) * (1 - fy);
  k[y0 * K + x0 + 1] += w * fx * (1 - fy);
  k[(y0 + 1) * K + x0] += w * (1 - fx) * fy;
  k[(y0 + 1) * K + x0 + 1] += w * fx * fy;
}

function normalise(a: Float64Array) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  if (s > 0) for (let i = 0; i < a.length; i++) a[i] /= s;
}

function peakNormalise(a: Float64Array) {
  let m = 0;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  if (m > 0) for (let i = 0; i < a.length; i++) a[i] /= m;
}

// ---------------------------------------------------------------------------
// Convolution, Wiener pre-distortion, sharpness
// ---------------------------------------------------------------------------

type Spectrum = { re: Float64Array; im: Float64Array; F: number };

/** FFT of an n×n image padded (edge value = border mean) to F×F. */
function imageSpectrum(img: Float64Array, n: number, F: number): Spectrum {
  let border = 0;
  for (let i = 0; i < n; i++) border += img[i] + img[(n - 1) * n + i] + img[i * n] + img[i * n + n - 1];
  border /= 4 * n;
  const re = new Float64Array(F * F).fill(border);
  const im = new Float64Array(F * F);
  const o = (F - n) >> 1;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) re[(y + o) * F + x + o] = img[y * n + x];
  fft2d(re, im, F);
  return { re, im, F };
}

function kernelSpectrum(kernel: Float64Array, K: number, F: number): Spectrum {
  const re = new Float64Array(F * F);
  const im = new Float64Array(F * F);
  const h = K >> 1;
  for (let y = 0; y < K; y++)
    for (let x = 0; x < K; x++)
      re[((y - h + F) % F) * F + ((x - h + F) % F)] += kernel[y * K + x];
  fft2d(re, im, F);
  return { re, im, F };
}

function cropCentre(re: Float64Array, F: number, n: number): Float64Array {
  const out = new Float64Array(n * n);
  const o = (F - n) >> 1;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[y * n + x] = re[(y + o) * F + x + o];
  return out;
}

const specCache = new WeakMap<Float64Array, Map<number, Spectrum>>();
function cachedSpectrum(img: Float64Array, n: number, F: number) {
  let m = specCache.get(img);
  if (!m) { m = new Map(); specCache.set(img, m); }
  let s = m.get(F);
  if (!s) { s = imageSpectrum(img, n, F); m.set(F, s); }
  return s;
}

function padSize(n: number, K: number) {
  let F = 1;
  while (F < n + K) F <<= 1;
  return F;
}

export function convolve(img: Float64Array, n: number, kernel: Float64Array, K: number): Float64Array {
  const F = padSize(n, K);
  const I = cachedSpectrum(img, n, F);
  const H = kernelSpectrum(kernel, K, F);
  const re = new Float64Array(F * F), im = new Float64Array(F * F);
  for (let i = 0; i < F * F; i++) {
    re[i] = I.re[i] * H.re[i] - I.im[i] * H.im[i];
    im[i] = I.re[i] * H.im[i] + I.im[i] * H.re[i];
  }
  fft2d(re, im, F, true);
  return cropCentre(re, F, n);
}

/**
 * Wiener pre-distortion (eyesim.predistort): P = F⁻¹[ I·H* / (|H|² + k) ], then clipped to
 * the displayable range [0, 1]. Returns both the display image and what the eye perceives.
 */
export function predistort(
  img: Float64Array, n: number, kernel: Float64Array, K: number, k = 0.01,
): { display: Float64Array; perceived: Float64Array } {
  const F = padSize(n, K);
  const I = cachedSpectrum(img, n, F);
  const H = kernelSpectrum(kernel, K, F);
  const re = new Float64Array(F * F), im = new Float64Array(F * F);
  for (let i = 0; i < F * F; i++) {
    const hr = H.re[i], hi = H.im[i];
    const d = hr * hr + hi * hi + k;
    re[i] = (I.re[i] * hr + I.im[i] * hi) / d;
    im[i] = (I.im[i] * hr - I.re[i] * hi) / d;
  }
  fft2d(re, im, F, true);
  const display = cropCentre(re, F, n);
  for (let i = 0; i < display.length; i++) display[i] = Math.min(1, Math.max(0, display[i]));
  const perceived = convolve(display, n, kernel, K);
  return { display, perceived };
}

/** Mean squared gradient — the sharpness objective used by eyesim.optimize. */
export function sharpness(img: Float64Array, n: number): number {
  let s = 0;
  for (let y = 1; y < n - 1; y++) {
    for (let x = 1; x < n - 1; x++) {
      const gx = img[y * n + x + 1] - img[y * n + x - 1];
      const gy = img[(y + 1) * n + x] - img[(y - 1) * n + x];
      s += gx * gx + gy * gy;
    }
  }
  return s / ((n - 2) * (n - 2));
}

// ---------------------------------------------------------------------------
// Eccentric photorefraction (eyesim.photorefraction.render_single)
// ---------------------------------------------------------------------------

export type PhotorefractionGeometry = {
  Dmm: number;
  eccMm: number;
  pupilRmm: number;
  sigmaMm: number;
};

export const DEFAULT_GEOMETRY: PhotorefractionGeometry = {
  Dmm: 1000, eccMm: 10, pupilRmm: 3, sigmaMm: 3,
};

export function renderPhotorefraction(
  p: PowerVector, ex: number, ey: number, g: PhotorefractionGeometry, n = 96,
): Float64Array {
  const out = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const y = -g.pupilRmm + (2 * g.pupilRmm * j) / (n - 1);
    for (let i = 0; i < n; i++) {
      const x = -g.pupilRmm + (2 * g.pupilRmm * i) / (n - 1);
      if (x * x + y * y > g.pupilRmm * g.pupilRmm) continue;
      const sx = 2 * ((p.M + p.J0) * x + p.J45 * y) * 1e-3;
      const sy = 2 * ((p.M - p.J0) * y + p.J45 * x) * 1e-3;
      const X = sx * g.Dmm - ex;
      const Y = sy * g.Dmm - ey;
      const L = Math.exp(-(X * X + Y * Y) / (2 * g.sigmaMm * g.sigmaMm));
      out[j * n + i] = L * 0.9 + 0.05;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Closed-loop optimiser (eyesim.optimize.run_closed_loop), step-wise so it can animate
// ---------------------------------------------------------------------------

export type LoopEval = {
  phase: "scan" | "refine";
  correction: PowerVector;
  residual: PowerVector;
  score: number;
};

/**
 * Generator form of the two-stage search: coarse M scan (−5…+5 D, 0.5 D) then
 * Nelder–Mead on (M, J0, J45). Yields one objective evaluation at a time.
 * With `warmStart`, the scan is skipped and Nelder–Mead starts at that point
 * (the pipeline's CNN-seeded refinement).
 */
export function* closedLoop(
  eye: PowerVector,
  target: Float64Array,
  n: number,
  pupilRmm = 2,
  warmStart?: PowerVector,
  maxEvals = 140,
): Generator<LoopEval, PowerVector> {
  const evaluate = (c: PowerVector, phase: LoopEval["phase"]): LoopEval => {
    const residual = pvAdd(eye, c);
    const { kernel, K } = fastPsf(residual, pupilRmm);
    const score = sharpness(convolve(target, n, kernel, K), n);
    return { phase, correction: c, residual, score };
  };

  let start: PowerVector;
  if (warmStart) {
    start = warmStart;
  } else {
    let best: LoopEval | null = null;
    for (let m = -5; m <= 5.0001; m += 0.5) {
      const e = evaluate({ M: m, J0: 0, J45: 0 }, "scan");
      yield e;
      if (!best || e.score > best.score) best = e;
    }
    start = best!.correction;
  }

  // Nelder–Mead (maximise score).
  const toV = (p: PowerVector) => [p.M, p.J0, p.J45];
  const toP = (v: number[]): PowerVector => ({ M: v[0], J0: v[1], J45: v[2] });
  const x0 = toV(start);
  const simplex: { v: number[]; f: number }[] = [];
  const pts = [x0, [x0[0] + 0.5, x0[1], x0[2]], [x0[0], x0[1] + 0.5, x0[2]], [x0[0], x0[1], x0[2] + 0.5]];
  for (const v of pts) {
    const e = evaluate(toP(v), "refine");
    yield e;
    simplex.push({ v, f: -e.score });
  }
  let evals = pts.length;
  const f = function* (v: number[]) {
    const e = evaluate(toP(v), "refine");
    evals++;
    yield e;
    return -e.score;
  };
  while (evals < maxEvals) {
    simplex.sort((a, b) => a.f - b.f);
    const size = Math.max(...simplex.slice(1).map((s) => Math.hypot(...s.v.map((x, i) => x - simplex[0].v[i]))));
    if (size < 0.004) break;
    const cen = [0, 1, 2].map((i) => (simplex[0].v[i] + simplex[1].v[i] + simplex[2].v[i]) / 3);
    const worst = simplex[3];
    const refl = cen.map((c, i) => c + (c - worst.v[i]));
    const fr = yield* f(refl);
    if (fr < simplex[0].f) {
      const exp = cen.map((c, i) => c + 2 * (c - worst.v[i]));
      const fe = yield* f(exp);
      simplex[3] = fe < fr ? { v: exp, f: fe } : { v: refl, f: fr };
    } else if (fr < simplex[2].f) {
      simplex[3] = { v: refl, f: fr };
    } else {
      const con = cen.map((c, i) => c + 0.5 * (worst.v[i] - c));
      const fc = yield* f(con);
      if (fc < worst.f) {
        simplex[3] = { v: con, f: fc };
      } else {
        for (let s = 1; s < 4; s++) {
          const v = simplex[s].v.map((x, i) => simplex[0].v[i] + 0.5 * (x - simplex[0].v[i]));
          simplex[s] = { v, f: yield* f(v) };
        }
      }
    }
  }
  simplex.sort((a, b) => a.f - b.f);
  return toP(simplex[0].v);
}

/** Deterministic PRNG so "simulated CNN" draws are reproducible per eye. */
export function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-12), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
