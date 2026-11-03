/**
 * Alvarez lens + actuator model used on the Alvarez page.
 *
 * Plate surface (spherical pair):   t(x, y) = A (x y² + x³/3)
 * Plate surface (astigmatic pair):  t(x, y) = A (x³/3 − x y²)
 * Sliding the two complementary plates by ±d gives a thickness change of
 * −2Ad(x² + y²) (sphere) or −2Ad(x² − y²) (J₀); sliding the astigmatic pair
 * along y gives 4Ad·xy (J₄₅). All three have the same optical gain:
 *
 *     P = 4 (n − 1) A d
 *
 * The actuator numbers are illustrative placeholders for a small voice-coil
 * + flexure stage, not measured values. They're chosen to be physically
 * plausible so that the shape of the response is right.
 */

export const N_PMMA = 1.49;
/** Design gain, dioptres per mm of plate displacement. */
export const GAIN_D_PER_MM = 4;
/** Cubic coefficient A (1/mm²) implied by the gain. */
export const A_PER_MM2 = GAIN_D_PER_MM / (4 * (N_PMMA - 1)) / 1000;

export const ACT = {
  massKg: 0.002,        // moving plate + coil
  springNPerM: 400,     // flexure stiffness k_s
  dampNsPerM: 0.3,
  forceNPerA: 2.0,      // voice-coil force constant k_f
  coilOhm: 8,
  iMaxA: 0.6,
  // controller (10 kHz): feedforward + PD with small integral, derivative on measurement
  kp: 300,              // A/m
  ki: 2000,             // A/(m·s)
  kd: 1.0,              // A·s/m
  hallNoiseM: 0.4e-6,   // position-sensor noise (1σ)
};

/** Steady-state coil current per dioptre of commanded lens power (A/D). */
export const AMPS_PER_DIOPTRE = ACT.springNPerM / (ACT.forceNPerA * GAIN_D_PER_MM * 1000);

export function displacementMm(powerD: number) {
  return powerD / GAIN_D_PER_MM;
}

export type StepTrace = { t: number[]; x: number[]; i: number[] };

/** Simulate one plate axis moving from x0 to x1 (metres). Returns ms, µm, mA samples. */
export function simulateAxis(x0: number, x1: number, seed = 1, horizonS = 0.03): StepTrace {
  const dt = 1e-5, ctrlEvery = 10, t0 = 0.002;
  let x = x0, v = 0, integ = 0, prevX = x0, I = (ACT.springNPerM * x0) / ACT.forceNPerA;
  let s = seed;
  const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  const noise = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-9))) * Math.cos(2 * Math.PI * rnd());
  const out: StepTrace = { t: [], x: [], i: [] };
  const steps = Math.round(horizonS / dt);
  for (let n = 0; n <= steps; n++) {
    const t = n * dt;
    const target = t >= t0 ? x1 : x0;
    if (n % ctrlEvery === 0) {
      const meas = x + noise() * ACT.hallNoiseM;
      const e = target - meas;
      integ += e * dt * ctrlEvery;
      const dxm = (meas - prevX) / (dt * ctrlEvery);
      prevX = meas;
      const ff = (ACT.springNPerM * target) / ACT.forceNPerA;
      I = Math.max(-ACT.iMaxA, Math.min(ACT.iMaxA, ff + ACT.kp * e + ACT.ki * integ - ACT.kd * dxm));
    }
    const a = (ACT.forceNPerA * I - ACT.springNPerM * x - ACT.dampNsPerM * v) / ACT.massKg;
    v += a * dt;
    x += v * dt;
    if (n % 20 === 0) {
      out.t.push(t * 1000);
      out.x.push(x * 1e6);
      out.i.push(I * 1000);
    }
  }
  return out;
}
