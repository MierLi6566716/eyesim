# EyeSim — Full Technical Walkthrough

*A companion document for presenting the codebase to faculty. Covers every
module: what it computes, the physics/math behind it, how it connects to
the rest of the system, and where the current results and known limitations
stand as of this write-up.*

---

## 0. One-paragraph pitch

EyeSim simulates, end to end, a wearable device that would measure a
person's eyeglass prescription with a camera, drive a tunable lens to
correct it, and re-measure until it converges — with no optometrist. Nothing
in this repository touches real hardware yet. Every optical element (the
eye's blur, the sensor, the corrective lens) is a physics simulation. The
purpose of doing it this way is that the **control algorithm** (how do you
turn a picture of a pupil into a lens command, and how do you know when
you're done?) is the actual hard problem, and it can be fully developed,
tested, and validated in software before a single physical part is ordered.

## 1. Repository map

```
eyesim/
  optics.py           physics core: prescription -> blur (Steps 1)
  predistort.py        Wiener deconvolution pre-distortion display (Step 2)
  optimize.py           closed-loop sharpness optimizer (Step 3)
  photorefraction.py    synthetic IR sensor image generator (Step 4 data)
  cnn.py                 PupilNet: CNN sensor -> prescription (Step 4 model)
  pipeline.py            wires all of the above into one control loop (Step 4.5)
scripts/
  generate_dataset.py    bulk-generates labeled training data
  train.py                trains PupilNet
  evaluate.py              reports accuracy on held-out test set
examples/
  simulate.py, visualize_pipeline.py     static demo figures (Steps 1-4)
  full_pipeline.py                         storyboard + cohort statistics for the full loop
  dashboard.py                              interactive slider-driven live demo
tests/                    pytest suite, 27 tests, all physics + pipeline paths
```

Every number quoted below (MAE, sharpness improvement dB, test counts) is a
live measurement from this exact codebase, not a projection.

---

## 2. `eyesim/optics.py` — the eye's blur

**What a prescription is.** A refractive prescription is written clinically
as (Sphere, Cylinder, Axis) — e.g. `-2.25 / -0.75 / 30°`. Sphere is the
overall focusing error (negative = nearsighted / myopic, positive =
farsighted / hyperopic). Cylinder + Axis describe astigmatism: the eye
focuses two perpendicular meridians differently, and axis says which
direction is which.

**Power vectors.** Axis is a bad number for math — 179° and 1° are almost
the same physically, but numerically they're 178° apart (the
"axis-wrapping" problem). The whole codebase instead uses the **Thibos power
vector** `(M, J0, J45)`, which has no discontinuity:

```
M   = S + C/2                    (spherical equivalent)
J0  = -(C/2) cos(2*theta)        (with-the-rule / against-the-rule astigmatism)
J45 = -(C/2) sin(2*theta)        (oblique astigmatism)
```

`prescription_to_power_vector()` does this conversion; `optimize.py`'s
`power_vector_to_prescription()` does the inverse. Every internal
computation (the optimizer's search space, the CNN's output, the pipeline's
control loop) works in `(M, J0, J45)` and only converts to clinical
`(S, C, axis)` at the input/output boundary.

**From dioptres to a wavefront.** A prescription is a statement about how
much a wavefront of light is bent. `power_vector_to_zernike()` converts
`(M, J0, J45)` in dioptres into three **Zernike coefficients** — defocus and
two astigmatism modes — in metres of optical path difference, given the
eye's pupil radius. `wavefront()` then builds the actual 2-D wavefront
aberration map `W(x, y)` over a unit disk (the pupil), as a weighted sum of
the three Zernike polynomials.

**From wavefront to blur — Fourier optics.** This is the physical core of
the whole project. Light passing through the pupil picks up a phase delay
proportional to the wavefront error: `pupil_function = aperture * exp(i * k
* W)`, where `k = 2π/λ`. The **point-spread function** (PSF — literally
"what a single point of light looks like after the eye blurs it") is the
squared magnitude of the Fourier transform of that pupil function:

```
PSF = |FFT(pupil_function)|²
```

This is standard Fraunhofer diffraction theory — the same math used to
design telescopes and microscopes, applied here in reverse to characterize
blur instead of resolution. `psf_from_prescription()` computes this on an
`n x n` grid and optionally crops it. `blur_image()` then convolves any
input image with the PSF via FFT (`scipy.signal.fftconvolve`) to produce
"what the eye sees." `simulate_vision()` chains these two steps.

**A bug I found and fixed here.** `psf_from_prescription()` accepts a
`psf_crop` parameter that windows the PSF to a smaller array (for speed).
The original default of `psf_crop=32` (used throughout Step 3's README
benchmarks and the Step 1 example figure) silently discards a large
fraction of the PSF's energy at moderate-to-high dioptre values — as much as
95% of the energy at 4 D. Cropping renormalizes what remains to sum to 1,
so the simulation runs without error but is physically wrong: it understates
blur severity and biases anything computed downstream (sharpness values,
CNN training labels' implied images, etc.) I added `min_psf_crop(M_max,
pupil_radius_mm)`, which computes the minimum crop that captures the
geometric blur disc for a given power, derived from the fact that the blur
disc diameter in pixels is `|M| * (2*pupil_radius_m)² / wavelength_m`,
independent of the FFT grid size. The full pipeline (`pipeline.py`) now
uses `grid=256, psf_crop=96`, verified by test to keep >98% of PSF energy
up to 4 D. This is exactly the kind of bug that's invisible in isolated unit
tests (each function runs, returns a normalized array, tests pass) and only
shows up when you look at the physical correctness of the number, which is
why the `test_min_psf_crop_contains_energy` test explicitly measures energy
fraction rather than just checking that the array shape is right.

---

## 3. `eyesim/predistort.py` — pre-distortion display

Separate from the corrective-lens idea: if you know a viewer's blur kernel
`h`, you can pre-distort an image `d` so that after the eye blurs it
(`retinal image = d ⊛ h`), the *perceived* image equals your intended sharp
target. This is solved with **Wiener deconvolution**, the standard
regularized inverse filter:

```
D(f) = H*(f) / (|H(f)|² + K)  ·  T(f)
```

where `H` is the PSF's Fourier transform, `T` is the target image's Fourier
transform, and `K` is a small regularization constant that prevents
amplifying noise at frequencies where `H` is near zero (the classic
ill-posed-inverse-problem fix). `wiener_deconvolve()` implements this;
`predistort()` wraps it with the PSF computation; `simulate_display_chain()`
produces all four panels shown in the Step 2 figure (original → predistorted
→ what the eye sees corrected → what the eye sees uncorrected).

One subtlety documented in the code: `_fft_blur()` and `wiener_deconvolve()`
both use **circular** convolution (via `np.fft.fft2`/`ifft2` directly,
rather than `fftconvolve`'s linear convolution), so that the forward blur
and its inverse are exact algebraic inverses of each other up to the noise
term `K`. This matters because linear and circular convolution disagree at
image edges; using two different conventions for the forward and inverse
step would introduce an artifact that has nothing to do with optics.

This step is a complete, self-contained result — it doesn't need any
hardware at all, since the "device" here is just a display.

---

## 4. `eyesim/optimize.py` — closed-loop optimizer

This is the algorithmic core of the actual device: given a scene and an
unknown eye, find the lens correction that makes the retinal image sharpest,
without ever being told the prescription.

**Sharpness metric.** `sharpness(image)` returns the mean squared Laplacian
of the image — a standard focus metric from autofocus systems (blurry images
have low high-frequency / edge energy; sharp images have high energy).

**Two-stage search, and why.** `run_closed_loop()` doesn't hand the
sharpness objective straight to a generic optimizer, because the sharpness
landscape over `(M, J0, J45)` is deceptive: it's nearly flat for several
dioptres around a wrong guess, then rises sharply near the truth, and
directional test images create spurious local maxima aligned with whatever
axis the image itself is oriented along.

1. **Coarse M-only scan** (`coarse_scan=True`, the default): evaluate
   sharpness at 21 candidate `M` values from -5 to +5 D in 0.5 D steps, with
   astigmatism held at zero. This finds the right basin cheaply and isn't
   fooled by astigmatic local optima because it never explores
   astigmatism.
2. **Nelder-Mead refinement**: a gradient-free simplex search (Nelder &
   Mead, 1965 — the right family of algorithm here because on real hardware
   you can't take a derivative of "how sharp does this look," you can only
   *evaluate* it) starting from the best coarse `M`, refining all three
   power-vector components together.

I added two parameters this session: `coarse_scan` (skip stage 1 when a
good starting guess already exists — e.g. from the CNN) and `simplex_size`
(shrink the initial simplex when warm-starting close to the optimum). These
make the optimizer usable as the *refinement* stage of the full pipeline
(§7) rather than always running the algorithm cold.

**Reported result** (from the README, on a 128×128 isotropic test image):
recovers a `-1.5/-0.5/30°` prescription exactly, with +26.1 dB sharpness
improvement, in ~120 total function evaluations.

---

## 5. `eyesim/photorefraction.py` — synthetic sensor images

This module simulates the actual *sensing* technique the hardware will use:
**eccentric photorefraction** (Kaakinen 1979; also called the "Brückner
test" family). An IR LED sits slightly off-axis from a camera. Light enters
the eye, bounces off the retina, and exits back through the pupil — but only
the part of the pupil whose exit ray happens to land back on the camera
aperture is bright; the rest is dark. Where in the pupil this bright
crescent appears, and how wide/asymmetric it is, encodes the eye's
refractive error.

**The physics implemented.** For a pupil point `(x, y)` and a source at
eccentricity `(ex, ey)`, both in millimetres, `render_single()` computes the
**double-pass wavefront gradient** — the local tilt of the wavefront at that
point, doubled because light traverses the eye's optics twice (in and back
out):

```
slope_x = 2 * ((M + J0)*x + J45*y) * 1e-3     [rad]
slope_y = 2 * ((M - J0)*y + J45*x) * 1e-3
```

That slope, multiplied by the camera distance `D_mm`, tells you where a ray
originating at pupil point `(x,y)` lands at the camera plane. Subtracting
the LED's own position `(ex, ey)` and applying a Gaussian falloff models the
camera's finite aperture — rays landing far from the LED position are dim,
rays landing on top of it are bright. This produces the crescent.

`render_multichannel()` renders two images per eye — one with the LED
offset horizontally, one vertically — because a single LED position can't
disambiguate `J0` from `J45`; you need two independent measurement
directions to recover the full 3-parameter power vector, exactly as
described in the docstring and matching the classical two-meridian
photorefraction literature.

**Default hardware geometry** baked into the module (used both for
synthetic-data generation and as the target the physical rig should match,
§ hardware doc): camera distance `D = 1000 mm`, LED eccentricity `10 mm`,
pupil radius `3 mm`, camera aperture Gaussian `σ = 3 mm`.

**Domain randomisation.** `generate_dataset.py` (§8) varies `D_mm` (±100
mm), eccentricity (±2 mm), pupil radius (±0.5 mm), and noise level per
sample when `--augment` is set — a standard sim-to-real gap mitigation
technique. I found a real limitation here (§11) that this randomisation, as
currently implemented, has a side effect worth understanding before more
training compute is spent on it.

`sample_prescription()` draws population-realistic prescriptions from
published epidemiological distributions (Vitale 2009, Kempen 2004): `M ~
Normal(-0.5, 2.0)` clipped to [-6, 4] D, astigmatic components `~
Normal(0, 0.4)` clipped to [-2, 2] D.

---

## 6. `eyesim/cnn.py` — PupilNet

**Architecture.** A small (~450k parameter) depthwise-separable CNN: a
strided stem, four depthwise-separable blocks (32→64→128→256 channels, each
halving spatial resolution), global average pooling, and a 2-layer
regression head with dropout — outputting `(M, J0, J45)` directly (not
`(S, C, axis)`, again to avoid the axis-wrapping discontinuity, and because
this is exactly the search space `optimize.py` operates in, so the CNN's
output plugs directly into the closed-loop refinement with no conversion).
The design is explicitly chosen to be INT8-quantization-friendly (ReLU only,
batch norm not layer norm, linear output head) for eventual edge deployment.

**Loss & metrics.** MSE in power-vector space (`power_vector_loss`), with
per-component and combined MAE reporting (`power_vector_mae`). Power-vector
MSE is used as a *proxy* for clinical accuracy — the actual clinical
question ("is S/C/axis within one optometry step?") is computed separately
in `scripts/evaluate.py`.

**Dataset class.** `PrescriptionDataset` loads the `.npz` files produced by
`generate_dataset.py` and applies light augmentation (intensity jitter,
additive noise) when `augment=True`, simulating exposure and sensor-noise
variation.

---

## 7. `eyesim/pipeline.py` — the full control loop (new this session)

Everything above existed as separate demonstrations. This module is what
was missing: it actually runs the device's control flow end to end for one
simulated eye.

```
[eye, unknown Rx]
     -> render n photorefraction frames (photorefraction.py)
     -> average frames, feed to CNN (cnn.py)                      = one-shot estimate
     -> command the lens with -(CNN estimate)                       (open loop)
     -> measure residual sharpness, refine with Nelder-Mead         (closed loop,
        warm-started at the CNN estimate, coarse_scan=False)         optimize.py)
     -> final residual error
```

`run_full_pipeline()` returns every intermediate artifact (frames, CNN
estimate, residual PSF at each stage, sharpness at each stage, the full
optimizer history) so it can be visualized or logged. `load_estimator()`
wraps a trained `checkpoints/best.pt` into a plain callable
`image -> (M, J0, J45)`, so the pipeline function itself has no PyTorch
dependency and can run with a mock estimator (useful for the test suite and
for demoing before a checkpoint exists).

**Why warm-starting matters here, concretely**: on a single test eye
(`S=-2.25, C=-0.75, axis=30°`), the CNN's one-shot estimate alone reduces
the residual power-vector error from 2.65 D to 0.03 D. The subsequent
closed-loop refinement (103 evaluations) then drives it to ~0.00 D. Without
the CNN's warm start, the same refinement from a cold `(0,0,0)` guess would
need the full coarse-scan-plus-refinement procedure and more evaluations to
find the right basin — this is the concrete justification for combining
sensing and control rather than treating Step 4 (CNN) and Step 3
(optimizer) as alternatives.

**Cohort result** (20 random eyes drawn from the population-realistic
distribution): median residual error goes 1.34 D (uncorrected) → 0.08 D
(after the CNN's one-shot correction, 95% of eyes already within the 0.25 D
clinical threshold) → ~0.00 D (after refinement, 100% within threshold).

---

## 8. `scripts/` — dataset generation, training, evaluation

- **`generate_dataset.py`**: draws `N` prescriptions from
  `sample_prescription()`, renders each through `render_multichannel()`
  (with domain randomization if `--augment`), and writes an 80/10/10
  train/val/test split as compressed `.npz`. The current dataset on disk is
  100,000 samples total (80k/10k/10k), 64×64×2 images, ~2.5 GB.
- **`train.py`**: AdamW (lr 1e-3, weight decay 1e-4), cosine LR schedule,
  100 epochs, MSE loss, saves the best-val-loss checkpoint to
  `checkpoints/best.pt`. Runs on CUDA/MPS/CPU automatically.
- **`evaluate.py`**: loads a checkpoint, runs inference on the held-out test
  set, reports power-vector and clinical-space errors, and the fraction of
  predictions within the clinical threshold (`|M error| < 0.25 D`, `|J
  error| < 0.12 D`).

**Current measured accuracy** (test set, 10,000 samples, checkpoint from
epoch 40 of 100 — this is genuinely the best checkpoint, val loss plateaued
by ~epoch 15, see `checkpoints/train_log.csv`):

| Component | MAE (D) | RMS (D) | P95 (D) | % within clinical threshold |
|---|---|---|---|---|
| M | 0.184 | 0.263 | 0.551 | 74.5% |
| J0 | 0.046 | 0.068 | 0.139 | 92.8% |
| J45 | 0.050 | 0.074 | 0.151 | 91.3% |

Clinical-space: S MAE = 0.187 D, C MAE = 0.131 D, axis MAE = 2.46°.

---

## 9. `examples/` — demos and the new visualization tools

- **`simulate.py`**: CLI for Step 1 (single prescription → blur).
- **`visualize_pipeline.py`**: generates the four static per-step figures
  used in the proposal (`step1_eye_blur.png` … `step4_photorefraction.png`).
- **`full_pipeline.py`** *(new)*: runs `run_full_pipeline()` and produces
  (a) a one-eye "storyboard" figure — three rows (SEE / SENSE / CORRECT &
  REFINE) walking through every stage for one prescription, and (b) a cohort
  figure — residual-error distribution and clinical-threshold pass rate
  across N random eyes (box in §7's numbers). Supports `--mock` to run
  without a trained checkpoint.
- **`dashboard.py`** *(new)*: a live, interactive matplotlib GUI. Sliders
  for `S`, `C`, `axis`, and sensor noise update the eye's blur, the rendered
  IR pupil images, and the CNN's estimate in real time as you drag; a button
  runs the closed-loop refinement and plots its convergence. This is the
  single best artifact for a live demo to the professor — it makes the
  entire pipeline's behavior explorable in real time rather than reading
  static numbers.

---

## 10. Test suite

27 tests across `tests/test_optics.py`, `test_predistort.py`,
`test_optimize.py`, and the new `tests/test_pipeline.py`. The new pipeline
tests use a perfect (ground-truth) mock estimator to check the refinement
path is a no-op when the CNN is already exact, and a deliberately biased
mock estimator to check the refinement actually removes an injected error —
i.e. the tests verify the *control logic*, not just that functions run
without throwing. `test_min_psf_crop_contains_energy` verifies the PSF
energy-truncation fix (§2) directly, by measuring energy fraction rather
than checking output shape.

---

## 11. Known limitations, in order of how much they currently limit accuracy

**1. The M·D degeneracy (the current bottleneck on M accuracy).** I proved
this directly: `render_single(M=2.0 D, D_mm=1000)` and `render_single(M=2.2
D, D_mm=909)` produce **pixel-identical** images. The reason is visible
directly in the slope formula in §5 — `slope * D_mm` only ever appears as a
product, so the sensor image is a function of `M · D`, never `M` and `D`
separately. `generate_dataset.py --augment` varies `D_mm` by ±10% without
telling the network what the actual value was for that sample, which makes
`M` fundamentally unrecoverable to better than roughly `|M| × 0.1`
precision from the image alone — and that is close to exactly the error
floor observed in training (`train_log.csv` plateaus around epoch 15; the
`M` MAE is 4× the `J0`/`J45` MAE, and the test-scatter plot shows the M
error fanning out proportional to `|M|`, the signature of a multiplicative,
not additive, ambiguity). This is not a training or architecture problem —
no amount of epochs or data fixes an information-theoretic ambiguity. The
fix is either (a) give the network the camera distance as an auxiliary
input (known on real hardware — this is the direction that connects
directly to the hardware proposal, since the physical rig will *know* its
own working distance exactly) or (b) have the network predict `M·D` and
divide by a separately measured `D`.

**2. PSF truncation (found and fixed this session, §2).** Affects any
number computed with the old `psf_crop=32` default — worth regenerating the
Step 1/3 example figures and README benchmark table under the corrected
window before using them in a presentation.

**3. Sim-to-real gap in the sensor model.** `render_single()` uses a smooth
Gaussian falloff for the camera aperture response. Real eccentric
photorefraction crescents have harder edges, a corneal glint (Purkinje
reflection), and are affected by higher-order aberrations, eyelid
occlusion, and gaze angle — none of which are modeled. This is the gap the
hardware rig described in the companion proposal exists to close.

**4. No higher-order aberrations.** Only defocus + 2nd-order astigmatism
(Zernike modes 2-4) are modeled anywhere in the pipeline. Real eyes have
coma, trefoil, spherical aberration, etc. — small for most people, but a
real source of residual error the CNN will encounter on real images that it
has never seen in training.

---

## 12. Where this leaves Step 5

The roadmap's Step 5 ("benchtop hardware... close the loop on a bench") has
two genuinely separable sub-problems that this session's work clarifies:

- **Getting real *labeled* sensor images** to fine-tune PupilNet and measure
  the true sim-to-real gap (limitation 3 above) — this needs an artificial
  eye with a known, controllable prescription. See the companion hardware
  proposal.
- **Getting a real *tunable lens*** to physically test the closed-loop
  optimizer (§4) against real optics and real camera noise instead of
  simulated ones — this is the Alvarez-lens actuator direction already
  described in `proposal/main.tex`.

The hardware proposal below argues these can be partially unified: the same
axial-slider "model eye" rig, because it *is* a physically-controllable
optical system with a known ground truth, can also serve as a first
physical test of the closed-loop control algorithm itself, before the
Alvarez actuator exists.
