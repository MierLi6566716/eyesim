# eyesim

In this project, we want to simulate how a human eye, given a refractive prescription **(Sphere, Cylinder, Axis)** sees the world, and build toward a **closed-loop self-refracting corrector**: ultimately, if possible, a wearable device that measures your refraction, drives a tunable lens to correct it, then re measures and converges, without needing any optometrist and no manul 1 or 2 selections.


**Not a medical device. Not for clinical use.** Research / engineering prototype only.

We currently have **Step 1** of that program: an eye-blur simulator. It is the foundation everything else is tested against.

---

## Why this exists

The three pieces of "glasses that measure and fix your prescription" each already exist in isolation:

- **Sensing** — CNNs predicting (S, C, axis) from eccentric photorefraction images are published and clinically validated (one 2025 system trained on 362k images).
- **Correction** — electronically tunable liquid-crystal glasses are funded and shipping prototypes (IXI raised ~$36M; Morrow, Laclarée, Deep Optics).
- **Pre-distortion displays** — Berkeley/MIT light-field "vision-correcting displays" and inverse-blur deconvolution are well-published and patented.

What is **not** yet a product: a wearable that **closes the full loop** — *senses* refraction objectively, *drives* tunable correction, *re-measures* residual blur, and *self-converges* — driven by measured refractive error rather than by gaze/distance. This integration is a sensing + optimization + control problem.

## The system as a control loop

```
[Sensor: measure refractive error]  ->  [Estimator: signal -> (S,C,axis)]
        ^                                          |
        |                                          v
[Feedback: residual blur]  <-  [Actuator: tunable lens]  <-  [Controller: (S,C,axis) -> voltages]
```


## Roadmap

- [x] **Step 1 — eye-blur simulator** (this release): prescription → Zernike wavefront → PSF → blurred image. Physics validated by tests.
- [ ] **Step 2 — pre-distortion display demo**: deconvolve an image by the eye's PSF so a blurry eye sees it sharp. Pure software, great standalone result.
- [ ] **Step 3 — closed-loop optimizer (in sim)**: maximize an image-sharpness objective over (S,C,axis) / voltage space using gradient-free optimization (Nelder–Mead / Bayesian / CMA-ES). The core IP.
- [ ] **Step 4 — photorefraction estimator**: synthesize eccentric-photorefraction images from the simulator, train a CNN to regress (S,C,axis), validate on real images.
- [ ] **Step 5 — benchtop hardware**: off-the-shelf tunable lens + IR camera, close the loop on a bench (not wearable).
- [ ] **Step 6 — clinical validation** against a real autorefractor (requires a vision-science partner; this is also the regulatory line).
- [ ] **Step 7 — wearable form factor** (last, not first).

## Install

```bash
git clone <your-repo-url> eyesim && cd eyesim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"
```

## Quickstart

```python
from eyesim import simulate_vision
import numpy as np

image = np.random.rand(256, 256)          # any grayscale image in [0,1]
seen, psf = simulate_vision(image, S=-2.0, C=-1.0, theta_deg=45)
```

CLI:

```bash
python examples/simulate.py --demo --S -2.0 --C -1.0 --axis 45
python examples/simulate.py --image photo.png --S -1.5
```

Run the physics tests:

```bash
pytest -q
```

## License

MIT. **Not a medical device.**
