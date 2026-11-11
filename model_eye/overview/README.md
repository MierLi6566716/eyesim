# Model-eye results: overview

How well the photorefraction camera and the PupilNet network read a real optical eye whose
refraction is known exactly.

Regenerate the figures with

    /opt/anaconda3/bin/python model_eye/overview/make_figures.py

(after `scripts/model_eye_report.py model_eye/sessions/*/ --reuse-clicks` has filled `model_eye/results/`).

---

## The short version

- **At 37 cm with a +4 or +5 D trial lens, the network reads the model eye correctly:**
  - average error 0.20 D over 26 settings from −6 to −10 D
  - no bias
  - after the eye was taken off the table and put back, the readings repeated to about 0.2 D
- **The network was trained only on simulated photos.** It never saw a real photo.
- **With the strong +8 D lens, and at 25–30 cm, it reads 0.6–1.0 D too weak.** That is not solved yet.
- **The simulator gets the physics right** (fig. 2, fig. 3): the crescent appears on the correct side and grows
  the same way. Real crescents are softer and shorter top-to-bottom.

---

## What was tested

- **Eye:** XSJ-501 schematic eye. Refraction set on its scale (+4 to −5 D, 0.25 D steps), 5 or 7 mm pupil.
  Trial lenses (+4, +5, +6, +8 D) in front of it reach −13 D.
- **Camera:** Raspberry Pi 4, IMX708 NoIR, two 850 nm LEDs 10 mm from the lens (H to the right, V above).
- **Photos:** 10 blocks, **170 settings × 20 shots**, each shot an H, a V and a dark frame.

| Block | Distance | Lens | Pupil | Settings |
|---|---|---|---|---|
| A | 37 cm | +4 | 7 mm | 21 |
| B | 37 cm | +8 | 7 mm | 11 |
| C | 37 cm | none | 7 mm | 37 |
| D1 / D2 | 37 cm | none / +5 | 5 mm | 19 / 11 |
| E1 / E2 | 30 cm | none / +5 | 7 mm | 19 / 11 |
| F1 / F2 | 25 cm | none / +6 | 7 mm | 19 / 11 |
| R | 37 cm | +4 | 7 mm | 11 (A repeated after re-placing the eye) |

- **Network:** `checkpoints/wide_m_only/best.pt`, the best of those tested (see "Which network").
- **Error:** network reading minus dial setting, with no corrections.
- **Which settings count:** only those where the photo shows a crescent. In the blind band there is nothing to
  measure, and the trial lens is there to move the eye out of it.

---

## Results (`tables/metrics.csv`)

| Block | Crescent settings | Average error | Mean error | Within 0.5 D |
|---|---|---|---|---|
| **A** 37 cm, +4 | 14 | **0.23 D** | +0.09 D | **100 %** |
| **D2** 37 cm, +5, 5 mm | 5 | **0.07 D** | −0.01 D | **100 %** |
| **R** 37 cm, +4, re-placed | 7 | **0.26 D** | +0.23 D | 86 % |
| B 37 cm, +8 | 11 | 0.90 D | +0.90 D | 18 % |
| E2 30 cm, +5 | 6 | 0.66 D | +0.57 D | 33 % |
| F2 25 cm, +6 | 4 | 1.01 D | +1.01 D | 0 % |

Positive = reads too weak (less myopic than set).

**Repeatability (R vs A):** within 0.22 D at 6 of 7 settings. The 7th sits at the blind-band edge and differs by 0.79 D.

---

## Figures

1. **`fig1_accuracy.png`**: network reading vs the true refraction, one panel per block.
2. **`fig2_real_vs_simulated.png`**:
   - Real photos next to the simulator's photo at the same refraction.
   - Same crop and scaling the network sees.
   - The crescent's side and growth match.
   - Real crescents are softer and shorter top-to-bottom.
3. **`fig3_physics.png`**: why it works.
   - (a) The crescent appearing and growing with myopia.
   - (b) Where the light sits in the pupil vs refraction, real vs simulator.
4. **`fig4_literature.png`**: published deep-learning refraction errors, for scale (see below).

---

## Which network

On all 62 crescent settings:

| Network | Range trained | Average error | Within 0.5 D |
|---|---|---|---|
| **`wide_m_only`** (used here) | −12 to +6 D, 25–100 cm | **0.51 D** | **53 %** |
| `v3_m_only` (retrained 10 Nov) | −16 to +8 D, 20–100 cm, lens reflections | 0.54 D | 47 % |

- The retrained network is better at what it was built for: strong myopia (B: 0.90 → 0.69 D) and 25 cm (1.01 → 0.65 D).
- It is worse at ±4–5 D (A: 0.23 → 0.37 D; D2: 0.07 → 0.44 D).
- Averaging the two (0.50 D) gains nothing.
- Per block: `tables/network_comparison.csv`.

---

## What we checked and ruled out

These were candidate explanations for the remaining error. Each was tested on the photos and rejected.

- **LED distance 11.5 mm instead of 10 mm.**
  - The blind-band edges in the photos fit 11.5 mm best.
  - But telling the network 11.5 mm made every reading worse: average 0.51 → 0.63 D.
  - The edges are the weaker evidence. A faint first crescent is lost in real noise, which widens the band
    whatever the geometry.
  - **10 mm, as built, is right.** The photos also rule out 5 mm: that would give a band about 2 D wide, not 8.
- **Trial lens magnifying the pupil.** Correcting for it made every block worse.
- **Trial lens modelled as a separate element.** To first order this is the same as the magnification
  correction above.

**Still unexplained:** at the same total power (−8.3 D), the +4 lens reads correctly and the +8 lens reads 1 D weak.

---

## Two things specific to this model eye

1. **The dark circle in the photos is not the pupil.** It is about 1.8× the set aperture, magnified by the model eye's
   front lens. The analysis uses the set 5/7 mm size.
2. **The aperture is not centred in that dark circle** (up to 2.6 mm off for the 5 mm hole). The analysis finds the
   aperture from where the crescents appear across a block. This matters because a 0.5 mm centring error moves the
   reading by about 0.3 D.

A human eye has neither problem, because the iris edge *is* the pupil.

---

## Comparison with published work

| Study | Eyes | Images | Reference | Result |
|---|---|---|---|---|
| **This work** | one schematic eye | IR photorefraction, 2 LEDs, 37 cm; network trained on **simulated photos only** | dial | average error 0.20 D (A, D2, R) |
| Xu et al. 2022 [1] | human, 18–56 y | NIR photorefraction, 6 LED directions, 1 m; 6,074 image groups | autorefractor | 0.18 D (spherical equivalent) |
| Varadarajan et al. 2018 [2] | human | retina photos, 226,870 for training | autorefraction / subjective | 0.56 D / 0.91 D |
| Chun et al. 2020 [3] | children, mean 4.3 y | smartphone photorefraction, 1 m; 305 images | cycloplegic refraction | 81.6 % correct into 7 classes |
| Linde et al. 2023 [4] | adults | IR red-reflex images | autorefractor | 75 % best class accuracy |

**This is not like-for-like.** A model eye has no accommodation, eyelids or tear film, and its refraction is known
exactly. Human studies measure thousands of people against a reference that is itself only good to ±0.25–0.5 D.

What it does show: a network that **never saw a real photo** reaches, on a real optical eye at its design distance,
the error range that published networks reach after training on thousands of real photos.

[1] Xu D, Ding S, Zheng T, Zhu X, Gu Z, Ye B, Fu W. Deep learning for predicting refractive error from multiple
photorefraction images. *BioMedical Engineering OnLine* 2022. https://doi.org/10.1186/s12938-022-01025-3

[2] Varadarajan AV, Poplin R, Blumer K, et al. Deep learning for predicting refractive error from retinal fundus
images. *Invest Ophthalmol Vis Sci* 2018;59:2861–2868. https://doi.org/10.1167/iovs.18-23887

[3] Chun J, Kim Y, Shin KY, et al. Deep learning–based prediction of refractive error using photorefraction images
captured by a smartphone. *JMIR Med Inform* 2020;8(5):e16225. https://doi.org/10.2196/16225

[4] Linde G, Chalakkal R, Zhou L, et al. Automatic refractive error estimation using deep learning-based analysis of
red reflex images. *Diagnostics* 2023;13(17):2810. https://doi.org/10.3390/diagnostics13172810
