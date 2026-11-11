# Model-eye results: overview

What the photorefraction camera and the PupilNet network do on a real optical eye whose
refraction is known exactly. Four figures, one table, and the open questions.

Regenerate everything here with

    /opt/anaconda3/bin/python model_eye/overview/make_figures.py

(after `scripts/model_eye_report.py model_eye/sessions/*/ --reuse-clicks` has produced
`model_eye/results/`).

---

## What was tested

- **Eye:** XSJ-501 schematic eye. Refraction set on its scale (+4 to −5 D, 0.25 D steps), 3/5/7 mm pupil.
  Trial lenses (+4, +5, +6, +8 D) in front of it extend the range to −13 D.
- **Camera:** Raspberry Pi 4 + IMX708 NoIR, two 850 nm LEDs 10 mm from the lens (H to the right, V above),
  gain 4, 20 ms, raw 10-bit frames.
- **Photos:** 10 blocks, **170 settings × 20 shots** (each shot = H frame, V frame, dark frame), 6 October 2026.

| Block | Distance | Lens | Pupil | Settings | Purpose |
|---|---|---|---|---|---|
| A | 37 cm | +4 | 7 mm | 21 | main accuracy test (eye + lens −4 to −9 D) |
| B | 37 cm | +8 | 7 mm | 11 | strong myopia (−8 to −13 D) |
| C | 37 cm | none | 7 mm | 37 | blind-band map |
| D1 | 37 cm | none | 5 mm | 19 | blind band, small pupil |
| D2 | 37 cm | +5 | 5 mm | 11 | small pupil + lens |
| E1 / E2 | 30 cm | none / +5 | 7 mm | 19 / 11 | distance |
| F1 / F2 | 25 cm | none / +6 | 7 mm | 19 / 11 | distance |
| R | 37 cm | +4 | 7 mm | 11 | A again after taking the eye off the table (repeatability) |

**Error** below is *network reading minus dial setting*, with no correction. The dial is calibrated in visible
light, and the model eye's own 850 nm offset is unknown, so any such offset is inside the error.
Only settings where the photo shows a crescent count. In the blind band there is physically nothing to measure,
and that is what the trial lens is for.

Network: `checkpoints/wide_m_only/best.pt` (trained only on simulated photos, −12 to +6 D, 25–100 cm).

---

## Results

| Block | Distance | Crescent settings | Mean \|error\| | Mean error | Within 0.5 D | Slope |
|---|---|---|---|---|---|---|
| **A** | 37 cm | 14 | **0.23 D** | +0.09 D | **100 %** | 1.01 |
| **D2** | 37 cm | 5 | **0.07 D** | −0.01 D | **100 %** | 0.98 |
| **R** | 37 cm | 7 | **0.26 D** | +0.23 D | 86 % | 1.20 |
| B | 37 cm | 11 | 0.90 D | +0.90 D | 18 % | 0.83 |
| E2 | 30 cm | 6 | 0.66 D | +0.57 D | 33 % | 0.54 |
| F2 | 25 cm | 4 | 1.01 D | +1.01 D | 0 % | 0.43 |

(`data/metrics.csv`. Positive error = reads too weak, i.e. less myopic than set.)

**Repeatability (R vs A, same settings, eye removed and put back):** within 0.22 D at 6 of 7 settings.
The 7th (−2.0) sits right at the blind-band edge and differs by 0.79 D. Mean difference +0.13 D, sd 0.32 D.

**Hyperopic side (no lens):** crescents only from about +2.25 to +3 D up (depending on distance), and the network reads them compressed:
+4.00 reads +3.44 (37 cm), +3.11 (30 cm), +3.46 (25 cm). The same network does this on clean simulated
photos too (+4 → +3.5), so it is the network's top end, not the camera.

### In one paragraph

At the design distance (37 cm) with a moderate trial lens (+4/+5 D), the network reads a real optical eye
from −6 to −9 D **with no bias, a slope of 1.0 and a mean error of 0.07–0.26 D**, and repeats to about 0.2 D
after the eye is removed and replaced, although it was trained only on simulated photos.
With the strong +8 D lens, and at 25–30 cm, it reads 0.6–1.0 D too weak. Those cases are not solved yet (see below).

---

## Figures

**`fig1_accuracy.png`**: network reading against the dial for each lens block. Filled dots: crescent visible.
Open dots: blind band, where the reading is a guess. Blue band: ±0.5 D.

**`fig2_pupil_centring.png`**: why the pupil centre matters.
- (a) Where light actually comes back through the aperture, pooled over two blocks: it fills a disc of exactly the
  set 7 mm size.
- (b) The same photos and network, analysed around the dark disc's centre (old) vs the aperture found from the
  reflex (new).
- (c) Measured cost of moving the centre along the LED axis: 0.5 mm changes the reading by 0.27 D (median, up to
  0.67 D), 1 mm by 0.63 D (median, up to 1.58 D). 2 mm sends readings off the scale.

**`fig3_physics.png`**: why it works.
- (a) The crescent appearing and growing with myopia (block A).
- (b) Crescent position against refraction, real vs simulator: same shape, but the real myopic edge is further out.

**`fig4_literature.png`**: published deep-learning refraction errors, for scale only (see below).

---

## Two things specific to the model eye (and relevant to any real device)

1. **The visible dark disc is not the pupil.** On the XSJ-501 it is about 1.8× the set aperture (the front lens
   magnifies it). Using its size as the pupil shifted readings by about 4 D. The analysis uses the set size; fig. 2a
   confirms that size independently.
2. **The aperture is not centred in the dark disc.** It is up to 8.4, 8.9 px off (≈ 2.6 mm) for the 5 mm hole, and
   2–6 px for the 7 mm hole. Analysed around the disc centre, block D2 read nonsense (mean error 11.8 D); around
   the true aperture it reads 0.07 D.

**How the aperture is found (`scripts/model_eye_report.py`, default):**
- Over a block the crescent moves with the dial, while glints, lens reflections and the dark disc stay put.
  So (max − median) over the block's photos keeps only light that came back through the aperture.
- A circle of the set pupil radius is fitted to the outer edge of that light.
- Blocks taken without moving the eye (A+B+C, D1+D2, E1+E2, F1+F2) are pooled: the no-lens block lights the
  hyperopic side of the aperture and the lens block the myopic side. One-sided light lets the circle slide toward
  the lit side.

**Honesty note.** This method was revised several times while the results were being looked at:
1. dark-disc centre
2. per-block reflex disc
3. per-block edge fit
4. pooled edge fit (final)

The final choice rests on geometry: the pooled light fills a disc of exactly the set radius, with edge points on
all sides. It was not chosen for accuracy. Across those four methods the A, D2 and R results stay good, but
**B, E2 and F2 move by up to about 1 D**. Their accuracy is therefore uncertain, not just poor.

---

## What is still wrong

1. **The +8 lens (block B) reads 0.4–1.75 D too weak, more so the stronger the myopia.**
   At the same total power (−8.3 D), block A (+4 lens) reads it correctly and block B (+8 lens) reads it 1 D weak.
   The eye did not move and the analysis centre is the same, so the difference is tied to the lens. Correcting for
   the lens's pupil magnification made every block *worse*, so that is not it. Part of B (beyond −12 D) is past the
   network's training range.
2. **Closer distances read too weak:** about +0.6 D at 30 cm and +1.0 D at 25 cm, with only 4–6 crescent settings
   each.
3. **The real blind band is wider than simulated, but only on the myopic side.** The real myopic edge is
   1.2–1.4× as far from the null as simulated at every placement:

   | Placement | Myopic edge (real ÷ sim) | Hyperopic edge (real − sim) |
   |---|---|---|
   | 37 cm, 7 mm | 1.38× | +0.5 D |
   | 37 cm, 5 mm | 1.24× | n/a (D1 had almost no crescents) |
   | 30 cm, 7 mm | 1.23× | +0.13 D |
   | 25 cm, 7 mm | 1.32× | +0.37 D |

   Edge = midpoint between the last dark and the first lit setting, using the report's crescent rule.

   **Every myopic-side photo was taken through a trial lens, and every hyperopic-side photo without one.**
   So the leading suspect is how the trial lens is modelled (eye + lens treated as one refraction at the pupil).
   This would fit items 1 and 2 as well. It is not yet shown.

---

## Comparison with published work

| Study | Eyes | Images | Reference | Result |
|---|---|---|---|---|
| **This work**, 37 cm | schematic eye | IR eccentric photorefraction, 2 LEDs; network trained on **simulated photos only** | dial setting | MAE 0.07–0.26 D (A, D2, R); 0.90 D (B) |
| Xu et al. 2022 [1] | human, 18–56 y | NIR eccentric photorefraction, 6 meridians, 1 m; 6,074 image groups | autorefractor (Topcon RM-800) | MAE 0.18 D (spherical equivalent) |
| Varadarajan et al. 2018 [2] | human | retinal fundus photos; 226,870 training images | autorefraction (UK Biobank) / subjective (AREDS) | MAE 0.56 D / 0.91 D |
| Chun et al. 2020 [3] | human children, mean 4.3 y | smartphone eccentric photorefraction, 1 m; 305 images | cycloplegic refraction | 81.6 % correct into 7 classes (no MAE) |
| Linde et al. 2023 [4] | human adults | IR red-reflex images, phone + handheld camera | autorefractor | 75 % best classification accuracy (no MAE) |

**These are not like-for-like.** A schematic eye has no accommodation, no tear film, no eyelids and a perfectly
known refraction; human studies have all of those, plus a reference that is itself ±0.25–0.5 D. Our numbers come
from 26 crescent settings on one eye (37 including block B), not thousands of people. What the comparison does show is that a network trained
**without a single real photo** reaches, on a real optical eye at its design distance, the same error range that
published networks reach after training on thousands of real images. The fair next test is human eyes against
subjective refraction.

[1] Xu D, Ding S, Zheng T, Zhu X, Gu Z, Ye B, Fu W. Deep learning for predicting refractive error from multiple
photorefraction images. *BioMedical Engineering OnLine* 2022. https://doi.org/10.1186/s12938-022-01025-3

[2] Varadarajan AV, Poplin R, Blumer K, et al. Deep learning for predicting refractive error from retinal fundus
images. *Invest Ophthalmol Vis Sci* 2018;59:2861–2868. https://doi.org/10.1167/iovs.18-23887

[3] Chun J, Kim Y, Shin KY, et al. Deep learning–based prediction of refractive error using photorefraction images
captured by a smartphone. *JMIR Med Inform* 2020;8(5):e16225. https://doi.org/10.2196/16225

[4] Linde G, Chalakkal R, Zhou L, et al. Automatic refractive error estimation using deep learning-based analysis
of red reflex images. *Diagnostics* 2023;13(17):2810. https://doi.org/10.3390/diagnostics13172810

---

## Files

- `make_figures.py`: builds everything here from `model_eye/results/`.
- `data/metrics.csv`: the results table.
- `data/disc_centre_*.csv`: the same blocks analysed around the dark-disc centre (for fig. 2b).
- `data/sensitivity.json`: readings with the pupil centre moved by ±0.5/1/2 mm (fig. 2c); 6 sessions from A and B.
