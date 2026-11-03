# A Motorized Artificial Eye for CNN Fine-Tuning and Closed-Loop Validation

*Extended proposal, continuing the email to Professor Praneeth. Covers the
physics justification, three build options at increasing cost/capability,
component-level recommendations, calibration procedure, and how the rig
integrates directly with the existing `eyesim` codebase.*

---

## 1. Why we need this, precisely

`eyesim.cnn.PupilNet` is trained entirely on synthetic photorefraction
images (`eyesim.photorefraction.render_multichannel`). On synthetic
held-out data it already reaches clinically useful accuracy (M MAE 0.18 D,
J0/J45 MAE ~0.05 D). But every one of those numbers is a statement about how
well the CNN fits *our simulator*, not about how well it will work on a
*real eye*. The synthetic sensor model uses a smooth Gaussian camera
aperture response; real eccentric photorefraction images have harder
crescent edges, a corneal glint, and aberrations we don't model. There is no
way to measure — let alone close — that gap without real images that carry
a **known, trustworthy ground-truth label**.

A real human eye cannot supply that label cheaply: you'd need a clinical
autorefractor or a trained optometrist for every single calibration frame,
which does not scale to the thousands of labeled images a fine-tuning run
needs. A commercial "model eye" (used for autorefractor calibration in
clinics) solves this by being a fixed optical system built to a certified,
factory-known refractive state — but a single certified prescription per
unit, and commercial units cost hundreds to low-thousands of dollars each,
mostly because they're calibrated for exactly one fixed value.

What we actually need is different from what a clinical model eye provides:
**one rig that can be set to a large number of known prescriptions, cheaply,
repeatably, and ideally automatically**, purely to generate labeled training
pairs. This document is that design.

## 2. Requirements, stated explicitly

1. Simulate sphere over a clinically realistic range (roughly -6 D to +4 D,
   matching the `sample_prescription()` distribution already used for
   synthetic training).
2. Simulate cylinder (0 to about -3 D) at any axis (0-180°).
3. The "retina" analog must reflect/scatter near-IR light back out through
   the pupil in a way that's at least qualitatively similar to a real
   fundus.
4. The aperture ("pupil") should be adjustable, ideally, so we can also
   validate the pupil-size nuisance parameter already present in the
   domain-randomized training data.
5. The whole thing needs to be rigid and light-sealed enough that repeated
   measurements are actually repeatable — no measurement is worth taking if
   ambient IR (sunlight, incandescent light) or mechanical drift swamps the
   signal.
6. Every configuration must produce an *exact, known* `(S, C, axis)` label,
   because the entire point is to generate ground truth.
7. Ideally: motorized, so a script can sweep hundreds or thousands of
   configurations unattended, matching the scale (`--n 100000`) of the
   existing synthetic dataset generator.

## 3. The physics — and the key design freedom this gives you

Your intuition is exactly right and is worth stating precisely, because it
determines several downstream design choices: **a real myopic eye is,
anatomically, usually an eye whose axial length is too long** — the retina
sits physically farther back than the point where the eye's cornea+lens
system actually brings parallel light to focus. Hyperopia is the opposite:
the retina sits in front of that focus point. Your proposed rig — a fixed
"cornea" lens with a movable "retina" screen — doesn't just *approximate*
ametropia, it **physically re-enacts its most common real mechanism** (axial
length variation), rather than the alternative of physically bending a lens
to change its power. That is a genuinely nice property to say out loud to
Professor Praneeth — it's not a hack, it's the same anatomy in miniature.

**The exact relationship.** Let the fixed lens have focal length `f0`. If
the "retina" screen sits exactly at distance `f0` behind the lens, that
configuration is emmetropic (0 D) by construction. If you shift the screen
by `Δx` (metres) *away* from the lens (deeper, past the focus point), the
lens power that *would* be needed at the corneal plane to pull the focus
back onto the new, farther-away screen position is, from the thin-lens
vergence relationship:

```
S  =  - Δx / f0²        (Δx in metres, f0 in metres, S in dioptres)
```

Moving the screen away from the lens (`Δx > 0`) gives negative `S` —
myopia, matching the sign convention already used everywhere in `eyesim`
(`optics.py`: "negative S = myopia"). Moving the screen closer than `f0`
gives positive `S` — hyperopia. This is exact to first order in `Δx/f0`
(a few percent error at the largest excursions you'd use; easy to correct
with a lookup table generated from the exact non-linear formula if it
matters, `S = 1/(f0+Δx) - 1/f0`).

**The important, non-obvious design freedom this reveals**: `f0` — the
focal length of your "cornea" lens — is *not* constrained to match a real
eye's ~17-22 mm effective focal length. Look at what actually determines
the crescent pattern the camera sees: `eyesim.photorefraction.render_single`
computes the pupil-plane wavefront slope directly from the dioptre value
`(M, J0, J45)`, the pupil radius, and the camera geometry `(D_mm, ex, ey)` —
**it never depends on how those dioptres were physically produced inside
the eye.** I verified this directly against the code: the physics only
cares about the emergent wavefront error *at the pupil*, in dioptres, plus
the external geometry (working distance, LED offset, pupil size) — all of
which you control and can match to the intended wearable's real values
independent of your rig's internal lens choice.

**Practically, this means: pick `f0` for mechanical convenience, not
anatomical fidelity.** The travel required per dioptre is `Δx = f0² × ΔS`.
For `f0 = 17 mm` (real-eye scale) you'd need ~0.29 mm of travel per dioptre
— a genuinely hard precision-machining target. For `f0 = 50 mm` (an
ordinary, cheap benchtop lens) you get `Δx ≈ 2.5 mm per dioptre` — trivially
achievable with an off-the-shelf linear stage, at 1/10 the precision
requirement. **Recommendation: use a 40-75 mm focal length lens for the
"cornea," not anything eye-scale.** This single choice removes most of the
mechanical difficulty from the sphere axis.

**Astigmatism composes exactly, for the same reason.** Inserting a
calibrated cylindrical trial lens of power `C` at rotation angle `theta`
directly in front of the pupil, in contact with (or very close to) the
fixed spherical system, adds its power directly in the conventional
clinical sense: the rig's realized prescription is simply
`(S = ΔS_from_slider, C = C_lens, axis = theta_from_rotation_stage)`, which
feeds straight into `eyesim.optics.prescription_to_power_vector()` with zero
adaptation. This is exact (to the same thin-system approximation as the
sphere formula above), not a fit or a rule of thumb — it's the same
composition rule a phoropter or trial-lens set relies on clinically.

## 4. Three build options, increasing cost and capability

I'd suggest treating these as **sequential**, not exclusive: Option A gets a
first real dataset in days, while B (and, opportunistically, C) are being
built.

### Option A — Manual trial-lens rig ("v0", days to build, ~$100-150)

A fixed lens (`f0 ≈ 50 mm`) + a fixed diffuse screen at its back focal
plane (the permanent "0 D" reference), with a standard optometric **trial
lens set** (spheres in 0.25 D steps, cylinders separately) held in a trial
frame or simple lens holder directly in front of the pupil aperture. To
change the simulated prescription, swap trial lenses by hand between shots;
a capture script fires the LEDs and saves the frame, and you type in the
label (or, better, log it from a lens-ID lookup table).

- No custom mechanics at all — only a lens mount, an iris, a diffuser, and a
  breadboard.
- Throughput: maybe 100-300 labeled frames per session, limited by hand
  swap speed. Enough for an initial fine-tuning experiment to *measure* the
  sim-to-real gap even before a fully automated dataset exists.
- Used trial lens sets (sphere + cylinder, in a case) are common secondhand
  optometric equipment and inexpensive relative to a dedicated "model eye."

### Option B — Motorized rig (recommended primary build, weeks, ~$400-700)

Automates exactly the two axes Option A does by hand:

- **Sphere axis**: the diffuse "retina" screen mounted on a **linear rail +
  lead-screw stage**, driven by a stepper motor. Do not 3D-print the
  sliding mechanism itself (plastic-on-plastic sliding surfaces backlash and
  wear, which is exactly the repeatability failure you want to avoid) —
  instead use an off-the-shelf linear-rail-and-leadscrew module of the kind
  sold cheaply for 3D printers/desktop CNC machines (e.g. an MGN12 rail +
  T8 leadscrew + NEMA17 stepper kit). These give sub-10-micron practical
  repeatability for ~$20-40, which at `f0=50mm` (2.5 mm travel/D) is
  vastly more precision than needed. 3D-printed parts are fine for
  *brackets and mounts* attached to this rail — just not as the precision
  sliding interface itself.
- **Astigmatism axis**: a small NEMA17 (or smaller NEMA11) stepper driving a
  rotating mount that holds one cylindrical trial lens, via a GT2 belt or a
  printed ring gear — this one *is* fine to 3D print, since rotational
  angular precision is far more forgiving to print-quality backlash than
  linear position is. A repurposed small motorized filter wheel (common,
  cheap, secondhand in the astrophotography hobby market) is a drop-in
  alternative if you'd rather not build the rotation mechanism from
  scratch.
- **Control**: any small microcontroller (Arduino + stepper drivers, or a
  Raspberry Pi with a stepper HAT) driven from a Python capture script — see
  §7.

This gets you continuous, computer-commanded sphere and cylinder-at-any-axis,
which means unattended overnight data-collection runs, matching the scale
of the existing synthetic generator.

### Option C — Repurposed phoropter (opportunistic, cost highly variable)

Worth actively checking whether the department, an affiliated
optometry/ophthalmology program, or a medical-equipment surplus channel has
a **decommissioned manual or digital phoropter** sitting unused. A
phoropter's internal sphere/cylinder/axis wheels are exactly the mechanism
described above, already built by an optical instrument manufacturer to
calibrated tolerances — you'd only need to attach a diffuse screen behind
its eyepiece (in place of a patient's eye) and image through it from the
front. Manual units are common and inexpensive on the secondhand medical
equipment market; some older **digital/automated phoropters** already
contain stepper-motor-driven dials, occasionally with a serial interface,
which — if one is findable — would solve essentially the entire mechanical
half of this proposal in a single acquisition. This is worth a low-effort
email/ask before committing shop time to Option B; if nothing turns up,
Option B is a fully sufficient fallback and arguably better for us anyway,
since we understand and control every part of it.

## 5. Component-level recommendations

**"Cornea" lens.** A single **plano-convex or bi-convex singlet**, 40-75 mm
focal length, 20-25 mm clear aperture, is sufficient and cheap (~$10-25):
you're only using the central 2-4 mm (the simulated pupil) of a much larger
optic, so off-axis and chromatic aberration are negligible at that
sub-aperture — and we're illuminating with narrowband 850 nm IR, so
chromatic correction buys nothing during actual data capture. An
**achromatic doublet** (~$30-80) is worth the extra cost only if you also
want to visually verify focus with a visible-light test chart (recommended
for calibration, see §6) — in that case pick a doublet color-corrected
across visible + near-IR if available; otherwise a cheap uncoated singlet
is genuinely fine for the IR-only data-collection path.

**Pupil / iris.** A standard adjustable mechanical iris diaphragm
(~$15-30), mounted at (or immediately behind) the cornea lens, set to
whatever pupil diameter you're simulating for that batch (3-4 mm typical,
matching `_PUPIL_R_MM = 3.0` in `photorefraction.py`). Log the actual set
diameter per batch — this becomes a free extra validation of the pupil-size
domain-randomization parameter already in the synthetic generator.

**"Retina" / diffuser.** This is the one component where "cheap and good"
genuinely coexist: **PTFE (Teflon) sheet or plumber's-tape-grade PTFE**, or
plain matte white cardstock/printer paper, both give a near-Lambertian,
high-reflectance diffuse surface in the visible-to-NIR range for a few
dollars, and are trivially disposable/replaceable if damaged. (A
laboratory-grade Lambertian standard like Spectralon exists but at a cost —
hundreds of dollars — that isn't justified here; we need consistency across
our own frames, not absolute radiometric calibration.) One documented
caveat worth stating plainly in the proposal rather than discovering later:
real fundus reflectance has a partial *retroreflective* component from
photoreceptor waveguiding (the mechanism behind "red-eye" in flash photos),
which a flat diffuser does not reproduce. If early captured crescents look
qualitatively much lower-contrast than published photorefraction images,
this is the first thing to revisit (e.g. a diffuser over a mildly specular
backing).

**Camera + illumination.** A NoIR (IR-cut filter removed) camera module —
e.g. the Raspberry Pi NoIR camera or an inexpensive USB webcam with the IR
filter physically removed — plus two 850 nm IR LEDs, mounted via
3D-printed brackets at **known, measured offsets from the camera's optical
axis**. Match these to the values already assumed by the simulator
(`D_mm=1000`, `eccentricity_mm=10`) if the rig's benchtop geometry allows a
1-metre working distance, or — the more practical choice for a benchtop
rig — pick a shorter, convenient working distance (e.g. 300-500 mm) and
**pass the actual measured `D_mm`/`ecc_mm` into `render_multichannel()`**
when generating any additional synthetic data meant to match this rig, so
the synthetic and real datasets stay geometrically self-consistent. This is
a one-line change to `generate_dataset.py`'s constants, not a code
redesign.

**Mounting.** A small aluminum optical breadboard (M6 or 1/4-20 tapped
holes, roughly 300×450 mm, ~$50-150) with standard post-and-clamp hardware
for every element — lens, iris, LEDs, camera, both motion stages. This is
the actual answer to "100% not moving": adhoc mounting (hot glue, tape,
improvised brackets) is the single most common source of irreproducible
optics-bench results, and a breadboard is the standard, well-understood fix
used in every teaching and research optics lab. Enclose the whole assembly
in a simple blackout-cloth hood or foam-core box — ambient IR from sunlight
or incandescent lighting will otherwise contaminate every frame, and unlike
visible light this contamination is invisible to the naked eye while
setting up, which makes it an easy mistake to not notice until the data
looks wrong.

## 6. Calibration and verification procedure

Trusting a label requires trusting the rig's zero point and its lens
specs, not just the formula in §3. Recommended procedure, front-loaded
before any large data-collection run:

1. **Measure the cornea lens's actual focal length** (autocollimation, or
   simply focusing a distant object and measuring the image distance)
   rather than trusting the datasheet nominal value — the `1/f0²` term in
   the sphere formula makes small errors in `f0` matter.
2. **Find the true zero (emmetropic) slider position empirically, not just
   by calculation.** Place a visible-light resolution test chart (e.g. a
   USAF 1951 chart, or reuse the repository's own `make_chart()` /
   `make_siemens_star()` test patterns printed out) at the screen position,
   and mechanically sweep the slider while maximizing image sharpness —
   literally reusing `eyesim.optimize.sharpness()` on the live camera feed.
   The slider position of maximum sharpness *is* the empirical zero point,
   independent of any focal-length calculation error. This is a nice
   detail for the proposal: **the rig can self-calibrate using the exact
   same sharpness-maximization code already written and tested for Step
   3**, rather than needing separate calibration software.
3. **Verify the rotation stage's angle reference** against a printed
   protractor scale or an alignment jig before trusting any astigmatism
   label.
4. Periodically re-verify zero position (thermal drift, mechanical
   settling) — cheap to check, since it's the same sharpness-max routine
   as step 2.

## 7. Data pipeline integration

The rig should write directly into the same `.npz` schema
`generate_dataset.py` already produces (`images: (N,H,W,2) float32`,
`labels: (N,3) float32` in `[M, J0, J45]`), so it needs zero adapter code to
be consumed by `eyesim.cnn.PrescriptionDataset` or `scripts/train.py`.
Concretely, a capture script would, per configuration:

1. Command the slider (and, if present, rotation stage) to the target
   `(S, C, axis)`.
2. Wait for mechanical settle.
3. Fire the horizontal LED, capture a frame; fire the vertical LED, capture
   a frame; assemble the 2-channel image exactly as
   `render_multichannel()` does.
4. Convert the commanded `(S, C, axis)` to `(M, J0, J45)` via the existing
   `eyesim.optics.prescription_to_power_vector()` — the *same* function
   the synthetic generator uses, so labels are computed identically on both
   sides.
5. Append to a running `bench.npz`, alongside the actual measured geometry
   (`D_mm`, `ecc_mm`, pupil diameter) as metadata for that batch.

Fine-tuning recipe: load `checkpoints/best.pt`, continue training at a
reduced learning rate on batches mixing real (`bench.npz`) and synthetic
data (to avoid catastrophic forgetting on the much larger synthetic set),
and evaluate the *change* in error specifically on a held-out slice of real
images — that delta is the actual measurement of how large the sim-to-real
gap was and how much it closed.

## 8. A second use for the same rig: validating the closed-loop optimizer on real hardware

Worth stating explicitly, because it strengthens the case for building
Option B rather than stopping at Option A: this rig, once motorized, is not
just a labeling tool. The slider **is** a physical actuator with a known
dioptre-per-step relationship, and the rotation stage **is** a physical
astigmatism actuator. That means `eyesim.optimize.run_closed_loop()`'s
control logic — the coarse scan plus Nelder-Mead search that currently only
ever "moves" a simulated lens — can be pointed at this rig instead, using
the camera's *actual measured* image sharpness in the objective function
instead of a simulated PSF convolution. This is a first physical validation
of Step 3's control algorithm (does it still converge with real motor
backlash, real camera noise, real vibration?) months before an Alvarez
lens actuator (`proposal/main.tex`, §"Proposed Alvarez Lens Integration")
needs to exist. It turns one hardware build into progress on two roadmap
items at once.

## 9. Safety

Even though initial imaging targets a diffuser, not a human eye, it's worth
designing within safe limits from the start rather than retrofitting later,
since the same LEDs and mounting will likely carry over into Phase 2 human
trials (`proposal/main.tex` roadmap). Use IR LEDs with a published radiant
intensity spec, keep drive current low (a few mA to tens of mA is typically
more than sufficient for this short working distance), and before any
human imaging, compute exposure against IEC 62471 (photobiological safety
of lamps) — infrared is invisible, so the blink/aversion reflex that
normally limits visible-light exposure doesn't protect a user, which makes
this an easy thing to get wrong by assuming "it's just an LED."

## 10. Estimated bill of materials (Option B)

| Item | Est. cost |
|---|---|
| Cornea lens (singlet, 40-75mm FL) | $10-25 |
| Iris diaphragm | $15-30 |
| PTFE / diffuser sheet ("retina") | $5-15 |
| Linear rail + leadscrew + NEMA17 (sphere axis) | $20-40 |
| Rotation stage / motorized mount (astigmatism axis) | $30-100 (DIY) or $50-150 (used filter wheel) |
| NoIR camera module | $25-40 |
| 850 nm IR LEDs (x2) + mounts | $10-20 |
| Stepper drivers + microcontroller | $20-40 |
| Optical breadboard + posts/clamps | $50-150 |
| Blackout enclosure materials | $20-40 |
| Trial lens set (cylinders, for astigmatism) — secondhand | $50-150 |
| Misc. (wiring, fasteners, 3D-print filament) | $20-40 |
| **Total (Option B)** | **≈ $275-650** |

Option A is a strict subset of this (no motion stages/controller needed) at
roughly $100-150. Option C's cost depends entirely on what surplus
equipment is findable and could range from near-zero (department has one in
storage) to a few hundred dollars (used market).

## 11. Suggested milestones

1. **Week 1-2**: Option A built and generating labeled frames by hand;
   first fine-tuning experiment run on ~200 real images to get an initial,
   even if noisy, measurement of the sim-to-real gap.
2. **Week 3-5**: Option B's sphere axis (linear rail) built and calibrated
   per §6; automated sphere-only sweep generates a few thousand
   spherical-only labeled frames.
3. **Week 5-7**: astigmatism axis added; full `(S, C, axis)` sweeps running
   unattended.
4. **Week 6-8 (parallel)**: point `run_closed_loop()` at the rig (§8) as a
   first real-hardware test of the control algorithm, independent of the
   CNN fine-tuning track.
5. **Ongoing**: fold newly-labeled real data into periodic PupilNet
   fine-tuning runs; track the real-vs-synthetic error gap over time as the
   actual success metric for this whole effort.

## 12. Open questions for Professor Praneeth

- Does the department (or an affiliated optometry/ophthalmology group) have
  any decommissioned phoropter, trial lens set, or model eye already in
  storage (Option C)? Worth a quick ask before ordering parts.
- Is there existing optics-bench / breadboard equipment or a machine
  shop / 3D-printer access we could use, versus needing to budget for
  a dedicated breadboard?
- Is there a budget line this ~$300-650 build (Option B) can draw from, or
  should we start strictly with Option A while that's sorted out?
- Any early guidance on human-subject / IRB requirements we should be
  aware of well before Phase 2 (real-eye imaging), even though this initial
  rig only ever images an artificial diffuser?

---

Let me know which pieces you'd like to change — happy to swap any of the
above for lab-available equipment if we already have some of it, or to
scope Option B down further if $300-650 is more than makes sense for a
first pass.

Best,
Mier
