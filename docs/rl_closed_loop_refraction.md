# Closed-loop refraction with an eye that refocuses

Design note, 10 November 2026. The reinforcement-learning part of the eyesim project.

## The problem in one paragraph

**The goal:** the self-refracting glasses (`hardware/alvarez_doc/`) turn an Alvarez lens until the camera sees a
neutral reflex. The lens power at that point is the prescription.

**The flaw:** the wearer's eye is a **second controller acting on the same lens**.
- If the lens is too minus, the eye focuses harder (accommodates) and cancels the error itself. The camera then sees
  "neutral" at the wrong lens, so a simple loop settles on an over-minus prescription.
- Clinicians avoid this with plus lenses ("fogging"). But the right amount and timing differ from person to person:
  too little fog leaves accommodation in play, and too much lets the eye drift to its resting focus (about 1 D).

**On our bench it is worse:** at 37 cm, the clinically useful amount of fog (+1 to +2 D) puts the eye inside the
photorefraction blind band, where there is no reading. Every lens power that *does* give a reading also disturbs the
eye: either heavy fog (the eye drifts) or minus (the eye focuses harder).

**The task:** get the relaxed refraction anyway, quickly, from an eye that reacts to every move. That is a
partially observed control problem with delays and person-specific dynamics, which is where learning a policy beats
writing one.

## Why this is new, and why it is not redundant

- **The measurement exists.** Video-rate infrared photorefraction of accommodation is established: PowerRef 3 at
  50 Hz was used to measure accommodation fluctuations [1].
- **The clinical need exists.** Fogging lenses are recommended anywhere from +0.75 to +6.00 D, timing matters, and
  +1.00 D fog does not fully relax accommodation in a closed-view instrument. Non-cycloplegic refraction in children
  remains a known problem [2, 3].
- **Not done before:** in our searches (web, not patents) we found no closed-loop, adaptive, or learned control of
  fogging or accommodation during refraction. Closest work: RL for a different eye test (perimetry, RLPeri [4]); RL
  driving a tunable lens to focus an OCT scanner, which measures no prescription [5]; RL for adaptive optics in
  astronomy [6].
- **Not redundant, tested directly:** simple controllers fail on this problem (results below). A hand-designed rule
  that knows the eye model does much better, and the agent has to beat that rule, not a strawman.

## Environment (`eyesim/rl/`)

**The eye** (`accommodation.py`). Every number is drawn per simulated person, with deliberately wide ranges:
- reaction delay 0.25–0.45 s
- exponential response, faster to focus (0.15–0.5 s) than to relax (0.25–0.9 s)
- lag and lead around the resting focus: gain 0.7–0.95, resting focus 0.25–1.75 D
- mild fog up to a personal limit (1–3 D) relaxes accommodation fully
- heavier fog: the eye drifts to its resting focus (time constant 1.5–6 s)
- instrument myopia of 0–1 D (more in children) that fades only while the eye is mildly fogged
- depth of focus 0.15–0.4 D
- microfluctuations of 0.05–0.25 D RMS
- maximum focusing range by age (Hofstetter)

Sources for these ranges: [1, 3] and the accommodation-dynamics literature (latency ~0.3 s, microfluctuations
<1 D with a peak near 2 Hz).

**The sensor** (`env.py`) is built from **real data**: the network's readings of the 37 cm model-eye photos
(blocks A, B, C).
- No reading inside the measured blind band (−5.83 to +2.25 D).
- Otherwise: true value + the network's measured bias at that refraction + 0.10 D noise per frame (the measured
  per-shot spread).

**One step = one camera frame (30 Hz)**, episodes up to 10 s. Actions: change the lens by −3, −1, −0.25, 0, +0.25,
+1 or +3 D, or report.

**The report is fixed, not learned:** the median of (reading + lens) over the last 6 readable frames. The agent
cannot guess the answer; it can only decide which lens powers to use, for how long, and when to stop.

**Reward:** −|reported − true relaxed refraction|, minus 0.002 per frame.

## Baselines (`scripts/rl_closed_loop.py`), same 2000 simulated eyes

| Policy | What it does | Average error | Within 0.25 D | Time |
|---|---|---|---|---|
| Error floor | true relaxed eye, no accommodation, real sensor | 0.34 D | 2 % | 0.2 s |
| One-shot | find a readable lens, read, report | 1.53 D | 9 % | 0.3 s |
| Null-seeking loop | the glasses design's proportional loop | 1.41 D | 9 % | 0.4 s |
| Fog-then-flash | hand-designed with full knowledge of the eye model: fog mildly for 3 s, then jump to a readable lens and report within 0.2 s, before the eye reacts | 0.84 D | 38 % | 3.5 s |

The floor's 0.34 D is the network's real bias on the model eye at −7.5 D. It is a sensor property, not a control
problem.

## Agent and results

**Agent:** PPO with a 2×256 network.
- **Input:** the last 0.5 s of (estimate, readable?, lens, time), plus a short memory: last estimate, time since it,
  number of readings, and time since the lens last moved.

**Training, as actually done** (failed attempts included):

| Attempt | Setup | Result |
|---|---|---|
| 1 | from scratch | stuck at ~3 D (rarely found a reading; reward only at the end) |
| 2 | copy fog-then-flash first, then PPO | got worse; the time penalty was too large, so it cut the fogging short |
| 3 | smaller time cost, value warm-up, a pull toward the copied rule, best checkpoint by a no-randomness score on separate eyes | improved fog-then-flash from 0.84 to ~0.63–0.72 D, but the stronger hand rule (fog-flash twice, 0.55 D) was better |
| 4 (final) | same, started from fog-flash twice | best on the separate check eyes 0.43 D; saved there; later updates drifted back up |

**Test: the same 2,000 unseen eyes for every policy** (different from the check eyes):

| Policy | Average error | Median | Within 0.25 D | Within 0.5 D | Misses > 1 D | No answer | Time |
|---|---|---|---|---|---|---|---|
| Fog-then-flash | 0.84 D | 0.38 | 38 % | 54 % | 31 % | 1.9 % | 3.5 s |
| Fog-flash twice (best hand rule) | 0.55 D | 0.26 | 49 % | 73 % | 16 % | 1.7 % | 5.4 s |
| Copy of that rule, no RL | 0.55 D | 0.25 | 50 % | 72 % | 16 % | 1.7 % | 5.5 s |
| **RL agent** | **0.46 D** | 0.27 | 47 % | 72 % | **11 %** | **0.8 %** | 5.8 s |

**RL minus the best hand rule, eye by eye:** −0.10 D (95 % bootstrap CI −0.13 to −0.07).

**What RL bought:** about a third fewer large misses and half the "no answer" cases, for 0.4 s more. The typical
eye (median, within 0.25 D) is unchanged. The error floor with no accommodation is 0.34 D, so the agent closes
about half the gap between the best rule and that floor.

**Robustness test: eyes outside the training ranges.** These eyes react later (0.45–0.6 s), tolerate less fog
(0.6–1.5 D), have a deeper resting focus (1.25–2.5 D), and more instrument myopia that fades slowly (4–10 s):

| Policy | Average error | Misses > 1 D |
|---|---|---|
| Fog-flash twice | 1.87 D | 83 % |
| RL agent | 1.75 D | 83 % |
| Null-seeking loop | 2.50 D | 95 % |

The agent is still better (−0.12 D, CI −0.15 to −0.10), but **every policy fails**: under 6 s of fogging cannot
relax an eye whose instrument myopia fades over 4–10 s. Clinical fogging in children is applied for minutes (+2 D
for 20 min in [2]).

**The deciding unknown is how fast a real eye relaxes.** The next experiment is to measure that with this device:
video-rate photorefraction of a real eye while a plus lens fogs it. The current capture is about 1 frame pair per
1.3 s (full-resolution raw), so this needs a fast cropped video mode first.

## Open risks

- **The eye model is assembled from literature ranges, not fitted to anyone.** The agent may learn quirks of the
  model. Mitigations:
  - wide randomisation
  - validation on the model eye (sensor real, accommodation emulated by moving the dial)
  - then on a real eye with a fast tunable lens
- **The policy needs a fast lens (one step per frame).** A stepper-driven Alvarez lens is slower; the environment has
  a `lens_rate` option to test that.
- **The sensor is one rig, one model eye, 37 cm.** The glasses geometry (Badal relay, null at 0 D) changes where the
  blind band sits, but not the conflict between reading and disturbing the eye.

## References

1. Bharadwaj SR et al. Fluctuations of steady-state accommodation is a marker for screening spasm of near reflex.
   *Transl Vis Sci Technol* 2021;10(11):9. https://pmc.ncbi.nlm.nih.gov/articles/PMC8431981
2. Refraction in children: a comparison of two methods of accommodation control. https://pubmed.ncbi.nlm.nih.gov/23142881/
3. Influence of power and the time of application of fogging lenses on accommodation. https://ouci.dntb.gov.ua/works/4LAJ5JVl
4. Verma T et al. RLPeri: accelerating visual perimetry test with reinforcement learning. AAAI 2024. https://arxiv.org/abs/2403.05112
5. Xu G et al. Deep reinforcement learning for automatic defocus correction using OCT image intensity.
   *Biomed Opt Express* 2025;16(10):4175. https://pmc.ncbi.nlm.nih.gov/articles/PMC12532345
6. Self-optimizing adaptive optics control with reinforcement learning. https://arxiv.org/abs/2012.01997
