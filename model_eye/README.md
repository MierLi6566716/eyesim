# Model-eye test (XSJ-501 schematic eye)

A model eye has a **known** refraction (its barrel scale), never moves, never
blinks and never focuses. If the network is right on it, what's left on a
real eye is real-eye stuff (iris, eyelid, movement). If it's wrong here too,
the rig or the simulator is off, and this test says which.

What it answers:

1. **Does the network track the scale?** The slope of network vs scale should
   be about 1. On your own eye it read about −5 whatever the lens, which looks
   like a slope well below 1.
2. **Is the simulated blind band right?** The crescent side for each setting
   is compared with what the simulator predicts.
3. **Is the pixel scale right?** The pupil size the script measures is
   compared with the model eye's real pupil (3 / 5 / 7 mm).

## The XSJ-501

| | |
|---|---|
| Refraction | Barrel scale **+4 to −5 D**, accurate to 0.25 D. Read it at the "Marker". |
| Pupils | **3, 5, 7 mm**: flip the sheet-shaped handle under the front of the barrel |
| Lens grooves | 3 arc grooves in front of the barrel; the axis dial is only for cylinder lenses |
| Tilt | Axis screw; set the barrel level, pointing straight at the camera |

**Ignore the manual's "pre-add −1.00 D at 1 m / +2.00 D at 0.5 m" step.**
That is for human retinoscopy at a working distance. Our software already
corrects for the camera distance. Only put a lens in when a block below says
so, and always pass it as the lens number.

Pupil choice: **7 mm** for the main blocks, **5 mm** for one extra block.
**Skip 3 mm**: at 37 cm its blind band covers about −10.5 to +5 D, so every
setting would be dark.

What to expect at 37 cm, 7 mm (eye + lens, from the simulator):
dark between about **−6 and +0.5 D**. A crescent on the far side (left in H)
above that, on the LED side (right in H) below it.

## Folders

| Path | What | On GitHub? |
|---|---|---|
| `model_eye/sessions/` | Raw photos, one folder per setting | No (GBs) |
| `model_eye/results/<block>/` | `results.csv`, `report.png`, `summary.txt` | Yes |
| `model_eye/sweep.sh` | Pi helper: one session per scale setting | Yes |
| `model_eye/NOTES.md` | Your notes from the day (distances, gain, anything odd) | Yes |

Session folder names say everything:
`20261106-101500-model-37cm-S+4-true-2.5-p7mm` = model eye, 37 cm, +4 lens,
scale −2.5 D, 7 mm pupil.

---

## Part 1: on the Mac, copy the new scripts to the Pi

```bash
cd ~/Desktop/eyesim
git pull
scp hardware/eye_capture.py scripts/pack_sessions.py model_eye/sweep.sh rpi:~/eyesim/
ssh rpi 'chmod +x ~/eyesim/sweep.sh; mkdir -p ~/eyesim/model_eye; df -h ~ | tail -1'
```

The last number is the Pi's free space. You need about **9 GB**. If it's less,
pack after each block (Part 3, step 9).

## Part 2: set up (on the Pi, `ssh rpi`, `cd ~/eyesim`)

**Step 1. Place it.** Put the XSJ-501 on the table, barrel level and pointing
straight into the camera. Its **front lens** should be **37.0 cm** from the
front of the camera lens. Pupil handle on **7 mm**, no lens in the grooves,
room dark. Write the measured distance in NOTES.md. From now on touch only the
barrel and the grooves.

**Step 2. Ruler photo** (pixel scale). Hold a mm ruler flat against the front
of the barrel, then:
```bash
python3 eye_capture.py --scale
```
On the Mac:
```bash
cd ~/Desktop/eyesim
scp rpi:~/eyesim/scale_H.npy model_eye/scale_H_37cm.npy
/opt/anaconda3/bin/python scripts/measure_eye.py --calibrate-scale model_eye/scale_H_37cm.npy --dist-cm 37
```
Click two ruler marks far apart, then type their distance in mm. Put the two
printed numbers into `~/eyesim/rig.json` on the Pi (`nano ~/eyesim/rig.json`):
```json
{"ecc_h_mm": 10.0, "ecc_v_mm": 10.0, "aperture_mm": 2.6, "dist_cm": 37,
 "px_per_mm": 4.92, "px_per_mm_dist_mm": 370}
```
(Use your numbers in place of 4.92 / 370.)

**Step 3. Exposure.** Set the barrel to **+4**:
```bash
python3 eye_capture.py --check --gain 4 --exposure-us 20000
```
Open `check_H.png`. You want a clearly visible crescent that is **not
clipped** (the printout says if it is). Too dark: raise `--gain` (max 16) or
`--exposure-us` (max 30000). Clipped: lower them. Then fix the values:
```bash
export GAIN=4 EXPOSURE=20000      # your numbers; re-type in every new ssh window
```

**Step 4. Sign check** (`python3 eye_capture.py --check --gain $GAIN --exposure-us $EXPOSURE`):

| Barrel | Lens in groove | Expected in `check_H.png` |
|---|---|---|
| +4 | none | crescent on the **left** (away from the LED) |
| −3 | none | **dark** pupil |
| −5 | **+4** | crescent on the **right** (LED side) |

If it's the other way round, write it in NOTES.md and tell me before going on.
Take the +4 lens out again.

## Part 3: take the photos (on the Pi)

Each line runs a whole block. It tells you which setting to turn the barrel
to, waits for Enter, then shoots. `s` + Enter skips one, `q` + Enter stops.

Arguments: `distance  lens  pupil  shots  scale_from  scale_to  step`.

**Step 5. Block A: 37 cm, no lens, 7 mm.** 19 sessions, about 12 min.
Hyperopic side: crescents from about +0.5 up, dark below.
```bash
./sweep.sh 37 0 7 10 4 -5 0.5
```

**Step 6. Block B: 37 cm, +4 lens, 7 mm.** The most important block: eye +
lens runs from −4 to −9, like your own eye. 11 sessions. Put the **+4.00**
lens in the groove **closest to the barrel**. Look at the first session's
`preview_H.png`: if a bright lens reflection sits near the pupil, rotate the
lens a little in its groove and redo that setting.
```bash
./sweep.sh 37 4 7 10 0 -5 0.5
```

**Step 7. Block C: 30 cm, closer.** Slide the model eye to **30.0 cm**. The
pixel scale adjusts for distance by itself.
```bash
./sweep.sh 30 0 7 5 4 -5 1        # lens out: 10 sessions
./sweep.sh 30 5 7 5 0 -5 1        # +5.00 lens in: 6 sessions
```

**Step 8. Block D: 37 cm, 5 mm pupil.** Back to **37.0 cm**, pupil handle on
**5 mm**.
```bash
./sweep.sh 37 0 5 5 4 -5 1        # lens out: 10 sessions
./sweep.sh 37 5 5 5 0 -5 1        # +5.00 lens in: 6 sessions
```

**Step 9. Pack on the Pi** (lossless, about 2.5x smaller and faster to copy):
```bash
python3 pack_sessions.py model_eye/*/
```

If there's time left: repeat **block B with 20 shots**
(`./sweep.sh 37 4 7 20 0 -5 0.5`). More photos of the same setting are what
we'd fine-tune on later.

## Part 4: on the Mac, copy and analyse

**Step 10. Copy** (re-run if the Wi-Fi drops; `rsync` resumes):
```bash
cd ~/Desktop/eyesim
rsync -av --partial rpi:~/eyesim/model_eye/ model_eye/sessions/
/opt/anaconda3/bin/python scripts/pack_sessions.py model_eye/sessions/*/
df -h . | tail -1
```

**Step 11. Run the report, one block at a time.** For the first session of
each block a window opens: click the pupil centre, then 3 points on its edge,
then click to accept. The other sessions reuse those clicks.
```bash
PY=/opt/anaconda3/bin/python
$PY scripts/model_eye_report.py model_eye/sessions/*-model-37cm-S+0-*-p7mm --reuse-clicks --out model_eye/results/A_37cm_nolens_7mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-37cm-S+4-*-p7mm --reuse-clicks --out model_eye/results/B_37cm_plus4_7mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-30cm-S+0-*-p7mm --reuse-clicks --out model_eye/results/C_30cm_nolens_7mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-30cm-S+5-*-p7mm --reuse-clicks --out model_eye/results/C_30cm_plus5_7mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-37cm-S+0-*-p5mm --reuse-clicks --out model_eye/results/D_37cm_nolens_5mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-37cm-S+5-*-p5mm --reuse-clicks --out model_eye/results/D_37cm_plus5_5mm
```
Each writes `results.csv`, `report.png` and `summary.txt` into its folder and
compares both networks (`checkpoints/best.pt` and `checkpoints/wide_m_only/best.pt`).

**Step 12. Look at `model_eye/results/B_37cm_plus4_7mm/report.png` first:**
- Left: dots along a line parallel to the dashed one means it works. A
  constant offset is fine: a plastic model eye's colour error at 850 nm isn't
  a human eye's. A flat line means the same squashing as on your eye.
- Middle: real crescent sides (dots) against the simulator (steps). If the
  dots switch sides at a different setting than the steps, the simulated
  blind band is off.
- Right: measured vs real pupil (7 or 5 mm). More than 10 % off means the
  pixel scale is wrong. That was suspect #1 for your eye.

**Step 13. Save to GitHub:**
```bash
git add model_eye/results model_eye/NOTES.md
git commit -m "Model-eye results"
git push
```
Then tell me the results are in.

## Storage

| Block | Sessions | Raw on Pi | Packed |
|---|---|---|---|
| A, B (10 shots) | 30 | 5.3 GB | 2.1 GB |
| C, D (5 shots) | 32 | 2.8 GB | 1.1 GB |
