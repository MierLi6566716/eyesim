# Model-eye test

A model eye has a **known** refraction (its dial), never moves, never blinks
and never focuses. If the network is right on it, what's left on a real eye
is real-eye stuff (iris, eyelid, movement). If it's wrong here too, the rig
or the simulator is off, and this test says which.

What it answers:

1. **Does the network track the dial?** Slope of network vs dial should be
   about 1. On your own eye it read about −5 whatever the lens, which looks like
   a slope well below 1.
2. **Is the simulated blind band right?** The crescent side for each dial
   setting is compared with what the simulator predicts.
3. **Is the pixel scale right?** The pupil size the script measures is
   compared with the model eye's real pupil.

## Folders

| Path | What | On GitHub? |
|---|---|---|
| `model_eye/sessions/` | Raw photos, one folder per dial setting | No (GBs) |
| `model_eye/results/<block>/` | `results.csv`, `report.png`, `summary.txt` | Yes |
| `model_eye/sweep.sh` | Pi helper: one session per dial setting | Yes |
| `model_eye/NOTES.md` | Your notes from the day (create it) | Yes |

Session folder names say everything:
`20261106-101500-model-37cm-S+3-true-4.5-p6mm` = model eye, 37 cm, +3 trial
lens, dial −4.5 D, 6 mm pupil.

## Before you start (5 min)

Write down from the model eye's box / manual, into `model_eye/NOTES.md`:
- the dial range and step (e.g. −6 to +6 in 0.5 D),
- which way the sign goes (− = myopic?),
- the pupil sizes it offers,
- where its "cornea" (front surface) is, since distance is measured to it.

The commands below assume dial −6..+6, pupils 6 mm (big) and 4 mm (small).
If yours differ, use **its largest pupil up to 8 mm** for "6" and **its
smallest from 3 mm** for "4", and its own dial range.

---

## Part 1: on the Mac, copy the new scripts to the Pi

```bash
cd ~/Desktop/eyesim
git pull
scp hardware/eye_capture.py scripts/pack_sessions.py model_eye/sweep.sh rpi:~/eyesim/
ssh rpi 'chmod +x ~/eyesim/sweep.sh; mkdir -p ~/eyesim/model_eye; df -h ~ | tail -1'
```

The last number is the Pi's free space. You need about **10 GB** (each
10-shot session is about 180 MB before packing). If it's less, pack after each
block (Part 3, step 9).

The new `eye_capture.py` finds the Arduino on `/dev/ttyACM0` or `/dev/ttyUSB0`
by itself.

## Part 2: set up the model eye (on the Pi, `ssh rpi`)

**Step 1. Place it.** Clamp the model eye so it looks straight into the
camera, with its front surface **37.0 cm** from the front of the camera lens
(measure with a ruler, write it down). Room dark. From now on **don't touch
anything except the dial** (and the trial lens in block B/C).

**Step 2. Ruler photo** (pixel scale). Hold a mm ruler flat against the model
eye's front, then:

```bash
cd ~/eyesim
python3 eye_capture.py --scale
```

On the Mac:
```bash
cd ~/Desktop/eyesim
scp rpi:~/eyesim/scale_H.npy model_eye/scale_H_37cm.npy
/opt/anaconda3/bin/python scripts/measure_eye.py --calibrate-scale model_eye/scale_H_37cm.npy --dist-cm 37
```
Click two marks far apart, type their distance in mm. Put the two printed
numbers into `~/eyesim/rig.json` on the Pi (`nano ~/eyesim/rig.json`), for example:
```json
{"ecc_h_mm": 10.0, "ecc_v_mm": 10.0, "aperture_mm": 2.6, "dist_cm": 37,
 "px_per_mm": 4.92, "px_per_mm_dist_mm": 370}
```

**Step 3. Exposure.** Set the dial to **+4** (strong crescent), big pupil:
```bash
python3 eye_capture.py --check --gain 4 --exposure-us 20000
```
Look at `check_H.png` and the printout. You want the crescent clearly visible
and **not clipped**. Too dark: raise `--gain` (max 16). Clipped: lower it.
Write the final pair into NOTES.md and use it for everything:
```bash
export GAIN=4 EXPOSURE=20000      # your numbers
```
(Re-type this `export` line whenever you open a new ssh window.)

**Step 4. Sign check.** With those settings, take quick checks at dial
**+4**, **−3** and **−6** (`python3 eye_capture.py --check --gain $GAIN --exposure-us $EXPOSURE`).
Expected at 37 cm: +4 = crescent on the side **away** from the LED (left in
`check_H.png`); −3 = dark pupil; −6 = dark or a faint crescent on the LED
side. If +4 shows it on the LED side, the dial sign is the other way round.
Note it in NOTES.md and tell me.

## Part 3: take the photos (on the Pi)

Each line runs a whole block. It tells you which dial value to set, waits for
Enter, then takes the session. `s` + Enter skips one, `q` + Enter stops.

Arguments: `distance  lens  pupil  shots  dial_from  dial_to  step`.

**Step 5. Block A: 37 cm, no lens, big pupil.** Most important. 25 sessions,
about 15 min.
```bash
./sweep.sh 37 0 6 10 6 -6 0.5
```

**Step 6. Block B: 37 cm, +3 lens, big pupil.** Like your own eye: myopic
side, crescent on the LED side. 13 sessions. Put the **+3.00** lens in the
model eye's lens holder (or tape it on the front, tilted about 10° so its
reflection misses the pupil). Check `preview_H.png` of the first session
for a bright lens reflection near the pupil.
```bash
./sweep.sh 37 3 6 10 -6 0 0.5
```

**Step 7. Block C: 30 cm, closer.** Move the model eye to **30.0 cm**. The
pixel scale is adjusted for distance automatically, so no new ruler photo is
needed.
```bash
./sweep.sh 30 0 6 5 6 -6 1          # no lens, 13 sessions
# now put the +4.00 lens in
./sweep.sh 30 4 6 5 -6 0 1          # 7 sessions
```

**Step 8. Block D: 37 cm, small pupil.** Back to **37.0 cm**, no lens,
switch to the small pupil.
```bash
./sweep.sh 37 0 4 5 6 -6 1          # 13 sessions
```

**Step 9. Pack on the Pi** (lossless, about 2.5x smaller, so the copy is faster):
```bash
python3 pack_sessions.py model_eye/*/
```

If there's time left: repeat block A or B with **20 shots**
(`./sweep.sh 37 0 6 20 6 -6 0.5`). More photos of the same setting are what
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
each block a window opens: click the pupil centre, then 3 points on its
edge, then click to accept. Every other session in the block reuses those
clicks (`--reuse-clicks`), because the model eye didn't move.

```bash
PY=/opt/anaconda3/bin/python
$PY scripts/model_eye_report.py model_eye/sessions/*-model-37cm-S+0-*-p6mm --reuse-clicks --out model_eye/results/A_37cm_nolens_6mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-37cm-S+3-*-p6mm --reuse-clicks --out model_eye/results/B_37cm_plus3_6mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-30cm-S+0-*-p6mm --reuse-clicks --out model_eye/results/C_30cm_nolens_6mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-30cm-S+4-*-p6mm --reuse-clicks --out model_eye/results/C_30cm_plus4_6mm
$PY scripts/model_eye_report.py model_eye/sessions/*-model-37cm-S+0-*-p4mm --reuse-clicks --out model_eye/results/D_37cm_nolens_4mm
```

Each writes `results.csv`, `report.png` and `summary.txt` into its folder and
compares both networks (`checkpoints/best.pt` and `checkpoints/wide_m_only/best.pt`).

**Step 12. Look at `model_eye/results/B_37cm_plus3_6mm/report.png` first:**
- Left: dots along a line parallel to the dashed one means it works (a
  constant offset is fine: a plastic model eye's colour error at 850 nm isn't a
  human eye's). A flat line means the same squashing as on your eye.
- Middle: real crescent sides (dots) against the simulator (steps). If the
  dots switch sides at a different dial value than the steps, the simulated
  blind band is off.
- Right: measured vs real pupil. More than 10 % off means the pixel scale is
  wrong. That was suspect #1 for your eye.

**Step 13. Save to GitHub:**
```bash
git add model_eye/results model_eye/NOTES.md
git commit -m "Model-eye results"
git push
```
Then send me the `summary.txt` contents, or just tell me the results are in.
