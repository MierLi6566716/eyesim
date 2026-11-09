# Model-eye photo day (XSJ-501)

A model eye has a **known** refraction (the number on its scale), never
moves, never blinks and never focuses. If the network reads it correctly, what
still goes wrong on a real eye is real-eye stuff. If it reads it wrong, the rig
or the simulator is off, and these photos say which.

Plan: **10 blocks, 170 settings, 20 shots each (about 10,000 photo pairs)**,
about 2 h 15 min of shooting plus 30 min of setup. Distances are **37, 30 and
25 cm** only (nothing farther: a glasses-style device has to work up close).
Stronger plus lenses make up for being close: they push the model eye + lens
past the dark band, where the crescent shows.

---

## 0. What goes where

You use **two terminal windows on the Mac**:

| Window | How to open it | Prompt looks like | Commands marked |
|---|---|---|---|
| **PI** | `ssh rpi` then `cd ~/eyesim` | `mierli@eyesim:~/eyesim $` | **[PI]** |
| **MAC** | `cd ~/Desktop/eyesim` | `... eyesim %` | **[MAC]** |

`rpi` already points to `mierli@172.18.75.155` (in `~/.ssh/config`).

Folders:

```
Pi   ~/eyesim/
       eye_capture.py  sweep.sh  crop_sessions.py  pack_sessions.py   the programs
       rig.json        camera.env                                     your settings
       calibration/    check_H.png, check_H_box.png, scale_H.npy ...  test photos
       own_eye/        your own eye sessions from October             (kept)
       model_eye/raw/A_37cm_plus4_7mm/<one folder per setting>/       full photos, stay on the Pi
       model_eye/crops/A_37cm_plus4_7mm/<one folder per setting>/     512x512 copies for the Mac

Mac  ~/Desktop/eyesim/model_eye/
       sessions/A_37cm_plus4_7mm/...     the crops, copied from the Pi   (not on GitHub)
       results/A_37cm_plus4_7mm/         report.png, summary.txt, results.csv
       results/SUMMARY.txt               all blocks in one file
       NOTES.md                          fill in during the day
       look/                             pictures copied by look.sh  (not on GitHub)
```

The full photos are about 61 GB in total; the Pi has 165 GB free. The crops
come to about 3.5 GB; the Mac has 16 GB free.

## 1. The blocks

| Block | Distance | Pupil | Trial lens | Scale settings | Count | Time |
|---|---|---|---|---|---|---|
| **A** | 37 cm | 7 mm | **+4.00** | 0 → −5, every 0.25 | 21 | 16 min |
| B | 37 cm | 7 mm | **+8.00** | 0 → −5, every 0.5 | 11 | 8 min |
| C | 37 cm | 7 mm | none | +4 → −5, every 0.25 | 37 | 28 min |
| D1 | 37 cm | **5 mm** | none | +4 → −5, every 0.5 | 19 | 14 min |
| D2 | 37 cm | **5 mm** | **+5.00** | 0 → −5, every 0.5 | 11 | 8 min |
| E1 | **30 cm** | 7 mm | none | +4 → −5, every 0.5 | 19 | 14 min |
| E2 | **30 cm** | 7 mm | **+5.00** | 0 → −5, every 0.5 | 11 | 8 min |
| F1 | **25 cm** | 7 mm | none | +4 → −5, every 0.5 | 19 | 14 min |
| F2 | **25 cm** | 7 mm | **+6.00** | 0 → −5, every 0.5 | 11 | 8 min |
| R | 37 cm | 7 mm | +4.00 | 0 → −5, every 0.5 | 11 | 8 min |

- **A is the most important**, so it goes first. Model eye + lens runs −4 to
  −9 D, like your own eye, and that's where the network failed on you.
- **B** (+8 lens) reaches −8 to −13 D: strong myopia, far past the dark band.
- The order needs the fewest changes: the distance moves only after D2.
- **R** repeats A after you take the model eye off the table and put it back.
  That measures how repeatable the whole thing is (a paper needs that number).
- **Not the 3 mm pupil**: at 37 cm it would be dark from about −10 to +5 D.
- **Not closer than 25 cm** yet: the networks were trained on 25 to 100 cm.
  20 cm needs a retrain first.

The dark band (no crescent) moves and widens as you get closer. Every lens
block is chosen to cross its edge part-way through the sweep (model eye +
lens, in IR, from the simulator):

| Distance | 7 mm pupil: dark between | 5 mm pupil: dark between |
|---|---|---|
| 37 cm | −4.75 and +1.5 D | −6.75 and +2.5 D |
| 30 cm | −6.5 and +2.0 D | −8.75 and +3.0 D |
| 25 cm | −8.0 and +2.25 D | −10.75 and +3.75 D |

Above the band: a crescent on the side **away from the LED** (left in the H
photo). Below it: a crescent on the **LED side** (right in H).

**Lenses you need:** +4.00, +5.00, +6.00, +8.00. Missing one? Tell me which
powers you have and I'll change the block.

---

## 2. Things to have on the table

- The XSJ-501 and the **+4.00, +5.00, +6.00 and +8.00** trial lenses
- A tape measure, a **mm ruler**, masking tape and a pen
- A few books or a box to raise the model eye to the camera's height
- Your phone (its flashlight is fine for reading the scale)

## 3. The room

850 nm infrared comes from the **sun**, **ordinary bulbs**, **halogen lamps**
and **heaters**, not from LED lights or screens.

1. Close the curtains or blinds fully. Shoot in the evening if you can.
2. Turn off the ceiling light. No ordinary bulbs, halogen lamps or heaters on.
3. To see the scale: use your phone flashlight or a **white LED** desk lamp
   pointed at the table, **not at the camera or the model eye**. Leave it
   exactly as it is for the whole block. Don't switch it on and off between
   settings.
4. Don't stand between the camera and the model eye. Don't lean on or bump
   the table while a setting is being shot (about 30 s each).

`--check` prints a `room light` line: it must say **OK**.

---

## 4. Setup

### Step 1 [MAC]: get the newest code and check the Pi

```bash
cd ~/Desktop/eyesim
git pull
scp hardware/eye_capture.py hardware/crop_sessions.py model_eye/sweep.sh scripts/pack_sessions.py rpi:~/eyesim/
ssh rpi 'chmod +x ~/eyesim/sweep.sh; df -h ~ | tail -1; ls /dev/ttyACM0'
```
The last line must show `/dev/ttyACM0` (the Arduino). The free space must be
over 60 GB.

Optional, sets the Pi's clock (it's a month behind, which only affects the
folder names):
```bash
ssh -t rpi "sudo date -s '$(date '+%Y-%m-%d %H:%M:%S')'"
```

### Step 2 (table): mark the distances

1. Put the camera rig at one end of the table, facing along the table, and
   **tape it down**.
2. Lay a long strip of masking tape on the table, straight out from the
   camera lens.
3. With the tape measure starting at the **front glass of the camera lens**,
   mark the strip at **25.0**, **30.0** and **37.0 cm**.

### Step 3 (table): place the model eye at 37 cm

1. Pupil handle (the flat sheet under the front of the barrel) on **7 mm**.
   Look into the front to check the hole size.
2. **No lens** in any groove.
3. Turn the scale to **0** (at the Marker).
4. Put the model eye on the strip so its **front lens** (the glass at the very
   front of the barrel) is exactly over the **37.0 cm** mark. Measure again
   from the camera lens to that glass, along the line: **37.0 cm**.
5. **Height**: measure the height of the centre of the camera lens above the
   table. Raise the model eye with books until the **centre of its barrel** is
   at the same height (±0.5 cm).
6. **Pointing**: crouch behind the model eye and sight along the top of the
   barrel like a rifle. It must point straight at the **camera lens** (not at
   an LED). The barrel must be level.

### Step 4 [PI]: centre the model eye in the picture

```bash
ssh rpi
cd ~/eyesim
python3 eye_capture.py --aim --dist 37 --gain 4 --exposure-us 20000
```
A live line shows how far the bright spot (the model eye's reflection) is from
the middle of the picture:
```
spot off centre by x  +312 px, y   -40 px ( +63.4 mm,  -8.1 mm)   move the eye ...
```
Slide the model eye **sideways** (or the books) until **x** is within ±100.
Change its **height** until **y** is within ±100. If a number gets bigger,
move the other way. Keep it on the 37.0 cm mark the whole time. When the line
says `OK, centred`, press **Ctrl-C**.

If the numbers jump around randomly, or stay stuck near x ±500 / y ±400 (the
edge of the area it searches), the camera isn't seeing the model eye. Check
the pointing (Step 3.6) and that the room is dark.

### Step 5 (table): tape it down

Tape the model eye's base (or the books) to the table on two sides. Trace
around the base with the pen so you can put it back exactly for block R.
**From now on touch only the scale ring, the pupil handle and the lens
grooves.** When you turn the scale, hold the base with the other hand.

### Step 6: ruler photo (pixel size) [PI] + [MAC]

Hold the mm ruler flat against the front of the barrel, across the front
glass, with the marks facing the camera. Keep it still:
```bash
python3 eye_capture.py --scale --dist 37                 # [PI]
```
Take the ruler away. Then:
```bash
scp rpi:~/eyesim/calibration/scale_H.npy model_eye/scale_H_37cm.npy                          # [MAC]
/opt/anaconda3/bin/python scripts/measure_eye.py --calibrate-scale model_eye/scale_H_37cm.npy --dist-cm 37
```
Click two ruler marks far apart (e.g. 0 and 20 mm), then type their distance in
mm. It prints two numbers. Put them in `rig.json` on the Pi:
```bash
nano rig.json                                            # [PI]
```
so it reads (with your numbers in place of 4.92 and 370):
```json
{"ecc_h_mm": 10.0, "ecc_v_mm": 10.0, "aperture_mm": 2.6, "dist_cm": 37,
 "px_per_mm": 4.92, "px_per_mm_dist_mm": 370}
```
Save: **Ctrl-O**, **Enter**, **Ctrl-X**.

### Step 7 [PI]: exposure

Turn the scale to **+4** (strongest crescent, no lens):
```bash
python3 eye_capture.py --check --dist 37 --gain 4 --exposure-us 20000
```
It prints:
```
room light (dark frame): 0.8% of full  OK                <- must be OK (else: Section 3)
spot off centre by x  +12 px, y   -30 px ...  OK, centred
exposure: OK                                              <- must be OK
wrote calibration/check_H_box.png: the 512 px box the Mac will get. ...
```
- `CLIPPED`: lower `--gain` (4 → 2 → 1), then `--exposure-us`.
- `dim`: raise `--gain` (4 → 8 → 16), then `--exposure-us` (max 30000).

Repeat until it says **OK**. Then look at the pictures **[MAC]**:
```bash
./model_eye/look.sh
```
It opens `1_check_H_whole_frame.png` and `2_check_H_box_for_mac.png`. In the
box picture the **round pupil must be clearly inside** with a bright crescent
in it, and it should be round, not oval. If it's oval, the barrel isn't
pointing at the camera (Step 3.6).

Measure the **trial-lens distance** too: with a lens in the groove closest to
the barrel, measure in mm from the lens to the model eye's front glass (about
5 to 15 mm). Measure it to the nearest mm: with the +8 lens, 1 mm off moves
the answer by about 0.07 D. Then save the settings **[PI]**:
```bash
echo "GAIN=4 EXPOSURE=20000 VERTEX=10" > camera.env     # your gain, exposure and lens distance
cat camera.env
```

### Step 8 [PI]: sign check (2 minutes, important)

Run `python3 eye_capture.py --check --dist 37` with your gain and exposure
each time, then `./model_eye/look.sh` on the Mac:

| Scale | Lens | Crescent in `1_check_H_whole_frame.png` should be |
|---|---|---|
| +4 | none | on the **left** of the pupil (away from the LED) |
| −3 | none | **none**: the pupil is dark |
| −5 | +4.00 in the groove nearest the barrel | on the **right** (LED side) |

If it's the other way round, write it in NOTES.md and **tell me before going
on**. Take the lens out again.

Write in `model_eye/NOTES.md`: distance, pixel numbers, gain, exposure,
VERTEX, sign check.

---

## 5. Shooting [PI]

Every block is one command. It prints the setup, waits for Enter, then for
each setting:

```
>>> Turn the scale to -2.25 at the Marker, hands off, Enter (s = skip, q = stop):
```

1. Hold the base with one hand. Turn the scale ring with the other until the
   number sits **exactly** on the Marker line. Look at it straight on, not from
   the side.
2. Take both hands off. Wait 2 s. Press **Enter**.
3. Don't touch the table for the about 30 s it takes (20 shots). The H-frame
   brightness is printed at the end of each one.

At the end of a block it makes the crops by itself and prints the free space.

- **Stopped half way** (`q`, a typo, Wi-Fi drop)? Run the same block again:
  settings already done are skipped. To start further down: `./sweep.sh A -2.5`.
- **Retake one setting?** Delete its folder (`rm -r model_eye/raw/A_37cm_plus4_7mm/*true-2.25-*`)
  and run the block again.
- **The ssh window dropped mid-shot?** Run the block again: that setting is
  done if its folder has a `meta.json`; if not, delete the folder first.
- `./sweep.sh list` shows all blocks.

### Block A: 37 cm, +4.00 lens, 7 mm (most important)

Put the **+4.00** lens in the groove **closest to the barrel**. Hold it by the
rim and don't touch the glass. Turn it so its handle points straight down.
```bash
./sweep.sh A
```
After the first setting, run `./model_eye/look.sh` **[MAC]**. If a bright
lens reflection sits on or next to the pupil in `4_newest_cropped_session.png`:
tilt the lens slightly in its groove, delete that setting and run `./sweep.sh A`
again.

**Checkpoint after block A [MAC]** (5 minutes, catches problems before the
other 2 hours):
```bash
rsync -a --partial rpi:~/eyesim/model_eye/crops/ model_eye/sessions/
/opt/anaconda3/bin/python scripts/model_eye_report.py model_eye/sessions/A_37cm_plus4_7mm --reuse-clicks --out model_eye/results/A_37cm_plus4_7mm
open model_eye/results/A_37cm_plus4_7mm/report.png
```
A window opens once. Click the **centre of the pupil**, then **3 points on the
pupil edge**, spread around it. Then click anywhere to accept. The other
settings reuse these clicks. Carry on whatever it shows; tell me later
if the right panel says the pupil is more than 10 % off.

### Block B: 37 cm, **+8.00 lens**, 7 mm

Swap the +4.00 for the **+8.00** in the same groove (closest to the barrel,
handle down). Do the reflection check again after the first setting.
```bash
./sweep.sh B
```

### Block C: 37 cm, no lens, 7 mm

Take the lens **out**.
```bash
./sweep.sh C
```

### Blocks D1, D2: 37 cm, **5 mm pupil**

Flip the pupil handle to **5 mm**. Nothing else moves.
```bash
./sweep.sh D1
```
Put the **+5.00** lens in the groove closest to the barrel:
```bash
./sweep.sh D2
```
Take the lens out, flip the pupil back to **7 mm**.

### Blocks E1, E2: **30 cm**

1. Un-tape the model eye, slide it straight toward the camera until its front
   glass is over the **30.0 cm** mark. Same height, still pointing at the lens.
2. `python3 eye_capture.py --aim --dist 30` (with your gain/exposure): centre
   it again, Ctrl-C. Tape it down.
3. Scale on **+4**, `python3 eye_capture.py --check --dist 30 --gain ... --exposure-us ...`.
   It's closer, so it's brighter. If it says CLIPPED, lower the gain and update
   `camera.env` (keep VERTEX).

```bash
./sweep.sh E1                 # no lens
./sweep.sh E2                 # +5.00 lens in the groove closest to the barrel
```

### Blocks F1, F2: **25 cm**

Same as for E, at the **25.0 cm** mark, with `--dist 25` in `--aim` and
`--check`. Brighter again, so lower the gain if it says CLIPPED.

```bash
./sweep.sh F1                 # no lens
./sweep.sh F2                 # +6.00 lens in the groove closest to the barrel
```

### Block R: repeat at 37 cm

1. Take the model eye **completely off the table**, then put it back on your
   pen outline at **37.0 cm**: books, height, pointing (Steps 3 to 5) from
   scratch.
2. `python3 eye_capture.py --aim --dist 37`, then `--check`. Set `camera.env`
   back to the 37 cm gain and exposure.
3. +4.00 lens in the groove closest to the barrel, pupil 7 mm.

```bash
./sweep.sh R
```

**Extra time?** Ask me for a block with 40 shots; more shots of the
same setting are what we'd fine-tune on later.

---

## 6. Copy, analyse, save [MAC]

```bash
cd ~/Desktop/eyesim
rsync -a --partial rpi:~/eyesim/model_eye/crops/ model_eye/sessions/      # re-run if it drops; it resumes
/opt/anaconda3/bin/python scripts/model_eye_report.py model_eye/sessions/*/ --reuse-clicks
cat model_eye/results/SUMMARY.txt
open model_eye/results/*/report.png
```

One click window opens per block (10 in all): pupil centre, 3 edge points,
accept. Each block gets `results.csv`, `report.png` and `summary.txt` in
`model_eye/results/<block>/`, comparing both networks (`checkpoints/best.pt`
and `checkpoints/wide_m_only/best.pt`).

**How to read `report.png`:**
- **Left**: network vs scale. Dots along a line **parallel** to the dashed
  one = it works. A constant shift is fine: a plastic eye's colour error at
  850 nm isn't a human eye's. A **flat** line = the same squashing as on your
  eye.
- **Middle**: real crescent side (dots) vs the simulator (steps). If the dots
  switch sides at a different setting than the steps, the simulated dark band
  is off.
- **Right**: measured vs real pupil (7 or 5 mm). More than 10 % off = the
  pixel size (ruler photo) is wrong.

Save the results (the photos stay off GitHub):
```bash
git add model_eye/results model_eye/NOTES.md
git commit -m "Model-eye results"
git push
```
Then tell me the results are in.

The full photos stay on the Pi in `~/eyesim/model_eye/raw/`. Don't delete
them: they're the master copy.
