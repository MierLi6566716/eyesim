#!/bin/bash
# Runs on the Pi. Takes one session per model-eye dial setting, asking you to
# turn the dial before each one.
#
#   ./sweep.sh DIST_CM LENS_D PUPIL_MM SHOTS DIAL_FROM DIAL_TO STEP
#   ./sweep.sh 37 0 6 10 6 -6 0.5      # 37 cm, no lens, 6 mm pupil, 10 shots, dial +6 down to -6
#
# Camera settings come from GAIN and EXPOSURE (set them once after --check):
#   GAIN=4 EXPOSURE=20000 ./sweep.sh ...
# Photos go to ~/eyesim/model_eye/<time>-model-<dist>cm-S<lens>-true<dial>-p<pupil>mm/

set -e
if [ $# -ne 7 ]; then
  sed -n '2,11p' "$0"; exit 1
fi
DIST=$1; LENS=$2; PUPIL=$3; SHOTS=$4; FROM=$5; TO=$6; STEP=$7
GAIN=${GAIN:?set GAIN first, e.g. GAIN=4}
EXPOSURE=${EXPOSURE:?set EXPOSURE first, e.g. EXPOSURE=20000}
cd ~/eyesim
DIALS=$(python3 -c "
import sys; a, b, s = map(float, sys.argv[1:]); s = abs(s) if b >= a else -abs(s)
n = int(round((b - a) / s)); print(' '.join('%+g' % round(a + i * s, 2) for i in range(n + 1)))" "$FROM" "$TO" "$STEP")
echo "dial settings: $DIALS"
for DIAL in $DIALS; do
  read -r -p ">>> Set the model eye dial to $DIAL D (dist $DIST cm, lens $LENS, pupil $PUPIL mm), then press Enter (s = skip, q = quit): " ans
  [ "$ans" = q ] && break
  [ "$ans" = s ] && continue
  python3 eye_capture.py --eye model --true-rx "$DIAL" --lens "$LENS" --pupil-mm "$PUPIL" \
    --dist "$DIST" --shots "$SHOTS" --gain "$GAIN" --exposure-us "$EXPOSURE" --out model_eye
done
echo "done. Free space left on the Pi:"; df -h ~ | tail -1
