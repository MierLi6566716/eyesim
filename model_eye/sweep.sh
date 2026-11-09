#!/bin/bash
# Runs on the Pi, from ~/eyesim. Takes one whole block of model-eye photos:
# tells you which number to turn the model eye to, waits for Enter, shoots,
# and at the end makes the small crops for the Mac.
#
#   ./sweep.sh A            # run block A (see the table below)
#   ./sweep.sh A -2.5       # start block A at -2.5 (after stopping half way)
#   ./sweep.sh list         # show all blocks
#
# Camera settings come from ~/eyesim/camera.env, one line, e.g.
#   GAIN=4 EXPOSURE=20000 VERTEX=10
# (VERTEX = mm from the trial lens to the model eye's front lens.)
#
# Settings already photographed in a block are skipped, so re-running a block
# just fills the gaps. To retake one, delete its folder first.

set -e
cd ~/eyesim

#      id  folder name                 dist lens pupil shots from to   step
BLOCKS="
A      A_37cm_plus4_7mm             37   4    7     20    0    -5   0.25
B      B_37cm_nolens_7mm            37   0    7     20    4    -5   0.25
C1     C1_37cm_nolens_5mm           37   0    5     20    4    -5   0.5
C2     C2_37cm_plus5_5mm            37   5    5     20    0    -5   0.5
D1     D1_30cm_nolens_7mm           30   0    7     20    4    -5   0.5
D2     D2_30cm_plus5_7mm            30   5    7     20    0    -5   0.5
E1     E1_50cm_nolens_7mm           50   0    7     20    4    -5   0.5
E2     E2_50cm_plus4_7mm            50   4    7     20    0    -5   0.5
R      R_37cm_plus4_7mm_repeat      37   4    7     20    0    -5   0.5
"

if [ "$1" = list ] || [ -z "$1" ]; then
  echo "  id  folder                       cm  lens pupil shots from  to  step"
  echo "$BLOCKS" | sed '/^$/d' | sed 's/^/  /'
  exit 0
fi
ROW=$(echo "$BLOCKS" | awk -v id="$1" '$1 == id')
[ -z "$ROW" ] && { echo "no block '$1'. ./sweep.sh list"; exit 1; }
read -r ID NAME DIST LENS PUPIL SHOTS FROM TO STEP <<< "$ROW"
[ -n "$2" ] && FROM=$2

[ -f camera.env ] || { echo 'make camera.env first, e.g.:  echo "GAIN=4 EXPOSURE=20000 VERTEX=10" > camera.env'; exit 1; }
# shellcheck disable=SC1091
. ./camera.env
: "${GAIN:?camera.env needs GAIN}" "${EXPOSURE:?camera.env needs EXPOSURE}"
VERTEX=${VERTEX:-10}

OUT=model_eye/raw/$NAME
mkdir -p "$OUT"
DIALS=$(python3 -c "
import sys; a, b, s = map(float, sys.argv[1:]); s = abs(s) if b >= a else -abs(s)
n = int(round((b - a) / s)); print(' '.join('%+g' % (round(a + i * s, 2) + 0.0) for i in range(n + 1)))" "$FROM" "$TO" "$STEP")

if [ "$LENS" = 0 ]; then LENSTXT="NO lens in any groove"; else LENSTXT="the +$LENS.00 lens in the groove CLOSEST to the barrel"; fi
cat <<EOF

================  BLOCK $ID: $NAME  ================
  Model eye front lens at   $DIST.0 cm from the camera lens (your tape mark)
  Pupil handle on           $PUPIL mm
  Trial lens                $LENSTXT
  Camera                    gain $GAIN, exposure $EXPOSURE us; $SHOTS shots per setting
  Scale settings            $DIALS
  Photos go to             ~/eyesim/$OUT/
EOF
read -r -p ">>> Everything set as above? Enter to start, q to quit: " ans
[ "$ans" = q ] && exit 0

t0=$(date +%s)
for DIAL in $DIALS; do
  if ls -d "$OUT"/*-true"$DIAL"-p* >/dev/null 2>&1; then
    echo "    $DIAL already done, skipping"; continue
  fi
  echo
  read -r -p ">>> Turn the scale to $DIAL at the Marker, hands off, Enter (s = skip, q = stop): " ans
  [ "$ans" = q ] && break
  [ "$ans" = s ] && continue
  python3 eye_capture.py --eye model --true-rx "$DIAL" --lens "$LENS" --pupil-mm "$PUPIL" \
    --dist "$DIST" --vertex-mm "$VERTEX" --shots "$SHOTS" --gain "$GAIN" --exposure-us "$EXPOSURE" \
    --out "$OUT" --note "block $ID"
done
echo
echo "Photos took $(( ($(date +%s) - t0) / 60 )) min. Making the crops for the Mac ..."
python3 crop_sessions.py "$OUT"
echo
echo "Block $ID finished: $(ls -d "$OUT"/*/ | wc -l) settings in ~/eyesim/$OUT"
df -h ~ | tail -1 | awk '{print "Free space on the Pi: " $4}'
