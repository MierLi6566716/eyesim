#!/bin/bash
# Mac: copies the Pi's newest pictures into model_eye/look/ and opens them.
#   ./model_eye/look.sh
D=~/Desktop/eyesim/model_eye/look
mkdir -p "$D"; rm -f "$D"/*.png
scp -q rpi:eyesim/calibration/check_H.png "$D/1_check_H_whole_frame.png" 2>/dev/null
scp -q rpi:eyesim/calibration/check_H_box.png "$D/2_check_H_box_for_mac.png" 2>/dev/null
scp -q rpi:eyesim/calibration/scale_H.png "$D/3_ruler.png" 2>/dev/null
NEWEST=$(ssh rpi 'ls -td ~/eyesim/model_eye/crops/*/*/ 2>/dev/null | head -1')
[ -n "$NEWEST" ] && scp -q "rpi:${NEWEST}preview_H.png" "$D/4_newest_cropped_session.png"
ls "$D"
open "$D"/*.png
