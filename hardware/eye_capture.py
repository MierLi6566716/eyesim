#!/usr/bin/env python3
"""
Takes the eye photos. Runs on the Raspberry Pi.

For each shot it grabs three raw frames: H LED on, V LED on, both off (the
"dark" frame removes room light). The camera keeps running with fixed
exposure while the Arduino flips the LEDs, so one shot takes about a second.
Frames are the raw sensor numbers (2x2 binned, saved as .npy), so brightness
is linear with no processing from the camera.

Works with the IMX708 NoIR (Camera Module 3) and the IMX477 HQ camera; it
reads the sensor's bit depth itself. The camera must see 850 nm, so no IR-cut
filter (run --check to find out).

  python3 eye_capture.py --check              # H, V and dark frame + brightness numbers
  python3 eye_capture.py --aim                # live: where the bright spot is, to centre the eye
  python3 eye_capture.py --focus              # live sharpness number, for manual-focus lenses
  python3 eye_capture.py --scale              # photo of a ruler at the eye position
  python3 eye_capture.py --eye right --lens 0            # a session with no trial lens
  python3 eye_capture.py --eye right --lens +3           # +3.00 D trial lens in front of the eye
  python3 eye_capture.py --eye right --lens -2 --cyl -1 --axis 90
  python3 eye_capture.py --eye model --true-rx -3 --pupil-mm 6 --shots 10   # model eye, dial at -3 D
  python3 eye_capture.py --video 60 --eye left --gain 10 --exposure-us 8000         # 60 s, ~7 H/V pairs per second
  python3 eye_capture.py --video 60 --eye left --gain 12 --exposure-us 6000 --fast  # ~15 pairs/s, coarser pixels

Your rig's measurements go in rig.json next to this script (see README):
  {"ecc_h_mm": 10.0, "ecc_v_mm": 10.0, "aperture_mm": 2.6, "dist_cm": 37,
   "px_per_mm": 21.4, "px_per_mm_dist_mm": 370}
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

# The Arduino shows up as ttyACM0 (Uno R3 / Leonardo) or ttyUSB0 (clones with a CH340).
SERIAL_PORT = next((p for p in ("/dev/ttyACM0", "/dev/ttyUSB0") if Path(p).exists()), "/dev/ttyUSB0")
BAUD = 115200
RIG = Path(__file__).with_name("rig.json")
CAL = Path(__file__).with_name("calibration")      # check_* and scale_* photos go here


class Leds:
    """The Arduino: sends H1/H0/V1/V0 over serial."""

    def __init__(self):
        import serial
        self.ser = serial.Serial(SERIAL_PORT, BAUD, timeout=5)
        time.sleep(2.0)                      # the Arduino resets when the port opens
        self.ser.reset_input_buffer()
        self.set(None)

    def set(self, which):
        """which: "H", "V" or None (both off)."""
        for ch in ("H", "V"):
            self.ser.write(("%s%d\n" % (ch, 1 if ch == which else 0)).encode())
            self.ser.readline()

    def close(self):
        self.set(None)
        self.ser.close()


class Camera:
    """Full-resolution raw stream with fixed exposure, gain and focus."""

    def __init__(self, exposure_us, gain, dist_cm):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        size = self.cam.sensor_resolution
        fmt = str(self.cam.sensor_format)                # e.g. SRGGB12_CSI2P or SBGGR10_CSI2P
        self.bits = int(re.search(r"(\d+)", fmt.split("_")[0]).group(1))
        self.white = 4 * (2 ** self.bits - 1)            # full scale after summing 2x2 pixels
        self.model = self.cam.camera_properties.get("Model", "?")
        cfg = self.cam.create_still_configuration(raw={"size": size, "format": fmt.replace("_CSI2P", "")},
                                                  buffer_count=2, queue=False)
        self.cam.configure(cfg)
        controls = {"AeEnable": False, "AwbEnable": False, "ExposureTime": int(exposure_us),
                    "AnalogueGain": float(gain), "ColourGains": (1.0, 1.0)}
        if "LensPosition" in self.cam.camera_controls:   # IMX708: focus on the eye, in dioptres
            controls.update({"AfMode": 0, "LensPosition": 100.0 / dist_cm})
        self.cam.set_controls(controls)
        self.cam.start()
        time.sleep(1.0)
        self.width = size[0]

    def raw(self):
        """One frame, each 2x2 Bayer block summed into one pixel, uint16."""
        a = self.cam.capture_array("raw")
        if a.dtype == np.uint8:
            a = a.view(np.uint16)
        a = a[:, :self.width].astype(np.uint32)
        if a.max() >= 2 ** self.bits:            # some versions left-align the bits
            a >>= 16 - self.bits
        b = a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]
        return b.astype(np.uint16)

    def fresh(self, skip=2):
        """Throw away frames that started before the LEDs changed."""
        for _ in range(skip):
            self.cam.capture_array("raw")
        return self.raw()

    def close(self):
        self.cam.stop()
        self.cam.close()


def preview(img, path):
    """8-bit PNG to look at (not used for the analysis)."""
    from PIL import Image
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.95)
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    Image.fromarray((255 * x ** 0.5).astype(np.uint8)).save(path)


def levels(img, white):
    return "max %5d (%3.0f%% of full)  99.9th pct %5d  clipped %.3f%%" % (
        img.max(), 100.0 * img.max() / white, np.percentile(img, 99.9), 100.0 * (img >= 0.98 * white).mean())


SEARCH = (500, 400)     # half-width, half-height of the central area searched for the eye


def bright_spot(img, search=SEARCH):
    """(x, y) of the brightest blob within +-search px of the frame centre
    (numpy only: 4x4 binned, 3x3 box blur, argmax). The eye is centred with
    --aim, so bright things near the frame edge are ignored."""
    H, W = img.shape
    ox, oy = (max(0, W // 2 - search[0]), max(0, H // 2 - search[1])) if search else (0, 0)
    img = img[oy:H - oy, ox:W - ox]
    h, w = (s // 4 * 4 for s in img.shape)
    b = img[:h, :w].astype(np.float32).reshape(h // 4, 4, w // 4, 4).sum((1, 3))
    p = np.pad(b, 1, mode="edge")
    b = sum(p[i:i + b.shape[0], j:j + b.shape[1]] for i in range(3) for j in range(3))
    j, i = np.unravel_index(np.argmax(b), b.shape)
    return ox + 4 * i + 2, oy + 4 * j + 2, float(b[j, i]) / 144.0


def px_per_mm_at(rig, dist_cm):
    if rig.get("px_per_mm"):
        return rig["px_per_mm"] * rig.get("px_per_mm_dist_mm", 10.0 * dist_cm) / (10.0 * dist_cm)
    return None


def where(img, dark, rig, dist_cm):
    """Text: where the brightest spot is relative to the image centre."""
    x, y, _ = bright_spot(np.clip(img.astype(np.int32) - dark.astype(np.int32), 0, None))
    h, w = img.shape
    dx, dy = x - w // 2, y - h // 2
    k = px_per_mm_at(rig, dist_cm)
    mm = " (%+5.1f mm, %+5.1f mm)" % (dx / k, dy / k) if k else ""
    ok = "OK, centred" if max(abs(dx), abs(dy)) < 100 else "move the eye (aim for both within 100 px)"
    return "spot off centre by x %+5d px, y %+5d px%s   %s" % (dx, dy, mm, ok)


def aim_loop(cam, leds, rig, dist_cm):
    print("Live position of the brightest spot in the H-LED image (the model eye's\n"
          "reflection). Slide / raise the model eye until both numbers are near 0\n"
          "(within 100 px). If a number grows when you move, go the other way. Ctrl-C to stop.")
    leds.set(None)
    dark = cam.fresh()
    leds.set("H")
    cam.fresh()
    try:
        while True:
            print("\r" + where(cam.raw(), dark, rig, dist_cm) + "   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")


def focus_loop(cam):
    from scipy.ndimage import laplace
    print("Hold some small print where your eye will be. Turn the focus ring until the\n"
          "number peaks, then lock the ring. Ctrl-C to stop.")
    best = 1e-9
    while True:
        a = cam.raw().astype(np.float32)
        h, w = a.shape
        c = a[h // 2 - 200:h // 2 + 200, w // 2 - 200:w // 2 + 200]
        s = float(laplace(c / (c.mean() + 1e-6)).var())
        best = max(best, s)
        print("\rsharpness %8.4f  best %8.4f  %-50s" % (s, best, "#" * int(50 * s / best)), end="", flush=True)



# ------------------------------------------------------------------ video mode
def clean_frames(frames, switches, readout_ns, margin_ns=1_000_000):
    """
    Which frames saw only one LED state. frames: list of (sensor_ts_ns, exposure_ns);
    switches: list of (time_ns, state) in time order, state in {"H", "V", None}.
    With a rolling shutter a frame collects light from its first row's start
    (sensor_ts) to its last row's end (sensor_ts + readout + exposure). A frame
    is clean if no LED switch falls inside that window (+- margin); its label is
    the state in force. Returns a list of state or "mixed" per frame.
    """
    out = []
    times = [t for t, _ in switches]
    for ts, exp in frames:
        start, end = ts - margin_ns, ts + readout_ns + exp + margin_ns
        i = np.searchsorted(times, start, side="right") - 1          # last switch before the window
        if i < 0 or (i + 1 < len(times) and times[i + 1] <= end):
            out.append("mixed")
        else:
            out.append(switches[i][1])
    return out


def record_video(args, rig, dist_cm):
    """
    Continuous H, V, H, V ... frames (a dark frame every DARK_EVERY pairs) for
    args.video seconds, using the sensor's 2x2-binned 2304x1296 mode, so the
    pixel scale is the same as the still sessions. Each LED state is held until
    one clean frame arrives (no LED switch during its exposure, from the frame
    timestamps), then the next state is switched on. Only a ROI x ROI box around
    the eye is kept, written to a memory-mapped file as it comes in.

    UNTESTED ON THE PI (written 12 Nov 2026 with the Pi offline). Check the
    printed rate and the "clean" share on the first run.
    """
    from picamera2 import Picamera2
    ROI, DARK_EVERY = 384, 8
    cam = Picamera2()
    # Default: the 2304x1296 binned mode (same pixel scale as the stills, max ~56 fps -> ~7 H/V pairs/s).
    # --fast: the 1536x864 mode (max ~120 fps -> ~15 pairs/s) at 2/3 of the pixel scale.
    want = 1536 if args.fast else 2304
    mode = min((m for m in cam.sensor_modes if m["size"][0] >= want), key=lambda m: m["size"][0])
    size = mode["size"]
    bits = int(re.search(r"(\d+)", str(mode["format"]).split("_")[0]).group(1))
    frame_us = int(max(args.exposure_us + 2500, 1e6 / mode["fps"] + 50))   # the mode's max rate caps it
    cfg = cam.create_video_configuration(raw={"size": size, "format": str(mode["format"]).replace("_CSI2P", "")},
                                         buffer_count=6, controls={"FrameDurationLimits": (frame_us, frame_us)})
    cam.configure(cfg)
    controls = {"AeEnable": False, "AwbEnable": False, "ExposureTime": int(args.exposure_us),
                "AnalogueGain": float(args.gain), "ColourGains": (1.0, 1.0)}
    if "LensPosition" in cam.camera_controls:
        controls.update({"AfMode": 0, "LensPosition": 100.0 / dist_cm})
    cam.set_controls(controls)
    cam.start()
    time.sleep(1.0)
    readout_ns = int(frame_us * 1000 * 0.9)          # rolling-shutter readout ~ frame time (conservative)

    def grab():
        req = cam.capture_request()
        try:
            a = req.make_array("raw")
            md = req.get_metadata()
        finally:
            req.release()
        if a.dtype == np.uint8:
            a = a.view(np.uint16)
        a = a[:, :size[0]]
        if a.max() >= 2 ** bits:
            a = a >> (16 - bits)
        return a, int(md["SensorTimestamp"]), int(md["ExposureTime"]) * 1000

    leds = Leds()
    switches = []
    def switch(state):
        leds.set(state)
        switches.append((time.clock_gettime_ns(time.CLOCK_BOOTTIME), state))

    # Find the eye once (H minus dark), then keep a fixed box around it.
    switch(None); [grab() for _ in range(3)]; dark0 = grab()[0].astype(np.int32)
    switch("H"); [grab() for _ in range(3)]; h0 = grab()[0].astype(np.int32)
    cx, cy, _ = bright_spot(np.clip(h0 - dark0, 0, None))
    H, W = h0.shape
    x0, y0 = int(np.clip(cx - ROI // 2, 0, W - ROI)), int(np.clip(cy - ROI // 2, 0, H - ROI))

    name = time.strftime("%Y%m%d-%H%M%S-") + "video-%s-%dcm-S%+g" % (args.eye, round(dist_cm), args.lens)
    folder = Path(args.out) / name
    folder.mkdir(parents=True)
    n_max = int(args.video * 1e9 / (frame_us * 1000)) + 10
    store = np.lib.format.open_memmap(folder / "frames.npy", mode="w+", dtype=np.uint16, shape=(n_max, ROI, ROI))
    rows, k, n_pairs = [], 0, 0
    order = ["H", "V"]
    state_i, state = 0, "H"
    print("Recording %.0f s. Look at the target you were told to; blink normally." % args.video)
    t_end = time.time() + args.video
    switch(state)
    while time.time() < t_end and k < n_max:
        a, ts, exp = grab()
        lab = clean_frames([(ts, exp)], switches, readout_ns)[0]
        store[k] = a[y0:y0 + ROI, x0:x0 + ROI]
        rows.append((k, ts, exp, lab))
        k += 1
        if lab == state:                               # got a clean frame of this state: move on
            if state == "V":
                n_pairs += 1
            if state is None:
                state = "H"
            elif state == "V" and n_pairs % DARK_EVERY == 0:
                state = None
            else:
                state_i = 1 - state_i if state is not None else 0
                state = order[state_i]
            switch(state)
    leds.close()
    cam.stop(); cam.close()
    store.flush()
    took = (rows[-1][1] - rows[0][1]) / 1e9 if len(rows) > 1 else 0
    labels = [r[3] for r in rows]
    with open(folder / "frames.csv", "w") as fh:
        fh.write("frame,sensor_ts_ns,exposure_ns,led\n")
        for r in rows:
            fh.write("%d,%d,%d,%s\n" % (r[0], r[1], r[2], {None: "dark"}.get(r[3], r[3])))
    with open(folder / "switches.csv", "w") as fh:
        fh.write("time_ns,led\n")
        for t, st in switches:
            fh.write("%d,%s\n" % (t, st or "dark"))
    (folder / "meta.json").write_text(json.dumps({
        "kind": "video", "eye": args.eye, "dist_mm": 10.0 * dist_cm,
        "ecc_h_mm": rig["ecc_h_mm"], "ecc_v_mm": rig["ecc_v_mm"], "ecc_mm": 0.5 * (rig["ecc_h_mm"] + rig["ecc_v_mm"]),
        "aperture_mm": rig["aperture_mm"],
        "px_per_mm": rig.get("px_per_mm") and rig["px_per_mm"] * size[0] / 2304.0,   # stills are 2304 wide
        "px_per_mm_dist_mm": rig.get("px_per_mm_dist_mm"), "lens_d": args.lens, "vertex_mm": args.vertex_mm,
        "exposure_us": args.exposure_us, "gain": args.gain, "frame_us": frame_us, "frames": k,
        "roi_x0": x0, "roi_y0": y0, "roi": ROI, "sensor_mode": list(size), "bits": bits,
        "white_level": 2 ** bits - 1, "bayer_raw": True, "note": args.note,
        "taken": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    n_h, n_v = labels.count("H"), labels.count("V")
    print("saved %s: %d frames in %.1f s (%.1f fps); clean H %d, V %d, dark %d, mixed %d -> %.1f H/V pairs per second"
          % (folder, k, took, k / max(took, 1e-9), n_h, n_v, labels.count(None), labels.count("mixed"),
             min(n_h, n_v) / max(took, 1e-9)))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eye", choices=["right", "left", "model"], default="right")
    ap.add_argument("--true-rx", type=float, default=None, help="model eye: refraction set on its dial, D")
    ap.add_argument("--pupil-mm", type=float, default=None, help="model eye: pupil (aperture) diameter, mm")
    ap.add_argument("--lens", type=float, default=0.0, help="trial lens SPHERE in front of the eye, D")
    ap.add_argument("--cyl", type=float, default=0.0, help="trial lens CYLINDER, D")
    ap.add_argument("--axis", type=float, default=0.0, help="cylinder axis as read on the trial frame, deg")
    ap.add_argument("--vertex-mm", type=float, default=12.0, help="trial lens to cornea")
    ap.add_argument("--dist", type=float, default=None, help="camera lens to eye, cm (default: rig.json)")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--exposure-us", type=int, default=20000)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--out", default="own_eye")
    ap.add_argument("--note", default="", help="free text saved in meta.json")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--focus", action="store_true")
    mode.add_argument("--aim", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--scale", action="store_true")
    mode.add_argument("--video", type=float, metavar="SECONDS", default=None,
                      help="record H/V pairs continuously for SECONDS (see record_video)")
    ap.add_argument("--fast", action="store_true", help="with --video: 1536x864 sensor mode, ~2x the pair rate")
    args = ap.parse_args()

    rig = json.loads(RIG.read_text()) if RIG.exists() else {}
    session = not (args.focus or args.check or args.scale or args.aim)
    if args.video:
        if not {"ecc_h_mm", "ecc_v_mm", "aperture_mm", "dist_cm"} <= rig.keys():
            raise SystemExit("rig.json needs ecc_h_mm, ecc_v_mm, aperture_mm and dist_cm (README, Pi step 4)")
        record_video(args, rig, args.dist or rig.get("dist_cm", 37.0))
        return
    if session and not {"ecc_h_mm", "ecc_v_mm", "aperture_mm", "dist_cm"} <= rig.keys():
        raise SystemExit("rig.json needs ecc_h_mm, ecc_v_mm, aperture_mm and dist_cm (README, Pi step 4)")
    dist_cm = args.dist or rig.get("dist_cm", 37.0)

    cam = Camera(args.exposure_us, args.gain, dist_cm)
    leds = None
    try:
        leds = Leds()
        if args.focus:
            leds.set("H")
            focus_loop(cam)
            return
        if args.aim:
            aim_loop(cam, leds, rig, dist_cm)
            return
        if args.check or args.scale:
            tag = "scale" if args.scale else "check"
            got = {}
            CAL.mkdir(exist_ok=True)
            for ch in (("H", "V", None) if args.check else ("H",)):
                leds.set(ch)
                got[ch or "dark"] = img = cam.fresh()
                np.save(CAL / ("%s_%s.npy" % (tag, ch or "dark")), img)
                preview(img, CAL / ("%s_%s.png" % (tag, ch or "dark")))
                print("%-4s %s" % (ch or "dark", levels(img, cam.white)))
            print("camera %s, %d-bit. Wrote %s/%s_*.png (look at these) and %s_*.npy (for the Mac)"
                  % (cam.model, cam.bits, CAL.name, tag, tag))
            if args.check:
                # The sensor reads ~64 (10-bit) per pixel with no light at all; 4 pixels are summed.
                black = 4 * 64 * 2 ** (cam.bits - 10)
                room = max(0.0, float(np.percentile(got["dark"], 99.9)) - black) / cam.white
                print("room light (dark frame): %.1f%% of full  %s" % (
                    100 * room, "OK" if room < 0.05 else "TOO BRIGHT: lights off, curtains shut, no lamps"))
                lift = float(np.percentile(got["H"], 99.99)) - float(np.percentile(got["dark"], 99.99))
                if lift < 0.01 * cam.white:
                    print("\nThe H frame is no brighter than the dark one: the camera is not seeing the\n"
                          "LEDs. Check the wiring, or the camera has an IR-cut filter (README, Pi step 1).")
                    return
                print(where(got["H"], got["dark"], rig, dist_cm))
                x, y, _ = bright_spot(np.clip(got["H"].astype(np.int32) - got["dark"].astype(np.int32), 0, None))
                hh, ww = got["H"].shape
                x0, y0 = int(np.clip(x - 256, 0, ww - 512)), int(np.clip(y - 256, 0, hh - 512))
                preview(got["H"][y0:y0 + 512, x0:x0 + 512], CAL / "check_H_box.png")
                print("wrote calibration/check_H_box.png: the 512 px box the Mac will get. The pupil must be in it.")
                hmax = float(got["H"].max()) / cam.white
                print("exposure: " + ("CLIPPED, lower --gain or --exposure-us" if hmax >= 0.98 else
                                      "OK" if hmax > 0.2 else
                                      "dim (peak %.0f%%), raise --gain or --exposure-us" % (100 * hmax)))
            return

        if args.eye == "model" and args.true_rx is None:
            raise SystemExit("--eye model needs --true-rx (the number on the model eye's dial)")
        name = time.strftime("%Y%m%d-%H%M%S-") + "%s-%dcm-S%+g" % (args.eye, round(dist_cm), args.lens)
        if args.true_rx is not None:
            name += "-true%+g" % args.true_rx
        if args.pupil_mm:
            name += "-p%gmm" % args.pupil_mm
        if args.cyl:
            name += "C%+gx%03.0f" % (args.cyl, args.axis)
        folder = Path(args.out) / name
        folder.mkdir(parents=True)
        print("Look at the far target, not the camera. Taking %d shots ..." % args.shots)
        t0 = time.time()
        for k in range(args.shots):
            for ch in ("H", "V", None):
                leds.set(ch)
                np.save(folder / ("%02d_%s.npy" % (k, ch or "dark")), cam.fresh())
        leds.set(None)
        took = time.time() - t0
        h0 = np.load(folder / "00_H.npy")
        preview(h0, folder / "preview_H.png")
        print("H frame: " + levels(h0, cam.white))
        (folder / "meta.json").write_text(json.dumps({
            "eye": args.eye, "dist_mm": 10.0 * dist_cm,
            "ecc_h_mm": rig["ecc_h_mm"], "ecc_v_mm": rig["ecc_v_mm"],
            "ecc_mm": 0.5 * (rig["ecc_h_mm"] + rig["ecc_v_mm"]),
            "aperture_mm": rig["aperture_mm"], "px_per_mm": rig.get("px_per_mm"),
            "px_per_mm_dist_mm": rig.get("px_per_mm_dist_mm"),
            "lens_d": args.lens, "lens_cyl": args.cyl, "lens_axis": args.axis, "vertex_mm": args.vertex_mm,
            "exposure_us": args.exposure_us, "gain": args.gain, "shots": args.shots,
            "camera": cam.model, "white_level": cam.white, "seconds": round(took, 2),
            "true_rx": args.true_rx, "pupil_mm": args.pupil_mm,
            "note": args.note, "taken": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        print("saved %s  (%.1f s for %d shots)" % (folder, took, args.shots))
    finally:
        if leds:
            leds.close()
        cam.close()


if __name__ == "__main__":
    main()
