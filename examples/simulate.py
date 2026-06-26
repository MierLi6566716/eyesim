"""
Simulate how a given prescription sees an image.

Usage:
    python examples/simulate.py --image path/to/photo.png --S -2.0 --C -1.0 --axis 45
    python examples/simulate.py --demo        

Saves <name>_blurred.png next to the input (or in examples/ for the demo).
"""
import argparse
import os
import numpy as np


def load_image(path):
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=float) / 255.0


def make_chart(n=400):
    img = np.ones((n, n))
    y = 40
    for size in [40, 28, 20, 14, 10, 7, 5]:
        x = 40
        for _ in range((n - 80) // (size * 2)):
            img[y:y + size, x:x + size] = 0.0
            x += size * 2
        y += size + 18
        if y > n - 40:
            break
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--S", type=float, default=-2.0, help="sphere (D)")
    ap.add_argument("--C", type=float, default=0.0, help="cylinder (D)")
    ap.add_argument("--axis", type=float, default=0.0, help="axis (deg)")
    ap.add_argument("--pupil", type=float, default=2.0, help="pupil radius (mm)")
    args = ap.parse_args()

    from eyesim.optics import simulate_vision

    if args.demo or not args.image:
        image = make_chart()
        out_path = os.path.join(os.path.dirname(__file__), "demo_blurred.png")
    else:
        image = load_image(args.image)
        base, _ = os.path.splitext(args.image)
        out_path = base + "_blurred.png"

    blurred, _ = simulate_vision(
        image, args.S, args.C, args.axis,
        pupil_radius_mm=args.pupil, grid=256, psf_crop=96,
    )

    from PIL import Image
    Image.fromarray((blurred * 255).astype(np.uint8)).save(out_path)
    print(f"Prescription S={args.S} C={args.C} axis={args.axis}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
