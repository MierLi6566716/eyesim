"""
Server-rendered shareable "result card" for a Case: composites the source /
blurred / pre-distorted thumbnails plus the prescription into one PNG, sized
for social sharing (1200x630, standard OG-image dimensions).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

from ..models import Case
from .storage import LOCAL_STORAGE_DIR

CARD_W, CARD_H = 1200, 630
BG = (10, 12, 14)
FG_ACCENT = (0, 255, 200)
FG_DIM = (140, 150, 150)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _load_image(url: str) -> Image.Image:
    if url.startswith("/static/"):
        data = (LOCAL_STORAGE_DIR / url.removeprefix("/static/")).read_bytes()
    else:
        data = httpx.get(url, timeout=10.0).content
    return Image.open(BytesIO(data)).convert("RGB")


def render_card(case: Case) -> bytes:
    card = Image.new("RGB", (CARD_W, CARD_H), BG)
    draw = ImageDraw.Draw(card)

    draw.text((40, 36), "EYESIM LAB", font=_font(28), fill=FG_ACCENT)
    if case.S is not None:
        rx = f"S {case.S:+.2f} D   C {case.C:+.2f} D   AXIS {case.axis:.0f}°"
    else:
        rx = f"M {case.M:+.2f}   J0 {case.J0:+.2f}   J45 {case.J45:+.2f}"
    draw.text((40, 80), rx, font=_font(22), fill=(255, 255, 255))
    if case.confidence is not None:
        draw.text(
            (40, 112), f"confidence {case.confidence:.0f}% (experimental estimate)",
            font=_font(15), fill=FG_DIM,
        )

    thumb_size = 340
    panels = [
        (case.source_image_url, "source"),
        (case.blurred_image_url, "what that eye sees"),
        (case.predistorted_image_url, "pre-distorted (corrected)"),
    ]
    x = 40
    for url, label in panels:
        if url:
            img = _load_image(url)
            img.thumbnail((thumb_size, thumb_size))
            card.paste(img, (x, 170))
            draw.text((x, 170 + thumb_size + 8), label, font=_font(15), fill=FG_DIM)
        x += thumb_size + 30

    buf = BytesIO()
    card.save(buf, format="PNG")
    return buf.getvalue()
