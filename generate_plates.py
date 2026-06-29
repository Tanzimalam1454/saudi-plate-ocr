"""
Synthetic Saudi license-plate generator for OCR recognition training.

Pipeline per plate:
    1. pick a random VALID plate (1-4 digits, 1-3 letters from the legal set)
    2. render a clean bilingual plate (Arabic top row, Latin bottom row)
    3. degrade it like a real drive-thru camera frame
       (perspective, blur, lighting, noise, JPEG, low-res)
    4. split into top (Arabic) and bottom (English) half-crops using the SAME
       split_saudi_plate() the inference pipeline uses
    5. write each half + its text label in PaddleOCR recognition format

Output:
    <out>/images/*.jpg
    <out>/train_label.txt      ("images/foo.jpg\\tLABEL" per line)
    <out>/val_label.txt
    <out>/saudi_plate_dict.txt (one char per line)

Run:
    python3 generate_plates.py --count 20000 --out dataset
    python3 generate_plates.py --count 12 --out _sample --montage   # quick eyeball
"""

import argparse
import io
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

import plate_spec as ps

# ---------------------------------------------------------------------------
# Fonts — candidates across macOS and Linux/Colab; only existing ones are kept.
# On Colab, install Arabic fonts first:
#   apt-get -y install fonts-noto-core fonts-kacst fonts-hosny-amiri
# Override either list with env vars ARABIC_FONTS / LATIN_FONTS (colon-separated).
# ---------------------------------------------------------------------------
_ARABIC_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/SFArabic.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/GeezaPro.ttc",
    # Linux / Colab
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/kacst/KacstOne.ttf",
    "/usr/share/fonts/truetype/kacst/KacstQurn.ttf",
    "/usr/share/fonts/truetype/hosny-amiri/Amiri-Regular.ttf",
]
_LATIN_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    # Linux / Colab
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _resolve_fonts(env_key, candidates):
    override = os.environ.get(env_key)
    paths = override.split(":") if override else candidates
    return [f for f in paths if os.path.exists(f)]


ARABIC_FONTS = _resolve_fonts("ARABIC_FONTS", _ARABIC_CANDIDATES)
LATIN_FONTS = _resolve_fonts("LATIN_FONTS", _LATIN_CANDIDATES)
assert ARABIC_FONTS, "No Arabic fonts found — on Colab run: apt-get install fonts-noto-core fonts-kacst"
assert LATIN_FONTS, "No Latin fonts found — on Colab run: apt-get install fonts-dejavu"

# Base render size (pre-degradation). Real plate aspect ~1.97:1.
BASE_W, BASE_H = 660, 330


# ---------------------------------------------------------------------------
# Plate content
# ---------------------------------------------------------------------------
def random_plate():
    n_d = random.randint(ps.MIN_DIGITS, ps.MAX_DIGITS)
    n_l = random.randint(ps.MIN_LETTERS, ps.MAX_LETTERS)
    digits = "".join(random.choice(ps.LATIN_DIGITS) for _ in range(n_d))
    letters = "".join(random.choice(ps.LATIN_LETTERS) for _ in range(n_l))
    return digits, letters


def _fit_font(path, target_px):
    return ImageFont.truetype(path, size=max(8, int(target_px)))


def _draw_row(draw, slots_chars, x0, x1, y_center, font, fill=(10, 10, 12)):
    """Draw chars centered in evenly spaced slots between x0..x1 (left-to-right)."""
    n = len(slots_chars)
    if n == 0:
        return
    cell = (x1 - x0) / n
    for i, ch in enumerate(slots_chars):
        if ch == " ":
            continue
        cx = x0 + cell * (i + 0.5)
        draw.text((cx, y_center), ch, font=font, fill=fill, anchor="mm")


def render_clean(digits, letters):
    """Render a clean plate. Returns PIL.Image (RGB)."""
    img = Image.new("RGB", (BASE_W, BASE_H), (250, 250, 248))
    draw = ImageDraw.Draw(img)

    # outer black border
    bw = 8
    draw.rectangle([bw // 2, bw // 2, BASE_W - bw // 2, BASE_H - bw // 2],
                   outline=(15, 15, 18), width=bw)

    # KSA blue strip on the right
    strip_w = int(BASE_W * ps.KSA_STRIP_FRAC)
    sx0 = BASE_W - strip_w
    draw.rectangle([sx0, bw, BASE_W - bw, BASE_H - bw], fill=(20, 60, 130))
    try:
        f_ksa = _fit_font(LATIN_FONTS[0], strip_w * 0.5)
        draw.text((sx0 + strip_w / 2, BASE_H * 0.72), "KSA",
                  font=f_ksa, fill=(245, 245, 245), anchor="mm")
        f_ar = _fit_font(ARABIC_FONTS[0], strip_w * 0.42)
        draw.text((sx0 + strip_w / 2, BASE_H * 0.30), "السعودية",
                  font=f_ar, fill=(245, 245, 245), anchor="mm")
    except Exception:
        pass

    # content area (excluding KSA strip)
    cx0, cx1 = bw + 6, sx0 - 6
    mid_y = BASE_H // 2

    # build slot sequence: digits + gap + letters (left-to-right)
    slots = list(digits) + [" "] + list(letters)

    char_h = BASE_H * 0.36
    f_latin = _fit_font(random.choice(LATIN_FONTS), char_h)
    f_arabic = _fit_font(random.choice(ARABIC_FONTS), char_h * 1.05)

    # top row = Arabic equivalents of the same slots
    ar_slots = []
    for ch in slots:
        if ch == " ":
            ar_slots.append(" ")
        elif ch in ps.LATIN_TO_ARABIC_DIGIT:
            ar_slots.append(ps.LATIN_TO_ARABIC_DIGIT[ch])
        else:
            ar_slots.append(ps.LATIN_TO_ARABIC[ch])

    _draw_row(draw, ar_slots, cx0, cx1, BASE_H * 0.27, f_arabic)
    _draw_row(draw, slots, cx0, cx1, BASE_H * 0.73, f_latin)

    # faint horizontal divider
    draw.line([cx0, mid_y, cx1, mid_y], fill=(120, 120, 125), width=2)
    return img


# ---------------------------------------------------------------------------
# Degradation (make synthetic look like real camera capture)
# ---------------------------------------------------------------------------
def _perspective_coeffs(src, dst):
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=float)
    B = np.array(src, dtype=float).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def degrade(img):
    w, h = img.size

    # 1) mild perspective + rotation, on a padded canvas so corners don't clip
    pad = int(max(w, h) * 0.18)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (200, 200, 200))
    canvas.paste(img, (pad, pad))
    W, H = canvas.size
    jx, jy = w * 0.10, h * 0.12
    def j(a): return random.uniform(-a, a)
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    dst = [(j(jx), j(jy)), (W + j(jx), j(jy)),
           (W + j(jx), H + j(jy)), (j(jx), H + j(jy))]
    coeffs = _perspective_coeffs(src, dst)
    canvas = canvas.transform((W, H), Image.PERSPECTIVE, coeffs,
                              Image.BICUBIC, fillcolor=(200, 200, 200))
    canvas = canvas.rotate(random.uniform(-4, 4), resample=Image.BICUBIC,
                           fillcolor=(200, 200, 200), expand=False)
    img = canvas.crop((pad // 2, pad // 2, W - pad // 2, H - pad // 2))

    # 2) lighting / color jitter
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.55, 1.35))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.4, 1.2))

    # 3) blur (focus / motion)
    if random.random() < 0.8:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 2.2)))

    # 4) low-res capture: shrink then enlarge
    scale = random.uniform(0.30, 0.75)
    sw, sh = max(20, int(img.width * scale)), max(10, int(img.height * scale))
    img = img.resize((sw, sh), Image.BILINEAR).resize(
        (BASE_W, BASE_H), Image.BILINEAR)

    # 5) sensor noise
    arr = np.asarray(img).astype(np.int16)
    arr += np.random.normal(0, random.uniform(2, 14), arr.shape).astype(np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 6) JPEG artifacts
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(28, 75))
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def build(count, out, val_frac, seed, montage):
    random.seed(seed)
    np.random.seed(seed)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    train_lines, val_lines = [], []
    montage_imgs = []

    for i in range(count):
        digits, letters = random_plate()
        clean = render_clean(digits, letters)
        plate = degrade(clean)

        arr = np.asarray(plate)[:, :, ::-1]          # RGB -> BGR for split
        top, bottom = ps.split_saudi_plate(arr)
        if top is None:
            continue

        top_img = Image.fromarray(top[:, :, ::-1])   # back to RGB
        bot_img = Image.fromarray(bottom[:, :, ::-1])

        stem = f"plate_{i:06d}"
        top_rel = f"images/{stem}_top.jpg"
        bot_rel = f"images/{stem}_bottom.jpg"
        top_img.save(os.path.join(out, top_rel), quality=92)
        bot_img.save(os.path.join(out, bot_rel), quality=92)

        ar_lbl = ps.arabic_label(digits, letters)
        la_lbl = ps.latin_label(digits, letters)

        bucket = val_lines if random.random() < val_frac else train_lines
        bucket.append(f"{top_rel}\t{ar_lbl}")
        bucket.append(f"{bot_rel}\t{la_lbl}")

        if montage and len(montage_imgs) < 12:
            montage_imgs.append((plate.copy(), la_lbl, ar_lbl))

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{count}")

    with open(os.path.join(out, "train_label.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(out, "val_label.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(os.path.join(out, "saudi_plate_dict.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ps.dictionary_chars()) + "\n")

    print(f"Done. {len(train_lines)} train / {len(val_lines)} val label rows "
          f"(2 crops per plate).")
    print(f"Dict: {len(ps.dictionary_chars())} chars -> {out}/saudi_plate_dict.txt")

    if montage and montage_imgs:
        _save_montage(montage_imgs, os.path.join(out, "_montage.jpg"))
        print(f"Montage: {out}/_montage.jpg")


def _save_montage(items, path):
    cols, cell_w, cell_h = 3, BASE_W, BASE_H + 40
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype(LATIN_FONTS[1], 22)
    except Exception:
        f = ImageFont.load_default()
    for idx, (im, la, ar) in enumerate(items):
        r, c = divmod(idx, cols)
        x, y = c * cell_w, r * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 6, y + BASE_H + 6), f"EN: {la}", fill=(0, 0, 0), font=f)
    sheet.save(path, quality=90)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--out", type=str, default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--montage", action="store_true",
                    help="save a _montage.jpg preview of the first 12 plates")
    args = ap.parse_args()
    build(args.count, args.out, args.val_frac, args.seed, args.montage)
