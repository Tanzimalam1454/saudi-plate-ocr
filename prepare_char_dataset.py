"""
Convert the Kaggle `saudi-license-plate-characters` dataset (riotulab) into
labeled half-crops for the recognizer.

The dataset gives, for every full two-row plate photo, per-character bounding
boxes labeled with the LATIN reading only (Western digits + the 17 Latin
letters). Each photo is a complete Saudi plate: Arabic row on top, Latin row on
the bottom. So for every photo we can emit TWO real labeled half-crops:

    bottom (Latin)  : cropped straight from the labeled character row,
                      label read directly from the gold boxes (e.g. "7147 VXA").
    top   (Arabic)  : the mirror band directly above the Latin row,
                      label DERIVED from the Latin reading via plate_spec
                      (same trick as build_real_labels.py), so the two halves
                      stay perfectly consistent with how the model was trained.

We crop using the gold character boxes (not a blind geometric split) so it works
regardless of where the KSA strip sits. "Long" plates whose Latin row is split
by a centre KSA emblem, and plates that don't parse to a valid Saudi plate, are
SKIPPED and reported rather than mislabeled.

Usage:
    python prepare_char_dataset.py \
        --src   "License-Characters-by-2-27classes" \
        --out   dataset \
        --val-frac 0.2

Outputs (under --out):
    chars/<stem>__AR.jpg , chars/<stem>__EN.jpg          half-crops
    char_train_label.txt , char_val_label.txt            "chars/..\\tLABEL" lines
    char_skipped.csv                                      what was dropped and why
"""

import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from plate_spec import (ARABIC_TO_LATIN, LATIN_LETTERS, arabic_label,
                        latin_label)

LETTERS = set(LATIN_LETTERS)
DIGITS = set("0123456789")


def find_xmls(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.xml"), recursive=True))


def image_for(xml_path):
    for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"):
        p = xml_path[:-4] + ext
        if os.path.exists(p):
            return p
    return None


def parse_chars(xml_path):
    """Return (W, H, [ (name, xmin, ymin, xmax, ymax), ... ])."""
    r = ET.parse(xml_path).getroot()
    W = float(r.findtext("size/width") or 0)
    H = float(r.findtext("size/height") or 0)
    chars = []
    for obj in r.iter("object"):
        nm = (obj.findtext("name") or "").strip().upper()
        b = obj.find("bndbox")
        if b is None:
            continue
        x0 = float(b.findtext("xmin")); y0 = float(b.findtext("ymin"))
        x1 = float(b.findtext("xmax")); y1 = float(b.findtext("ymax"))
        chars.append((nm, x0, y0, x1, y1))
    return W, H, chars


def read_plate(chars):
    """From gold character boxes, recover (digits, letters) in left-to-right
    reading order. Returns (digits, letters) or None if it isn't a valid plate.

    Accepts a few Arabic class names too, mapping them back to Latin, in case a
    handful of boxes were labeled in Arabic."""
    norm = []
    for nm, x0, y0, x1, y1 in chars:
        c = nm
        if c in ARABIC_TO_LATIN:           # tolerate stray Arabic labels
            c = ARABIC_TO_LATIN[c]
        cx = (x0 + x1) / 2
        norm.append((cx, c, x0, y0, x1, y1))
    norm.sort(key=lambda t: t[0])          # left -> right
    digits = "".join(c for _, c, *_ in norm if c in DIGITS)
    letters = "".join(c for _, c, *_ in norm if c in LETTERS)
    # anything that's neither a known digit nor a known letter -> invalid
    unknown = [c for _, c, *_ in norm if c not in DIGITS and c not in LETTERS]
    if unknown:
        return None, "unknown char(s): " + ",".join(sorted(set(unknown)))
    if not (1 <= len(digits) <= 4):
        return None, f"digit count {len(digits)} (need 1-4)"
    if not (1 <= len(letters) <= 3):
        return None, f"letter count {len(letters)} (need 1-3)"
    return (digits, letters), None


def latin_band(chars):
    """Bounding band of the labeled (Latin) characters: (x0,y0,x1,y1)."""
    x0 = min(c[1] for c in chars); y0 = min(c[2] for c in chars)
    x1 = max(c[3] for c in chars); y1 = max(c[4] for c in chars)
    return x0, y0, x1, y1


def has_center_ksa_gap(chars):
    """Detect 'long' plates whose Latin row is split by a centre KSA emblem:
    a horizontal gap between consecutive characters far larger than a normal
    inter-character / digit-letter space."""
    xs = sorted(((c[1], c[3]) for c in chars))      # (xmin, xmax) left->right
    widths = [b - a for a, b in xs]
    med_w = sorted(widths)[len(widths) // 2] if widths else 0
    for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
        gap = b0 - a1
        if med_w and gap > 2.5 * med_w:             # huge gap => emblem in middle
            return True
    return False


def crop(img, box):
    W, H = img.size
    x0, y0, x1, y1 = box
    x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
    x1 = min(W, int(round(x1))); y1 = min(H, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((x0, y0, x1, y1))


def make_halfcrops(img, chars, pad_x=0.04, pad_y=0.06):
    """Return (top_arabic_pil, bottom_latin_pil).

    The character boxes in this dataset bound the full-height *column* of each
    slot (Arabic glyph stacked above its Latin partner), labeled with the Latin
    value. So the union of all boxes is the plate's character region (KSA strip
    excluded, since it carries no labeled char), and the Arabic/Latin row
    boundary is that region's vertical midline.

    pad_x : horizontal padding as a fraction of the region width.
    pad_y : vertical padding (top of Arabic / bottom of Latin) as a fraction of
            region height.
    """
    x0, y0, x1, y1 = latin_band(chars)         # union of all char boxes
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None, None
    px, py = bw * pad_x, bh * pad_y
    lx0, lx1 = x0 - px, x1 + px
    mid = (y0 + y1) / 2.0                       # Arabic / Latin boundary
    top = crop(img, (lx0, y0 - py, lx1, mid))           # Arabic row
    bottom = crop(img, (lx0, mid, lx1, y1 + py))        # Latin row
    return top, bottom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="dataset root (contains train/ and test/)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N plates (for a quick test)")
    ap.add_argument("--sample-dir", default=None,
                    help="if set, also copy crops here for eyeballing")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, "chars")
    os.makedirs(out_dir, exist_ok=True)
    if args.sample_dir:
        os.makedirs(args.sample_dir, exist_ok=True)

    xmls = find_xmls(args.src)
    if args.limit:
        xmls = xmls[:args.limit]
    assert xmls, f"no .xml annotations found under {args.src}"

    lines, skipped = [], []
    for xp in xmls:
        stem = os.path.splitext(os.path.basename(xp))[0]
        img_path = image_for(xp)
        if not img_path:
            skipped.append((stem, "", "no image file"))
            continue
        W, H, chars = parse_chars(xp)
        if not chars:
            skipped.append((stem, "", "no boxes"))
            continue
        parsed, why = read_plate(chars)
        if parsed is None:
            skipped.append((stem, "", why))
            continue
        if has_center_ksa_gap(chars):
            skipped.append((stem, "".join(parsed[0]) + " " + parsed[1],
                            "center-KSA long plate (skipped for safety)"))
            continue
        digits, letters = parsed
        img = Image.open(img_path).convert("RGB")
        top, bottom = make_halfcrops(img, chars)
        if top is None or bottom is None:
            skipped.append((stem, latin_label(digits, letters), "crop failed"))
            continue

        en_name = f"chars/{stem}__EN.jpg"
        ar_name = f"chars/{stem}__AR.jpg"
        bottom.save(os.path.join(args.out, en_name), quality=95)
        top.save(os.path.join(args.out, ar_name), quality=95)
        lines.append((en_name, latin_label(digits, letters)))
        lines.append((ar_name, arabic_label(digits, letters)))

        if args.sample_dir:
            tag = f"{digits}_{letters}"
            bottom.save(os.path.join(args.sample_dir, f"{stem}__{tag}__EN.jpg"),
                        quality=95)
            top.save(os.path.join(args.sample_dir, f"{stem}__{tag}__AR.jpg"),
                     quality=95)

    # split by PLATE (keep a plate's two halves together)
    stems = sorted({ln[0].split("__")[0] for ln in lines})
    stems.sort(key=lambda s: (sum(ord(c) for c in s) % 997, s))   # stable shuffle
    n_val = max(1, int(round(len(stems) * args.val_frac))) if stems else 0
    val_stems = set(stems[:n_val])
    train = [ln for ln in lines if ln[0].split("__")[0] not in val_stems]
    val = [ln for ln in lines if ln[0].split("__")[0] in val_stems]

    def _write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for name, label in rows:
                f.write(f"{name}\t{label}\n")

    _write(os.path.join(args.out, "char_train_label.txt"), train)
    _write(os.path.join(args.out, "char_val_label.txt"), val)

    with open(os.path.join(args.out, "char_skipped.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["stem", "reading", "reason"])
        w.writerows(skipped)

    print(f"plates OK     : {len(stems)}  ->  {len(lines)} half-crops")
    print(f"  train/val   : {len(train)} / {len(val)} half-crops")
    print(f"  skipped     : {len(skipped)}  (see {args.out}/char_skipped.csv)")
    reasons = {}
    for _, _, why in skipped:
        key = why.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {v:4d}  {k}")


if __name__ == "__main__":
    main()
