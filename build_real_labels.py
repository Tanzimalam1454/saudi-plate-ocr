"""
Stage 2 of the REAL-data pipeline: take the human-corrected English readings
and turn each real plate crop into TWO labeled half-crops (Arabic top + Latin
bottom), exactly like the synthetic generator does, then split them into a
train/val label file PaddleOCR can read.

The human only ever supplies the ENGLISH string (e.g. "3495 NQR"). The Arabic
label is derived automatically from plate_spec, so both halves stay perfectly
consistent with how the model was trained.

Run AFTER you have corrected.json (downloaded from label_me.html).

Usage:
    python build_real_labels.py \
        --crops     real_data/crops \
        --corrected corrected.json \
        --out       dataset \
        --val-frac  0.2

Outputs (under --out):
    real/<stem>__AR.jpg , real/<stem>__EN.jpg     half-crops
    real_train_label.txt , real_val_label.txt     "real/..__EN.jpg\\tLABEL" lines
"""

import argparse
import json
import os
import re

import numpy as np
from PIL import Image

from plate_spec import (LATIN_LETTERS, arabic_label, latin_label,
                        split_saudi_plate)

LETTERS = set(LATIN_LETTERS)


def parse_english(text):
    """'3495 NQR' -> ('3495', 'NQR'). Returns None if it isn't a valid plate.

    Accepts any spacing/case; requires 1-4 digits then 1-3 allowed letters."""
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.strip().upper())
    # pull the digit run and the letter run regardless of order/extra spaces
    digits = "".join(re.findall(r"\d", t))
    letters = "".join(ch for ch in t if ch.isalpha())
    if not (1 <= len(digits) <= 4):
        return None
    if not (1 <= len(letters) <= 3) or any(ch not in LETTERS for ch in letters):
        return None
    return digits, letters


def deterministic_order(names):
    """Stable shuffle without Random (Kaggle reproducibility) -- order by a
    simple hash of the filename so the same plates always land in val."""
    return sorted(names, key=lambda s: (sum(ord(c) for c in s) % 997, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="real_data/crops")
    ap.add_argument("--corrected", default="corrected.json")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    with open(args.corrected, "r", encoding="utf-8") as f:
        corrected = json.load(f)

    real_dir = os.path.join(args.out, "real")
    os.makedirs(real_dir, exist_ok=True)

    lines, skipped = [], []
    for fname, text in corrected.items():
        parsed = parse_english(text)
        if parsed is None:
            if text.strip():
                skipped.append((fname, text, "invalid format"))
            continue
        digits, letters = parsed
        crop_path = os.path.join(args.crops, fname)
        if not os.path.exists(crop_path):
            skipped.append((fname, text, "crop image missing"))
            continue
        img = np.array(Image.open(crop_path).convert("RGB"))   # RGB
        top, bottom = split_saudi_plate(img)
        if top is None or top.size == 0 or bottom.size == 0:
            skipped.append((fname, text, "crop too small to split"))
            continue
        stem = os.path.splitext(fname)[0]
        en_name = f"real/{stem}__EN.jpg"
        ar_name = f"real/{stem}__AR.jpg"
        Image.fromarray(bottom).save(os.path.join(args.out, en_name), quality=95)
        Image.fromarray(top).save(os.path.join(args.out, ar_name), quality=95)
        lines.append((en_name, latin_label(digits, letters)))
        lines.append((ar_name, arabic_label(digits, letters)))

    # split by PLATE (keep a plate's two halves together) so val is honest
    stems = deterministic_order(sorted({ln[0].split("__")[0] for ln in lines}))
    n_val = max(1, int(round(len(stems) * args.val_frac))) if stems else 0
    val_stems = set(stems[:n_val])

    train_lines = [ln for ln in lines if ln[0].split("__")[0] not in val_stems]
    val_lines = [ln for ln in lines if ln[0].split("__")[0] in val_stems]

    def _write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for name, label in rows:
                f.write(f"{name}\t{label}\n")

    _write(os.path.join(args.out, "real_train_label.txt"), train_lines)
    _write(os.path.join(args.out, "real_val_label.txt"), val_lines)

    print(f"plates labeled : {len(stems)}")
    print(f"  -> {len(train_lines)} train half-crops, {len(val_lines)} val half-crops")
    print(f"  real_train_label.txt / real_val_label.txt written to {args.out}/")
    if skipped:
        print(f"\nskipped {len(skipped)} (fix the English and re-download if you want them):")
        for fn, tx, why in skipped:
            print(f"  {fn!r}: {tx!r}  ({why})")


if __name__ == "__main__":
    main()
