"""
Saudi license plate specification — shared by the synthetic generator and the
inference pipeline so that both agree on the character set and the top/bottom
split geometry.

Saudi civilian plates encode the SAME information twice:
    - top row    : Arabic letters + Arabic-Indic digits
    - bottom row : the Latin transliteration of those letters + Western digits
A blue vertical "KSA" strip sits on the right edge.

Only 17 letters are ever used. Each Arabic letter has a fixed Latin partner.
"""

# Latin letter  ->  Arabic letter   (the 17 letters allowed on Saudi plates)
LATIN_TO_ARABIC = {
    "A": "ا",
    "B": "ب",
    "J": "ح",
    "D": "د",
    "R": "ر",
    "S": "س",
    "X": "ص",
    "T": "ط",
    "E": "ع",
    "G": "ق",
    "K": "ك",
    "L": "ل",
    "Z": "م",
    "N": "ن",
    "H": "ه",
    "U": "و",
    "V": "ى",
}
ARABIC_TO_LATIN = {ar: la for la, ar in LATIN_TO_ARABIC.items()}

LATIN_LETTERS = list(LATIN_TO_ARABIC.keys())       # 17 letters
ARABIC_LETTERS = list(LATIN_TO_ARABIC.values())

LATIN_DIGITS = list("0123456789")
ARABIC_DIGITS = list("٠١٢٣٤٥٦٧٨٩")                  # U+0660..U+0669
LATIN_TO_ARABIC_DIGIT = dict(zip(LATIN_DIGITS, ARABIC_DIGITS))

# Plate composition rules
MIN_DIGITS, MAX_DIGITS = 1, 4
MIN_LETTERS, MAX_LETTERS = 1, 3

# Fraction of plate width occupied by the KSA strip on the right.
# The inference split removes this before reading.
KSA_STRIP_FRAC = 0.12


def to_arabic_digits(s: str) -> str:
    """Convert a string of Western digits to Arabic-Indic digits."""
    return "".join(LATIN_TO_ARABIC_DIGIT.get(c, c) for c in s)


def latin_label(digits: str, letters: str) -> str:
    """Bottom-row (English) label, e.g. '3417 UAJ'."""
    return f"{digits} {letters}"


def arabic_label(digits: str, letters: str) -> str:
    """
    Top-row (Arabic) label rendered in the SAME left-to-right visual order as
    the bottom row, so CTC sees image order == label order.
    """
    ar_digits = to_arabic_digits(digits)
    ar_letters = "".join(LATIN_TO_ARABIC[c] for c in letters)
    return f"{ar_digits} {ar_letters}"


def dictionary_chars():
    """
    Every character the recognizer is allowed to output, one unified dict so a
    single model can read both halves.

    NOTE: space is intentionally NOT included here. PaddleOCR appends the space
    character itself when `use_space_char: true`; adding it to the dict file too
    would duplicate it and shift class indices. The CTC blank is also implicit.
    """
    chars = []
    chars += LATIN_DIGITS
    chars += LATIN_LETTERS
    chars += ARABIC_DIGITS
    chars += ARABIC_LETTERS
    # de-dupe while preserving order
    seen, out = set(), []
    for c in chars:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def split_saudi_plate(plate_crop):
    """
    Split a plate crop into (top_half=Arabic, bottom_half=English).

    Used BOTH when generating training crops and at inference time, so the
    model trains on exactly the kind of crop it will later be asked to read.

    `plate_crop` is an HxW or HxWxC numpy array (OpenCV BGR or grayscale).
    Returns (top_half, bottom_half) or (None, None) if the crop is empty.
    """
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
        return None, None
    h, w = plate_crop.shape[:2]
    # drop the KSA strip on the right
    plate_no_ksa = plate_crop[:, : int(w * (1 - KSA_STRIP_FRAC))]
    mid = h // 2
    top_half = plate_no_ksa[:mid, :]
    bottom_half = plate_no_ksa[mid:, :]
    return top_half, bottom_half
