"""
Drop-in Saudi plate reader backed by the TRAINED PaddleOCR recognizer (ONNX).

Replaces the EasyOCR path in the vision pipeline. Runs the same model on both
the Arabic (top) and English (bottom) halves produced by split_saudi_plate().

Jetson note: this uses onnxruntime so it runs anywhere. To use the TensorRT
engine instead, build it with
    trtexec --onnx=saudi_rec.onnx --saveEngine=saudi_rec.engine --fp16
and swap the session for a TRT runtime (same pre/post-processing).

Usage:
    reader = SaudiPlateReader("saudi_rec.onnx", "saudi_plate_dict.txt")
    result = reader.read(plate_crop_bgr)      # numpy BGR crop of the whole plate
    # -> {"english": "3417 UAJ", "arabic": "٣٤١٧ واح",
    #     "english_conf": 0.97, "arabic_conf": 0.83}
"""

import re
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

from plate_spec import LATIN_LETTERS, arabic_label, split_saudi_plate

_ALLOWED_LETTERS = set(LATIN_LETTERS)

REC_H, REC_W = 48, 320          # must match training image_shape [3, 48, 320]


def _load_charset(dict_path):
    """Rebuild PaddleOCR's class list: [blank] + dict chars + [space]."""
    with open(dict_path, "r", encoding="utf-8") as f:
        chars = [ln.strip("\n").strip("\r") for ln in f]
    chars = [c for c in chars if c != ""]
    return ["<blank>"] + chars + [" "]


def _preprocess(img):
    """Resize keeping aspect to height 48, pad to width 320, normalize."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    ratio = w / float(h)
    new_w = min(REC_W, max(1, int(REC_H * ratio)))
    resized = cv2.resize(img, (new_w, REC_H), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((REC_H, REC_W, 3), dtype=np.float32)
    canvas[:, :new_w, :] = resized
    # This exported ONNX bakes the mean/scale step into its own graph, so it
    # expects pixels in [0,1]. Feeding the usual (x/255-0.5)/0.5 double-
    # normalizes and silently destroys the letters — must be plain x/255.
    canvas = canvas / 255.0
    return canvas.transpose(2, 0, 1)            # CHW


class SaudiPlateReader:
    def __init__(self, onnx_path, dict_path, providers=None):
        self.charset = _load_charset(dict_path)
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def _decode(self, probs):
        """Greedy CTC decode of one (T, C) probability matrix."""
        idxs = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        out, kept_conf, prev = [], [], -1
        for t, idx in enumerate(idxs):
            if idx != prev and idx != 0:        # 0 == CTC blank
                out.append(self.charset[idx])
                kept_conf.append(confs[t])
            prev = idx
        text = "".join(out).strip()
        conf = float(np.mean(kept_conf)) if kept_conf else 0.0
        return text, conf

    def _run(self, half):
        x = _preprocess(half)
        if x is None:
            return "", 0.0
        out = self.sess.run(None, {self.input_name: x[None].astype(np.float32)})[0]
        return self._decode(out[0])             # (1, T, C) -> (T, C)

    def read(self, plate_crop_bgr):
        top, bottom = split_saudi_plate(plate_crop_bgr)
        ar_text, ar_conf = self._run(top)
        en_text, en_conf = self._run(bottom)
        return {
            "english": en_text,
            "arabic": ar_text,
            "english_conf": en_conf,
            "arabic_conf": ar_conf,
        }


def _split_en(text):
    """'3417 UAJ' (any spacing/case) -> ('3417', 'UAJ')."""
    t = re.sub(r"\s+", "", text.upper())
    digits = "".join(c for c in t if c.isdigit())
    letters = "".join(c for c in t if c.isalpha())
    return digits, letters


def vote_plate(reader, plate_crops, min_frame_conf=0.40, min_agreement=0.5,
               min_frames=2):
    """
    PRODUCTION reader: confidence-weighted, per-character voting across many
    frames of the SAME car. This is what turns a ~70%-per-frame model into
    ~95%+ per car, because a wrong character in one frame is out-voted by the
    correct character in the others.

    plate_crops : list of whole-plate BGR crops of one car (from the detector).

    Returns a dict:
        plate         : "3417 UAJ"  (voted English reading; the plate ID)
        arabic        : "٣٤١٧ واح"  (derived from the voted English)
        confidence    : 0..1  how strongly the frames agreed
        is_confident  : True  -> use it automatically
                        False -> fall back (cashier confirms / mark unverified)
        frames_used   : how many frames actually contributed
        valid_format  : letters are all in the Saudi 17-letter set, valid counts
    """
    reads = []
    for crop in plate_crops:
        r = reader.read(crop)
        if r["english"] and r["english_conf"] >= min_frame_conf:
            d, l = _split_en(r["english"])
            if d and l:
                reads.append((d, l, r["english_conf"]))

    blank = {"plate": None, "arabic": None, "confidence": 0.0,
             "is_confident": False, "frames_used": 0,
             "total_frames": len(plate_crops), "valid_format": False}
    if not reads:
        return blank

    # 1) pick the most likely plate SHAPE (digit-count, letter-count),
    #    weighted by confidence -- guards against a frame that dropped a char.
    shape_w = defaultdict(float)
    for d, l, c in reads:
        shape_w[(len(d), len(l))] += c
    nd, nl = max(shape_w, key=shape_w.get)
    cand = [(d, l, c) for d, l, c in reads if len(d) == nd and len(l) == nl]

    # 2) per-position, confidence-weighted majority vote
    def vote(items, n):
        chars, agrees = [], []
        for i in range(n):
            w = defaultdict(float)
            for s, c in items:
                w[s[i]] += c
            best = max(w, key=w.get)
            chars.append(best)
            agrees.append(w[best] / sum(w.values()))
        return "".join(chars), (sum(agrees) / len(agrees) if agrees else 0.0)

    digits, da = vote([(d, c) for d, l, c in cand], nd)
    letters, la = vote([(l, c) for d, l, c in cand], nl)

    valid = (1 <= nd <= 4 and 1 <= nl <= 3
             and all(ch in _ALLOWED_LETTERS for ch in letters))
    agreement = (da + la) / 2.0
    is_conf = (valid and agreement >= min_agreement
               and len(cand) >= max(min_frames, int(0.3 * len(plate_crops))))

    return {
        "plate": f"{digits} {letters}",
        "arabic": arabic_label(digits, letters) if valid else None,
        "confidence": round(agreement, 3),
        "is_confident": bool(is_conf),
        "frames_used": len(cand),
        "total_frames": len(plate_crops),
        "valid_format": valid,
    }


def read_plate_multi_frame(reader, plate_crops):
    """Back-compat thin wrapper: returns just the voted plate string (no space),
    or None. Prefer vote_plate() for the confidence + fallback flag."""
    r = vote_plate(reader, plate_crops)
    return r["plate"].replace(" ", "") if r["plate"] else None
