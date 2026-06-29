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

import cv2
import numpy as np
import onnxruntime as ort

from plate_spec import split_saudi_plate

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
    # PaddleOCR rec normalization: (x/255 - 0.5) / 0.5, channels-first
    canvas = (canvas / 255.0 - 0.5) / 0.5
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


def read_plate_multi_frame(reader, plate_crops):
    """
    Multi-frame voting (the production approach in PROJECT_STATUS.md): read N
    crops of the same car, return the most common normalized English string.
    """
    from collections import Counter
    votes = []
    for crop in plate_crops:
        r = reader.read(crop)
        if r["english"] and r["english_conf"] > 0.4:
            votes.append(r["english"].replace(" ", "").upper())
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]
