# Saudi Plate OCR — Jetson Deployment Guide

Everything needed to run the trained Saudi license-plate recognizer on an
NVIDIA Jetson, in production, with multi-frame voting.

> TL;DR: copy 4 files onto the Jetson, `pip install onnxruntime-gpu opencv-python numpy`,
> then call `vote_plate(reader, frames_of_one_car)`. Read the **Critical gotchas**
> section — one of them (normalization) will silently break letter reads if ignored.

---

## 1. What this is

A PaddleOCR recognizer (SVTR_LCNet) fine-tuned to read Saudi civilian plates.
A Saudi plate shows the **same** info twice: Arabic row on top, Latin row on the
bottom. The reader splits the plate into those two halves and reads each.

- Input : a **tight, whole-plate crop** (BGR image) from your plate detector.
- Output: the plate reading (Latin = the plate ID) + a confidence + a
  `is_confident` flag for the cashier-fallback logic.

**Accuracy (measured on held-out real plates):**
- Per single crop: ~70% exact (digits ~88%, letters ~76%).
- Per **car** with multi-frame voting across ~15–30 frames: **~95%+** (this is
  the number that matters in the drive-thru).

---

## 2. Files you need on the Jetson

Copy these into one folder (e.g. `/opt/plate_ocr/`):

| File | What it is | Where it is now |
|---|---|---|
| `saudi_rec.onnx` | the trained model | `ocr_training/saudi_rec_onnx/saudi_rec.onnx` |
| `saudi_plate_dict.txt` | the character list | `ocr_training/saudi_rec_onnx/saudi_plate_dict.txt` |
| `saudi_rec_infer.py` | the reader + voting code | `ocr_training/saudi_rec_infer.py` |
| `plate_spec.py` | plate rules + the top/bottom split | `ocr_training/plate_spec.py` |

`saudi_rec_infer.py` imports `plate_spec.py`, so keep them in the same folder.

---

## 3. Dependencies

On the Jetson (JetPack already provides CUDA/cuDNN/TensorRT):

```bash
pip install numpy opencv-python
# GPU onnxruntime for Jetson — use NVIDIA's prebuilt wheel for your JetPack.
# See https://elinux.org/Jetson_Zoo#ONNX_Runtime for the matching wheel, e.g.:
#   pip install onnxruntime_gpu-<ver>-cp<py>-cp<py>-linux_aarch64.whl
# If you just want it working first, plain CPU also runs (slower):
#   pip install onnxruntime
```

The model is tiny (~7 MB) and fast; CPU works for testing, GPU/TensorRT for
production throughput.

---

## 4. How the pipeline fits together

```
camera frame
   │
   ▼
[ vehicle + plate detector ]        ← RF-DETR (checkpoint_best_regular.pth), separate model
   │  tight plate crop (BGR)
   ▼
[ SaudiPlateReader.read(crop) ]     ← split into Arabic/Latin halves, run ONNX on each
   │  per-frame reading
   ▼
[ vote_plate(reader, [crops]) ]     ← majority vote across many frames of the SAME car
   │
   ▼
plate ID + confidence + is_confident
```

Collect **many crops of the same car** (one per frame while it's in view), then
vote once. Voting is what turns ~70%-per-frame into ~95%+-per-car.

---

## 5. Usage

### One frame (debugging)
```python
from saudi_rec_infer import SaudiPlateReader

reader = SaudiPlateReader(
    "saudi_rec.onnx", "saudi_plate_dict.txt",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],  # CPU-only: ["CPUExecutionProvider"]
)

r = reader.read(plate_crop_bgr)      # numpy BGR whole-plate crop
# -> {"english": "3417 UAJ", "arabic": "٣٤١٧ واح",
#     "english_conf": 0.95, "arabic_conf": 0.88}
```

### Production (multi-frame voting)
```python
from saudi_rec_infer import SaudiPlateReader, vote_plate

reader = SaudiPlateReader("saudi_rec.onnx", "saudi_plate_dict.txt")

# frames = list of whole-plate BGR crops of ONE car (from your detector, one per frame)
result = vote_plate(reader, frames)
# result = {
#   "plate": "3417 UAJ",        # voted Latin reading (the plate ID)
#   "arabic": "٣٤١٧ واح",       # derived from the voted Latin, kept consistent
#   "confidence": 0.93,          # 0..1, how strongly frames agreed
#   "is_confident": True,        # True -> use it; False -> fall back
#   "frames_used": 22, "total_frames": 25, "valid_format": True,
# }

if result["is_confident"]:
    plate = result["plate"]          # use automatically
else:
    plate = None                     # low confidence -> cashier confirms / mark "unverified"
```

### Tuning knobs (`vote_plate` args)
- `min_frame_conf=0.40` — ignore per-frame reads below this confidence.
- `min_agreement=0.50` — how much the frames must agree to be "confident".
- `min_frames=2` — need at least this many usable frames.

Raise `min_agreement` for fewer wrong-but-confident reads (more "unverified"
fall-throughs); lower it to auto-accept more. Start at defaults, adjust from
real logs.

---

## 6. Critical gotchas (read these)

1. **Normalization = `x/255` (pixels in [0,1]), NOT `(x/255-0.5)/0.5`.**
   This exported ONNX bakes the mean/scale step into its own graph. Feeding the
   usual PaddleOCR normalization double-normalizes and **silently destroys the
   letter reads** (digits still look ~ok, so it's easy to miss). This is already
   handled correctly in `saudi_rec_infer.py` `_preprocess()` — do **not** change
   it back. If you ever re-export the model, re-check this.

2. **`split_saudi_plate` assumes a stacked plate with the KSA strip on the right.**
   It drops the right 12% (the blue KSA strip) and splits top/bottom at the
   mid-height. It works for normal stacked plates. It is NOT reliable for the
   wide "long" plates that have the KSA emblem in the **center** (digits | KSA |
   letters) — those need a different crop. Feed the reader tight, stacked-style
   plate crops from the detector.

3. **Feed BGR, not RGB.** `cv2.imread` gives BGR, which is what the reader
   expects. If your frames come from another source, convert to BGR first.

4. **Detector crop quality matters.** The tighter and more consistent the plate
   crop, the better. Loose crops with lots of background or angle hurt reads.

---

## 7. (Optional) TensorRT for max speed

onnxruntime with the CUDA provider is usually plenty. For maximum throughput you
can build a TensorRT engine:

```bash
trtexec --onnx=saudi_rec.onnx --saveEngine=saudi_rec.engine --fp16
```

Then swap the onnxruntime session for a TRT runtime, keeping the **exact same
pre/post-processing** (the `x/255` normalization and the CTC decode). Only do
this if you need the extra speed.

---

## 8. Expected accuracy & the confidence gate

- Don't judge it on single frames (~70%). Judge per **car** after voting.
- Design the product so a **low-confidence read never silently becomes a wrong
  charge/match**: high confidence → auto-use; low confidence → cashier confirms
  or mark "unverified". A system that's right most of the time and says "not
  sure" otherwise is far safer than one that's confidently wrong.
- Suggested target for the plate feature to feel reliable: **~98%+ per car**,
  reached via voting + the confidence gate.

---

## 9. Updating / retraining the model later

Training happens on Kaggle, not the Jetson. Two notebooks in this repo:

- `saudi_ocr_real_only_kaggle.ipynb` — clean, real-data-only, auto-saves +
  auto-exports ONNX. This produced the current model (~70%).
- `saudi_ocr_train_kaggle.ipynb` — full pipeline incl. synthetic; the
  synthetic-assisted path reads letters better (~76%). Use this if you want to
  push accuracy up.

To improve accuracy: (a) retrain **with synthetic** to lift letters, (b) label
more of your **own** drive-thru footage (real plates from your cameras/angles),
(c) recover the ~56 skipped center-KSA "long" plates.

After retraining, the notebook exports a new `saudi_rec.onnx` — copy it to the
Jetson (same 4 files). **Re-verify the `x/255` normalization still applies** to
any newly exported ONNX.

---

## 10. Quick sanity check on the Jetson

After copying the files, confirm the model loads and reads:

```python
import cv2
from saudi_rec_infer import SaudiPlateReader
reader = SaudiPlateReader("saudi_rec.onnx", "saudi_plate_dict.txt",
                          providers=["CPUExecutionProvider"])
print(reader.read(cv2.imread("some_plate_crop.jpg")))
```

You should get sensible `english`/`arabic` strings with confidences ~0.8–0.99
on a clear plate. If letters come back empty or garbled, you almost certainly
hit gotcha #1 (normalization).

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Digits read, **letters empty/garbled** | double normalization | must be `x/255` (gotcha #1) |
| Everything garbled | RGB fed instead of BGR, or wrong crop | feed BGR whole-plate crop |
| Letters from wrong row / mixed | center-KSA long plate | crop differently (gotcha #2) |
| Slow | running on CPU | install `onnxruntime-gpu` / TensorRT |
| `onnxruntime` no CUDA provider | CPU wheel installed | install Jetson GPU wheel (Jetson Zoo) |
| Low per-car accuracy | too few frames / loose crops | vote over more frames, tighten detector crop |
