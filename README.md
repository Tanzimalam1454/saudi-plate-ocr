# Saudi Plate OCR — Training Pipeline

Trains a **custom PaddleOCR recognizer** specialized on Saudi license plates,
replacing the generic EasyOCR. One model reads **both** halves of a plate:
the Arabic top row and the Latin bottom row.

Why custom: EasyOCR is a generalist (~70–85% on plates). A model that only ever
sees Saudi plates gets far more accurate and consistent — and consistency is
what the session-matching logic actually needs.

## Why synthetic data

The Roboflow `saudi-plate-tijbb` dataset is **detection-only** (boxes, no text),
so it can't teach a reader. Instead we *generate* thousands of valid plates with
correct fonts — every image comes with a perfect label for free. Plates are
highly standardized, so this works well. Mix in real labeled crops later for the
final accuracy boost.

## Files

| File | What it is |
|---|---|
| `plate_spec.py` | Char sets, Arabic⇄Latin mapping, `split_saudi_plate()` (shared by gen + inference) |
| `generate_plates.py` | Synthetic plate generator with realistic camera degradation |
| `saudi_rec_config.yml` | PaddleOCR PP-OCRv4 rec fine-tune config |
| `train_saudi_ocr_colab.ipynb` | End-to-end Colab notebook (install → generate → train → export) |
| `saudi_rec_infer.py` | Drop-in trained reader (ONNX) replacing EasyOCR |
| `requirements.txt` | Deps |

## Quick start

**1. Eyeball the generator locally** (macOS/Linux):
```bash
pip install -r requirements.txt
python generate_plates.py --count 12 --out _sample --montage
open _sample/_montage.jpg          # check realism
```

**2. Train on Colab (GPU):**
Open `train_saudi_ocr_colab.ipynb`, set runtime to GPU, run all cells. It uploads
the 3 source files, generates ~30k plates (60k crops), fine-tunes from the
pretrained Arabic model, and exports an inference model + `saudi_rec.onnx`.

**3. Deploy:**
```python
from saudi_rec_infer import SaudiPlateReader
reader = SaudiPlateReader("saudi_rec.onnx", "saudi_plate_dict.txt")
print(reader.read(plate_crop_bgr))     # {"english": "3417 UAJ", ...}
```
On the Jetson, convert to TensorRT: `trtexec --onnx=saudi_rec.onnx --saveEngine=saudi_rec.engine --fp16`.

## Layout assumption

The generator places **digits on the left, letters on the right** (label order
`"NNNN LLL"`), matching `PROJECT_STATUS.md`. If real footage shows the opposite
horizontal order, flip the order in `generate_plates.py::render_clean` and
regenerate — the model is internally consistent either way, but matching real
plates keeps the output human-readable.

## Improving real-world accuracy

Synthetic gets a strong baseline. The biggest gains come from **fine-tuning on
real crops**: once deployed, save plate crops, label a few hundred, add them to
`dataset/train_label.txt`, and re-run training. Synthetic + real together beats
either alone.
