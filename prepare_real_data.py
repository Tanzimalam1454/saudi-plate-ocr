"""
Stage 1 of the REAL-data pipeline: turn a Roboflow detection export into
cropped plate images, pseudo-label each with the trained model, and emit a
single self-contained HTML page for the human to correct the guesses.

The Roboflow project labels REGIONS (license-plate, arabic-number, english-text,
...) with boxes only -- there is no text in the export. We use the
`license-plate` box (or, as a fallback, the union of the sub-region boxes) to
crop a tight plate from every photo. The trained recognizer then reads each
crop so the human only has to FIX wrong guesses instead of typing from scratch.

Run this on Kaggle AFTER:
  - the model has been trained  (output/saudi_rec/best_accuracy exists)
  - it has been exported to ONNX (inference/saudi_rec.onnx exists; notebook step 7)

Usage:
    python prepare_real_data.py \
        --coco   <roboflow_export_dir> \
        --onnx   inference/saudi_rec.onnx \
        --dict   dataset/saudi_plate_dict.txt \
        --out    real_data

Outputs (under --out):
    crops/<stem>.jpg        one tight plate crop per source photo
    pseudo_labels.csv       filename, guess, confidence
    label_me.html           open in a browser, correct, click "Download corrections"
"""

import argparse
import base64
import csv
import glob
import json
import os

import numpy as np
from PIL import Image

from plate_spec import LATIN_LETTERS, split_saudi_plate

# Roboflow class names we treat as "the whole plate" / "part of the plate".
PLATE_NAMES = {"license-plate", "license_plate", "licenseplate", "plate"}
REGION_NAMES = {
    "arabic-number", "arabic-text", "english-number", "english-text",
    "arabic_number", "arabic_text", "english_number", "english_text",
}


def _norm(name):
    return str(name).strip().lower().replace(" ", "-").replace("_", "-")


def find_coco_jsons(root):
    """Roboflow COCO exports drop one _annotations.coco.json per split folder."""
    hits = glob.glob(os.path.join(root, "**", "_annotations.coco.json"),
                     recursive=True)
    if not hits:
        # some exports name it differently
        hits = glob.glob(os.path.join(root, "**", "*.coco.json"), recursive=True)
    return sorted(hits)


def load_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_dir = os.path.dirname(json_path)
    cats = {c["id"]: _norm(c["name"]) for c in data.get("categories", [])}
    images = {im["id"]: im for im in data.get("images", [])}
    by_image = {}
    for ann in data.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)
    return img_dir, cats, images, by_image


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def pick_plate_box(anns, cats):
    """Return [x0,y0,x1,y1] for the plate: prefer a license-plate box,
    else the bounding union of the sub-region boxes."""
    plate = [_xywh_to_xyxy(a["bbox"]) for a in anns
             if _norm(cats.get(a["category_id"], "")) in PLATE_NAMES]
    if plate:
        # largest license-plate box if several
        return max(plate, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    regions = [_xywh_to_xyxy(a["bbox"]) for a in anns
               if _norm(cats.get(a["category_id"], "")) in REGION_NAMES]
    if regions:
        xs0 = min(b[0] for b in regions); ys0 = min(b[1] for b in regions)
        xs1 = max(b[2] for b in regions); ys1 = max(b[3] for b in regions)
        return [xs0, ys0, xs1, ys1]
    return None


def crop_with_margin(img, box, margin):
    W, H = img.size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - bw * margin)); y0 = max(0, int(y0 - bh * margin))
    x1 = min(W, int(x1 + bw * margin)); y1 = min(H, int(y1 + bh * margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((x0, y0, x1, y1))


def build_reader(onnx_path, dict_path):
    """Import lazily so the crop-only path doesn't need onnxruntime."""
    from saudi_rec_infer import SaudiPlateReader
    return SaudiPlateReader(onnx_path, dict_path,
                            providers=["CPUExecutionProvider"])


def pseudo_label(reader, pil_crop):
    """reader expects a BGR numpy whole-plate crop; returns english guess+conf."""
    bgr = np.array(pil_crop.convert("RGB"))[:, :, ::-1]
    r = reader.read(bgr)
    return r["english"], float(r["english_conf"])


def write_html(rows, out_html):
    """rows: list of dicts {file, b64, guess, conf}. Sorted worst-first."""
    rows = sorted(rows, key=lambda r: r["conf"])
    letters = " ".join(LATIN_LETTERS)
    items = []
    for r in rows:
        items.append(
            '<div class="row">'
            f'<img src="data:image/jpeg;base64,{r["b64"]}">'
            '<div class="meta">'
            f'<input data-file="{_html_attr(r["file"])}" '
            f'value="{_html_attr(r["guess"])}" autocomplete="off" '
            'autocapitalize="characters" spellcheck="false">'
            f'<span class="conf">conf {r["conf"]:.2f}</span>'
            '</div></div>'
        )
    html = _HTML_TEMPLATE.replace("__LETTERS__", letters)
    html = html.replace("__COUNT__", str(len(rows)))
    html = html.replace("__ITEMS__", "\n".join(items))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def _html_attr(s):
    return (str(s).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


_HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>Correct plate readings</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1a1a2e}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
        padding:12px 20px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{font-size:18px;margin:0 0 4px}
 .hint{font-size:13px;color:#555}
 .hint code{background:#eee;padding:1px 5px;border-radius:4px}
 #save{background:#6c2bd9;color:#fff;border:0;padding:10px 18px;border-radius:8px;
       font-size:15px;cursor:pointer;margin-top:8px}
 #save:hover{background:#5a23b6}
 .wrap{max-width:760px;margin:18px auto;padding:0 16px}
 .row{display:flex;align-items:center;gap:14px;background:#fff;border:1px solid #e3e3e8;
      border-radius:10px;padding:10px 14px;margin-bottom:10px}
 .row img{height:64px;border:1px solid #ccc;border-radius:6px;background:#fafafa;
          image-rendering:pixelated}
 .meta{display:flex;flex-direction:column;gap:4px;flex:1}
 .meta input{font-size:20px;letter-spacing:1px;padding:8px 10px;border:1px solid #bbb;
             border-radius:6px;text-transform:uppercase;font-family:monospace}
 .conf{font-size:12px;color:#888}
 footer{position:sticky;bottom:0;background:#fff;border-top:1px solid #ddd;
         padding:12px 20px;text-align:center}
</style></head><body>
<header>
 <h1>Correct the plate readings (__COUNT__ plates)</h1>
 <div class="hint">Each box is the model's guess. Fix the wrong ones. Format:
   <code>1234 ABC</code> (digits, one space, letters). Allowed letters:
   <code>__LETTERS__</code>. Leave a box <b>empty</b> to skip a plate you can't read.</div>
 <button id="save">Download corrections</button>
</header>
<div class="wrap">
__ITEMS__
</div>
<footer><button id="save2">Download corrections</button></footer>
<script>
 function download(){
   const out={};
   document.querySelectorAll('input[data-file]').forEach(function(i){
     out[i.getAttribute('data-file')] = i.value.trim().toUpperCase();
   });
   const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
   const a=document.createElement('a');
   a.href=URL.createObjectURL(blob); a.download='corrected.json';
   document.body.appendChild(a); a.click(); a.remove();
 }
 document.getElementById('save').onclick=download;
 document.getElementById('save2').onclick=download;
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Roboflow COCO export dir")
    ap.add_argument("--onnx", default=None, help="trained ONNX model (optional)")
    ap.add_argument("--dict", default="dataset/saudi_plate_dict.txt")
    ap.add_argument("--out", default="real_data")
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    jsons = find_coco_jsons(args.coco)
    assert jsons, f"no _annotations.coco.json found under {args.coco}"
    print("annotation files:", *jsons, sep="\n  ")

    reader = build_reader(args.onnx, args.dict) if args.onnx else None
    if reader is None:
        print("NOTE: --onnx not given -> guesses left blank (crop-only mode)")

    rows, n_seen, n_cropped = [], 0, 0
    for jp in jsons:
        img_dir, cats, images, by_image = load_coco(jp)
        for img_id, im in images.items():
            n_seen += 1
            anns = by_image.get(img_id, [])
            box = pick_plate_box(anns, cats)
            if box is None:
                print("  skip (no plate box):", im["file_name"])
                continue
            path = os.path.join(img_dir, im["file_name"])
            if not os.path.exists(path):
                print("  skip (missing image):", path)
                continue
            crop = crop_with_margin(Image.open(path).convert("RGB"), box,
                                    args.margin)
            if crop is None:
                continue
            stem = f"{img_id:04d}_{os.path.splitext(os.path.basename(im['file_name']))[0]}"
            stem = stem[:60]
            out_path = os.path.join(crops_dir, stem + ".jpg")
            crop.save(out_path, quality=95)
            n_cropped += 1

            guess, conf = ("", 0.0)
            if reader is not None:
                try:
                    guess, conf = pseudo_label(reader, crop)
                except Exception as e:                       # noqa: BLE001
                    print("  read failed:", stem, e)

            # thumbnail for the HTML, capped width so the file stays small
            thumb = crop.copy()
            thumb.thumbnail((360, 360))
            from io import BytesIO
            buf = BytesIO(); thumb.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            rows.append({"file": stem + ".jpg", "guess": guess,
                         "conf": conf, "b64": b64})

    # CSV
    with open(os.path.join(args.out, "pseudo_labels.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "guess", "confidence"])
        for r in sorted(rows, key=lambda r: r["conf"]):
            w.writerow([r["file"], r["guess"], f"{r['conf']:.4f}"])

    write_html(rows, os.path.join(args.out, "label_me.html"))
    print(f"\n{n_cropped}/{n_seen} photos cropped.")
    print("  crops      ->", crops_dir)
    print("  csv        ->", os.path.join(args.out, "pseudo_labels.csv"))
    print("  LABEL HERE ->", os.path.join(args.out, "label_me.html"))


if __name__ == "__main__":
    main()
