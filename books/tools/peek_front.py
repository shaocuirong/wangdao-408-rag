# -*- coding: utf-8 -*-
"""OCR a small page range into a side file, for designing structure detection."""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ["OMP_NUM_THREADS"] = "1"
from rapidocr_onnxruntime import RapidOCR
import pypdfium2 as pdfium

ROOT = r"E:\DE_workSpace\lesrn_1"
BOOKS = os.path.join(ROOT, "books")
OUT = os.path.join(BOOKS, "rag_data", "ocr", "_peek.jsonl")

key = sys.argv[1]
start, end = int(sys.argv[2]), int(sys.argv[3])
FILES = {
    "计算机组成原理": "2027计算机组成原理_高清带书签版.pdf",
}
eng = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
doc = pdfium.PdfDocument(os.path.join(BOOKS, FILES[key]))
with open(OUT, "w", encoding="utf-8") as f:
    for p in range(start, end):
        img = doc[p].render(scale=2.5).to_numpy()
        h, w = img.shape[:2]
        res, _ = eng(img, use_cls=False)
        lines = []
        if res:
            for box, text, score in res:
                xs = [float(q[0]) for q in box]; ys = [float(q[1]) for q in box]
                lines.append({"x0": round(min(xs)), "y0": round(min(ys)),
                              "x1": round(max(xs)), "y1": round(max(ys)),
                              "t": text, "s": round(float(score), 3)})
        f.write(json.dumps({"key": key, "page": p, "w": w, "h": h, "lines": lines, "err": None},
                           ensure_ascii=False) + "\n")
        f.flush()
        print(f"p{p} lines={len(lines)}", flush=True)
doc.close()
