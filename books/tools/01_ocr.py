# -*- coding: utf-8 -*-
"""Stage 1 - OCR every textbook page into books/rag_data/ocr/<book>.jsonl

Findings that shaped this script (all measured on this machine):
  * The four PDFs are image-only scans, so OCR is mandatory.
  * The angle classifier costs ~45% of runtime and the pages are upright -> disabled.
  * ONNX Runtime defaults to intra_op = -1 (every core).  Left alone, N workers
    oversubscribe badly; pinned to 1 thread each, CPU per page is ~14 s.
  * OpenCV/OpenMP/BLAS are pinned too, for the same reason.
  * Aggregate throughput saturates at ~0.13 pages/s from 4 workers upward -- the box
    is shared with the desktop, browser and antivirus -- so extra workers only add
    heat and context switching.  The default here is deliberately modest.
  * The parent owns the output files: free resume, no lock contention, instant
    heartbeat (every record is flushed).
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import json, time
sys.stdout.reconfigure(encoding="utf-8")

import multiprocessing as mp
import pypdfium2 as pdfium

ROOT   = r"E:\DE_workSpace\lesrn_1"
BOOKS  = os.path.join(ROOT, "books")
OUT    = os.path.join(BOOKS, "rag_data", "ocr")
os.makedirs(OUT, exist_ok=True)

RENDER_SCALE = 2.5
NPROC        = int(os.environ.get("OCR_NPROC", "6"))

BOOKS_MAP = {
    "操作系统":       "王道2027操作系统-高清带书签.pdf",
    "数据结构":       "2027数据结构_高清带书签版.pdf",
    "计算机组成原理": "2027计算机组成原理_高清带书签版.pdf",
    "计算机网络":     "2027计算机网络_高清带书签版.pdf",
}

_engine = None
_doc = None
_doc_path = None


def _init():
    global _engine
    import numpy as np
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    from rapidocr_onnxruntime import RapidOCR
    _engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    _engine(np.full((64, 64, 3), 255, dtype=np.uint8), use_cls=False)


def _get_doc(path):
    global _doc, _doc_path
    if _doc_path != path:
        if _doc is not None:
            _doc.close()
        _doc = pdfium.PdfDocument(path)
        _doc_path = path
    return _doc


def work(task):
    key, pdf_path, page_index = task
    try:
        doc = _get_doc(pdf_path)
        img = doc[page_index].render(scale=RENDER_SCALE).to_numpy()
        h, w = img.shape[:2]
        res, _ = _engine(img, use_cls=False, use_det=True, use_rec=True)
        lines = []
        if res:
            for box, text, score in res:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                lines.append({
                    "x0": round(min(xs)), "y0": round(min(ys)),
                    "x1": round(max(xs)), "y1": round(max(ys)),
                    "t": text, "s": round(float(score), 3),
                })
        return {"key": key, "page": page_index, "w": w, "h": h, "lines": lines, "err": None}
    except Exception as e:
        return {"key": key, "page": page_index, "w": 0, "h": 0, "lines": [], "err": repr(e)}


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["page"])
                except Exception:
                    pass
    return done


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else list(BOOKS_MAP)
    tasks, meta, pending = [], {}, []
    for key in only:
        pdf_path = os.path.join(BOOKS, BOOKS_MAP[key])
        out_path = os.path.join(OUT, key + ".jsonl")
        doc = pdfium.PdfDocument(pdf_path)
        n = len(doc)
        doc.close()
        done = load_done(out_path)
        meta[key] = (out_path, n, len(done))
        todo = [p for p in range(n) if p not in done]
        if todo:
            pending.append([key, pdf_path, todo, 0])
        print(f"[plan] {key}: {n} pages, {len(done)} already done", flush=True)

    # Round-robin across books rather than finishing one at a time.  An interrupted run
    # then leaves every subject partly covered instead of three subjects with nothing,
    # and a mid-run index rebuild is useful for all four books at once.
    while True:
        progressed = False
        for entry in pending:
            key, pdf_path, todo, idx = entry
            if idx < len(todo):
                tasks.append((key, pdf_path, todo[idx]))
                entry[3] = idx + 1
                progressed = True
        if not progressed:
            break

    if not tasks:
        print("[done] nothing to do", flush=True)
        return

    print(f"[start] {len(tasks)} pages, {NPROC} workers x 1 thread", flush=True)
    t0 = time.time()
    handles = {k: open(v[0], "a", encoding="utf-8") for k, v in meta.items()}
    per_book = {k: 0 for k in meta}
    done_n, last_report = 0, time.time()
    try:
        with mp.Pool(NPROC, initializer=_init) as pool:
            for rec in pool.imap_unordered(work, tasks, chunksize=1):
                handles[rec["key"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
                handles[rec["key"]].flush()
                done_n += 1
                per_book[rec["key"]] += 1
                now = time.time()
                if now - last_report >= 30:
                    last_report = now
                    el = now - t0
                    rate = done_n / el
                    detail = " ".join(f"{k}={v}" for k, v in per_book.items() if v)
                    print(f"[prog] {done_n}/{len(tasks)} {rate:.3f} pg/s elapsed={el/60:.1f}m "
                          f"eta={(len(tasks)-done_n)/rate/60:.1f}m | {detail}", flush=True)
    finally:
        for h in handles.values():
            h.close()
    print(f"[done] {done_n} pages in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
