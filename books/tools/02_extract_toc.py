# -*- coding: utf-8 -*-
"""Stage 2a: extract bookmark (outline) trees with resolved page indexes."""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import pypdfium2 as pdfium

ROOT = r"E:\DE_workSpace\lesrn_1"
BOOKS = os.path.join(ROOT, "books")
OUT = os.path.join(ROOT, "rag", "structure")
os.makedirs(OUT, exist_ok=True)

BOOKS_MAP = {
    "操作系统":       "王道2027操作系统-高清带书签.pdf",
    "数据结构":       "2027数据结构_高清带书签版.pdf",
    "计算机组成原理": "2027计算机组成原理_高清带书签版.pdf",
    "计算机网络":     "2027计算机网络-高清带书签版.pdf",
}
# fix: exact filename from listing
BOOKS_MAP["计算机网络"] = "2027计算机网络_高清带书签版.pdf"

def main():
    for key, fname in BOOKS_MAP.items():
        path = os.path.join(BOOKS, fname)
        doc = pdfium.PdfDocument(path)
        n = len(doc)
        items = []
        for bm in doc.get_toc():
            try:
                title = (bm.get_title() or "").strip()
            except Exception:
                title = ""
            try:
                idx = bm.get_dest().get_index()
            except Exception:
                idx = None
            items.append({"level": int(bm.level), "title": title, "page": idx})
        doc.close()
        rec = {"key": key, "file": fname, "pages": n, "count": len(items), "items": items}
        with open(os.path.join(OUT, key + ".toc.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        lv = {}
        for it in items:
            lv[it["level"]] = lv.get(it["level"], 0) + 1
        print(f"{key}: pages={n} bookmarks={len(items)} levels={lv}")
        if items:
            print("   first:", items[0]["title"], "| last:", items[-1]["title"])

if __name__ == "__main__":
    main()
