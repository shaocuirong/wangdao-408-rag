# -*- coding: utf-8 -*-
"""Structural QA for the four books: is the extracted outline usable?

Run it after a build, or mid-OCR, to see how far each book has got and whether any
chapter came out malformed.  It is cheap - it reads only the derived artefacts.

    python books/tools/verify_structure.py
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding="utf-8")

ROOT  = r"E:\DE_workSpace\lesrn_1"
BOOKS = os.path.join(ROOT, "books")
RAGD  = os.path.join(BOOKS, "rag_data")
SUMM  = os.path.join(BOOKS, "summary_books")
BOOK_KEYS = ["操作系统", "数据结构", "计算机组成原理", "计算机网络"]
EXPECTED_CHAPTERS = {"操作系统": 5, "数据结构": 8, "计算机组成原理": 7, "计算机网络": 6}
# Front/back matter that is deliberately kept as Markdown but excluded from the index.
MATTER = ("目录", "前言", "致读者", "王道训练营", "参考文献", "思维拓展")


def pdf_pages(key):
    import pypdfium2 as pdfium
    FILES = {
        "操作系统": "王道2027操作系统-高清带书签.pdf",
        "数据结构": "2027数据结构_高清带书签版.pdf",
        "计算机组成原理": "2027计算机组成原理_高清带书签版.pdf",
        "计算机网络": "2027计算机网络_高清带书签版.pdf",
    }
    d = pdfium.PdfDocument(os.path.join(BOOKS, FILES[key]))
    n = len(d)
    d.close()
    return n


def ocr_pages(key):
    p = os.path.join(RAGD, "ocr", key + ".jsonl")
    if not os.path.exists(p):
        return 0
    n = 0
    for line in open(p, encoding="utf-8"):
        if line.strip():
            n += 1
    return n


def main():
    problems = []
    for key in BOOK_KEYS:
        total, done = pdf_pages(key), ocr_pages(key)
        tree_p = os.path.join(RAGD, "structure", key + ".tree.json")
        src, anchors, unmatched = "-", 0, 0
        if os.path.exists(tree_p):
            d = json.load(open(tree_p, encoding="utf-8"))
            src = d.get("source", "-")
            anchors = len(d.get("anchors", []))
            unmatched = sum(1 for a in d["anchors"] if not a.get("matched"))
        bdir = os.path.join(SUMM, key)
        chapters = []
        if os.path.isdir(bdir):
            for f in sorted(os.listdir(bdir)):
                if f.endswith(".md"):
                    chapters.append((f, os.path.getsize(os.path.join(bdir, f))))
        body = [c for c in chapters if not any(m in c[0] for m in MATTER)]
        pct = 100.0 * done / total if total else 0
        print(f"\n=== {key} ===")
        print(f"  OCR      : {done}/{total} pages ({pct:.0f}%)")
        print(f"  outline  : source={src} anchors={anchors} unmatched={unmatched}")
        print(f"  markdown : {len(chapters)} files, {len(body)} content chapters")
        for f, sz in body:
            flag = ""
            if sz < 400:
                flag = "   <-- looks empty"
            if re.match(r"^\d+_第\d+章\.md$", f):
                flag = "   <-- chapter title missing"
            if flag:
                problems.append(f"{key}/{f}{flag}")
            print(f"      {f:<34} {sz/1024:>8.1f} KB{flag}")
        want = EXPECTED_CHAPTERS[key]
        if done == total and len(body) != want:
            problems.append(f"{key}: {len(body)} content chapters, expected {want}")

    print("\n================ PROBLEMS ================")
    print("\n".join(problems) if problems else "none")


if __name__ == "__main__":
    main()
