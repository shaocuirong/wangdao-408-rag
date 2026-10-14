# -*- coding: utf-8 -*-
"""Audit OCR completeness: find pages that produced suspiciously little text.

A page that OCR'd to a handful of lines is effectively missing from the RAG - it will
never be retrieved, and nothing downstream reports it.  This finds those pages so they
can be re-OCR'd or inspected.
"""
import os, sys, json, statistics
sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\DE_workSpace\lesrn_1"
OCR = os.path.join(ROOT, "books", "rag_data", "ocr")
BOOK_KEYS = ["操作系统", "数据结构", "计算机组成原理", "计算机网络"]
THRESH = 8

for key in BOOK_KEYS:
    path = os.path.join(OCR, key + ".jsonl")
    if not os.path.exists(path):
        continue
    recs = {}
    errs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            recs[r["page"]] = r
            if r.get("err"):
                errs.append((r["page"], r["err"][:60]))
    counts = {p: len(r["lines"]) for p, r in recs.items()}
    vals = sorted(counts.values())
    med = statistics.median(vals)
    low = sorted(p for p, n in counts.items() if n < THRESH)
    chars = {p: sum(len(l["t"]) for l in r["lines"]) for p, r in recs.items()}
    empties = sorted(p for p, c in chars.items() if c < 40)
    print(f"\n=== {key} (pages OCR'd {len(recs)}) ===")
    print(f"  lines/page: median={med} min={vals[0]} max={vals[-1]}")
    print(f"  pages with <{THRESH} lines : {len(low)}  {low[:20]}{' ...' if len(low) > 20 else ''}")
    print(f"  pages with <40 characters : {len(empties)}  {empties[:20]}{' ...' if len(empties) > 20 else ''}")
    if errs:
        print(f"  pages that errored       : {len(errs)}  {errs[:5]}")
    # front matter legitimately has little text; report it separately
    front = [p for p in empties if p < 12]
    print(f"  of the near-empty, pages<12 (front matter, expected): {len(front)}")
