# -*- coding: utf-8 -*-
"""Frequency analysis of page-edge lines, to derive safe header/footer/watermark filters."""
import os, sys, json, collections, re
sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\DE_workSpace\lesrn_1"
OCR = os.path.join(ROOT, "books", "rag_data", "ocr")

def load(key):
    recs = []
    with open(os.path.join(OCR, key + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def norm(t):
    t = re.sub(r"[\s\u3000]+", "", t)
    t = re.sub(r"[0-9０-９]", "#", t)
    return t

for key in ["操作系统"]:
    recs = load(key)
    print(f"== {key}: {len(recs)} pages ==")
    top = collections.Counter(); bot = collections.Counter()
    right = collections.Counter(); left = collections.Counter()
    for r in recs:
        w = r["w"] or 1; h = r["h"] or 1
        for L in r["lines"]:
            ry0 = L["y0"]/h; ry1 = L["y1"]/h; rx0 = L["x0"]/w
            if ry1 < 0.08: top[norm(L["t"])] += 1
            if ry0 > 0.92: bot[norm(L["t"])] += 1
            if rx0 > 0.80 and 0.08 < ry0 < 0.92: right[norm(L["t"])] += 1
            if L["x1"]/w < 0.20 and 0.08 < ry0 < 0.92: left[norm(L["t"])] += 1
    for name, c in (("TOP(<8%)", top), ("BOTTOM(>92%)", bot), ("RIGHT(x>80%)", right), ("LEFT(x<20%)", left)):
        print(f"\n--- {name} : {len(c)} distinct ---")
        for t, n in c.most_common(12):
            print(f"   {n:>4}x  {t[:70]}")
