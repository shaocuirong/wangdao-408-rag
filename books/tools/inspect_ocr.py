# -*- coding: utf-8 -*-
"""Peek at raw OCR records: geometry, header/footer zones, line metrics."""
import os, sys, json, collections
sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\DE_workSpace\lesrn_1"
OCR = os.path.join(ROOT, "books", "rag_data", "ocr")

key = sys.argv[1] if len(sys.argv) > 1 else "操作系统"
pages_wanted = [int(x) for x in sys.argv[2:]] or [0, 8, 12, 30]

recs = {}
with open(os.path.join(OCR, key + ".jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        recs[r["page"]] = r

print(f"{key}: {len(recs)} pages OCR'd, max page {max(recs) if recs else -1}")
for p in pages_wanted:
    r = recs.get(p)
    if not r:
        print(f"\n--- page {p}: NOT DONE ---")
        continue
    print(f"\n=== page {p}  {r['w']}x{r['h']}  lines={len(r['lines'])} err={r['err']} ===")
    for L in r["lines"]:
        rel_y = L["y0"] / max(r["h"], 1)
        flag = ""
        if rel_y < 0.07: flag = " <TOP"
        elif L["y0"] / max(r["h"],1) > 0.93: flag = " <BOT"
        print(f"  y{L['y0']:>5}-{L['y1']:<5} x{L['x0']:>5}-{L['x1']:<5} ({rel_y:.2f}){flag} {L['t']}")
