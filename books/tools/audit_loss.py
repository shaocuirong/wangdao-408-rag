# -*- coding: utf-8 -*-
"""How much of the OCR'd text actually reaches the Markdown?

Every other check measures what IS in the output.  This one measures what is NOT: text
that was recognised but never landed in any section would be silently missing from the
RAG, and nothing downstream would notice.
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\DE_workSpace\lesrn_1\books\tools")
import booklayout as BL

ROOT = r"E:\DE_workSpace\lesrn_1\books"
OCR = os.path.join(ROOT, "rag_data", "ocr")
SUMM = os.path.join(ROOT, "summary_books")
BOOKS = ["操作系统", "数据结构", "计算机组成原理", "计算机网络"]

for key in BOOKS:
    op = os.path.join(OCR, key + ".jsonl")
    if not os.path.exists(op):
        continue
    raw = 0          # all recognised characters
    body = 0         # characters that survive the margin/watermark filter
    for line in open(op, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        h, w = r["h"], r["w"]
        for L in r["lines"]:
            n = len(L["t"].strip())
            raw += n
            if L["y1"] < BL.TOP_FRAC * h or L["y0"] > BL.BOT_FRAC * h:
                continue
            if BL.is_promo(L["t"]):
                continue
            body += n

    d = os.path.join(SUMM, key)
    md = 0
    for f in os.listdir(d):
        if not f.endswith(".md"):
            continue
        for ln in open(os.path.join(d, f), encoding="utf-8"):
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("<!--"):
                continue
            md += len(ln)

    print(f"{key:<16} 识别 {raw:>8}  过滤后 {body:>8}  Markdown {md:>8}   "
          f"占识别 {100*md/raw:5.1f}%   占过滤后 {100*md/max(body,1):5.1f}%")
