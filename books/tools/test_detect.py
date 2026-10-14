
import os, sys, json, shutil
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\DE_workSpace\lesrn_1\books\tools")
RAGD = r"E:\DE_workSpace\lesrn_1\books\rag_data"
peek = os.path.join(RAGD, "ocr", "_peek.jsonl")
tgt = os.path.join(RAGD, "ocr", "计算机组成原理.jsonl")

recs = {}
if os.path.exists(tgt):
    for line in open(tgt, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); recs[r["page"]] = r
added = 0
for line in open(peek, encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        if r["page"] not in recs:
            recs[r["page"]] = r; added += 1
with open(tgt, "w", encoding="utf-8") as f:
    for p in sorted(recs):
        f.write(json.dumps(recs[p], ensure_ascii=False) + "\n")
print("merged pages:", len(recs), "newly added:", added)

import importlib
bm = importlib.import_module("03_build_markdown")
pages = bm.load_pages("计算机组成原理")
rbp = bm.rows_by_page(pages)
rows = bm.all_rows(rbp)
anchors, skip = bm.detect_sections(rbp, rows)
print(f"pages={len(pages)} anchors={len(anchors)} skipped={len(skip)}")
for a in anchors:
    print(f'  L{a["level"]} p{a["page"]:>3}  {a["title"][:60]}')
