import json
import sys

path = r"books\rag_data\ocr\计算机组成原理.jsonl"
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for ln in open(path, encoding="utf-8"):
    d = json.loads(ln)
    if lo <= d["page"] <= hi:
        head = " | ".join(x["t"] for x in d["lines"][:2])
        print("=" * 20, "pdf_page", d["page"], "HEAD:", head)
        print("".join(x["t"] for x in d["lines"]))
        print()
