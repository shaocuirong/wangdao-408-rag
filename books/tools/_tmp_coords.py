import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = r"books\rag_data\ocr\计算机组成原理.jsonl"
page = int(sys.argv[1])
try:
    lo_y = int(sys.argv[2])
    hi_y = int(sys.argv[3])
except IndexError:
    lo_y, hi_y = 0, 100000

for ln in open(path, encoding="utf-8"):
    d = json.loads(ln)
    if d["page"] == page:
        print("=== pdf_page", page)
        for x in d["lines"]:
            if lo_y <= x["y0"] <= hi_y:
                print("y=%4d x0=%4d x1=%4d | %s" % (x["y0"], x["x0"], x["x1"], x["t"]))
