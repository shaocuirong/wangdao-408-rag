import json
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = r"books\rag_data\ocr\计算机组成原理.jsonl"
pats = sys.argv[1:]

pages = []
for ln in open(path, encoding="utf-8"):
    d = json.loads(ln)
    text = "".join(x["t"] for x in d["lines"])
    # printed page number = first short numeric token if it looks like a page number
    printed = ""
    for x in d["lines"][:2]:
        if re.fullmatch(r"\d{2,3}", x["t"].strip()):
            printed = x["t"].strip()
            break
    pages.append((d["page"], printed, text))

for p in pats:
    print("### pattern:", p)
    for page, printed, text in pages:
        for m in re.finditer(p, text):
            s = max(0, m.start() - 40)
            e = min(len(text), m.end() + 60)
            print("  pdf_page=%s printed=%s ...%s..." % (page, printed or "?", text[s:e].replace("\n", " ")))
    print()
