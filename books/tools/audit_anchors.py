# -*- coding: utf-8 -*-
"""Do two bookmarks resolve to the same row?

That is the failure mode behind "this section is empty while its neighbour has twice the
text": whichever anchor comes second wins the rows and the first is starved.  Two kinds
of collision exist and only one is a problem:

  * parent/child at one position - benign.  A parent section's body legitimately lives
    under its first child, so both anchors pointing at the same heading is correct.
  * same-level (or out-of-order) at one position - harmful.  Two sibling sections cannot
    both start at the same row.

Collisions on pages that have not been OCR'd yet are also benign: with no rows to match,
every bookmark for that page falls back to row 0.  They separate once the page arrives.
"""
import os, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
bm = importlib.import_module("03_build_markdown")

BOOK_KEYS = ["操作系统", "数据结构", "计算机组成原理", "计算机网络"]
problems = []

for key in BOOK_KEYS:
    pages = bm.load_pages(key)
    if not pages:
        continue
    rbp = bm.rows_by_page(pages)
    toc = bm.load_toc(key)
    if toc:
        anchors = bm.anchors_from_toc(rbp, toc)
        src = "bookmarks"
    else:
        anchors, _ = bm.detect_sections(rbp, bm.all_rows(rbp))
        src = "detected"

    bypos = collections.defaultdict(list)
    for a in anchors:
        bypos[a["pos"]].append(a)

    harmful, pending = [], 0
    for pos, group in bypos.items():
        if len(group) < 2:
            continue
        # a page with no rows means "not OCR'd yet", not "ambiguous heading"
        if not rbp.get(pos[0]):
            pending += 1
            continue
        lv = [a["level"] for a in group]
        if not all(lv[i] < lv[i + 1] for i in range(len(lv) - 1)):
            harmful.append((pos, group))

    print(f"{key:<16} [{src}] anchors={len(anchors):<4} "
          f"有害碰撞={len(harmful)}  待抽取页上的碰撞={pending}")
    for pos, group in harmful[:5]:
        print(f"      p{pos[0]} row{pos[1]}: " +
              " | ".join(f"L{a['level']}:{a['title'][:24]}" for a in group))
    if harmful:
        problems.append(key)

print("\n================ PROBLEMS ================")
print("\n".join(problems) if problems else "none")
