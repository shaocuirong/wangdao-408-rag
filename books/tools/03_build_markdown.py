# -*- coding: utf-8 -*-
"""Stage 3 - rebuild each textbook as Markdown under books/summary_books/.

Structure comes from the PDF bookmarks when the book has them (three of the four do,
with page-exact destinations).  The 计算机组成原理 volume ships no outline at all, so
its hierarchy is recovered from the body text instead (see detect_sections).

Every section's body is the set of OCR rows lying between its own heading and the next
heading anywhere in the book, which is what makes the printed page order survive even
when several bookmarks share one page.
"""
import os, sys, json, re, difflib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import booklayout as BL

ROOT  = r"E:\DE_workSpace\lesrn_1"
BOOKS = os.path.join(ROOT, "books")
RAGD  = os.path.join(BOOKS, "rag_data")
SUMM  = os.path.join(BOOKS, "summary_books")

BOOK_KEYS = ["操作系统", "数据结构", "计算机组成原理", "计算机网络"]

# In the 计算机组成原理 volume the chapter-number glyph is set in a decorative face that
# OCR drops entirely, so "第1章 计算机系统概述" arrives as a bare "第 章" followed by the
# title on its own line.  Chapter numbers are therefore counted, not read.
CHAP_RE = re.compile(r"^第\s*([0-9一二三四五六七八九十百]*)\s*章\s*(.*)$")
SEC_RE  = re.compile(r"^\*?\s*([0-9]{1,2})\.([0-9]{1,2})(?:\.([0-9]{1,2}))?\s*(.*)$")


# ---------------------------------------------------------------- loading

def load_pages(key):
    path = os.path.join(RAGD, "ocr", key + ".jsonl")
    recs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                # The OCR writer may be mid-append, so a trailing partial line is normal.
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                recs[r["page"]] = r
    return recs


def load_toc(key):
    path = os.path.join(RAGD, "structure", key + ".toc.json")
    if not os.path.exists(path):
        return None
    items = json.load(open(path, encoding="utf-8")).get("items") or []
    return [it for it in items if it.get("page") is not None] or None


def page_rows(rec):
    """Filtered, row-merged content for one page."""
    w, h = rec["w"], rec["h"]
    if not w or not h:
        return [], None
    body, head, _foot = BL.split_margins(rec["lines"], w, h)
    body = [L for L in body if not BL.is_promo(L["t"])]
    label = BL.book_page_label(head, w)
    rows = [r for r in BL.merge_rows(body) if not BL.is_promo(r["t"])]
    return rows, label


def rows_by_page(pages):
    """{page: [row, ...]} with margins, promos and split heading boxes already resolved."""
    return {pg: page_rows(pages[pg])[0] for pg in pages}


def all_rows(rbp):
    """[(page, row_index, row)] in reading order."""
    out = []
    for pg in sorted(rbp):
        for i, r in enumerate(rbp[pg]):
            out.append((pg, i, r))
    return out


def paragraphs(rows):
    """Rebuild paragraphs, restarting the vertical-gap logic at each page break."""
    out, buf, cur = [], [], None
    for (p, i, r) in rows:
        if p != cur:
            if buf:
                out.extend(BL.build_blocks([x[2] for x in buf]))
            buf, cur = [], p
        buf.append((p, i, r))
    if buf:
        out.extend(BL.build_blocks([x[2] for x in buf]))
    return out


# ---------------------------------------------------------------- anchoring

_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)")


def _num_prefix(normalised):
    """Leading section number of a heading, e.g. "8.1.2", or None."""
    m = _NUM_RE.match(normalised)
    return m.group(1) if m else None


def find_anchor(rows, title, occurrence=0):
    """Index of the row that renders this heading, or None.

    The occurrence index disambiguates headings that repeat on one page.  These books
    print "二、综合应用题" twice in every section - once under 试题精选 and once under
    答案与解析 - and both bookmarks carry the same title AND the same page.  Always
    taking the first match pointed both anchors at the same row, so the question section
    was left with no body while the answer section silently absorbed everything.
    """
    nk = BL.norm_key(title)
    if not nk:
        return None
    want_num = _num_prefix(nk)
    # Returning None rather than clamping matters: a second bookmark for the same title
    # on a page that prints it once must be allowed to look elsewhere, otherwise both
    # anchors land on one row and the earlier section is left with no body.
    exact = [i for i, r in enumerate(rows) if BL.norm_key(r["t"]) == nk]
    if exact:
        return exact[occurrence] if occurrence < len(exact) else None
    loose = []
    for i, r in enumerate(rows):
        rk = BL.norm_key(r["t"])
        if rk and min(len(rk), len(nk)) >= 3 and (nk in rk or rk in nk):
            loose.append(i)
    if loose and want_num:
        # "8.1.2 本节试题精选" and "8.2.4 本节试题精选" both contain the bare row
        # "本节试题精选"; without the number check they collapse onto the same row and
        # one of the two sections is left with no body at all.
        numbered = [i for i in loose if _num_prefix(BL.norm_key(rows[i]["t"])) == want_num]
        if numbered:
            loose = numbered
    if loose:
        return loose[occurrence] if occurrence < len(loose) else None
    scored = []
    for i, r in enumerate(rows):
        rk = BL.norm_key(r["t"])
        # Never let fuzzy matching cross section numbers: "8.2.4 本节试题精选" is 0.9
        # similar to "8.1.2 本节试题精选" and would otherwise be handed to it, putting a
        # whole section's body under the wrong heading.
        if want_num:
            rn = _num_prefix(rk)
            if rn is not None and rn != want_num:
                continue
        s = difflib.SequenceMatcher(None, rk, nk).ratio()
        if s >= 0.78:
            scored.append((s, i))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    return scored[occurrence][1] if occurrence < len(scored) else None


def drop_prefix_duplicates(toc):
    """Several volumes bookmark both "第1章" (the divider) and "第1章 计算机系统概述"
    on the same page.  Keeping both yields an empty chapter file, so drop the shorter."""
    kept = []
    for i, it in enumerate(toc):
        nxt = toc[i + 1] if i + 1 < len(toc) else None
        if nxt and nxt["level"] == it["level"] and nxt["page"] == it["page"]:
            a, b = BL.norm_key(it["title"]), BL.norm_key(nxt["title"])
            if a and b.startswith(a) and len(b) > len(a):
                continue
        kept.append(it)
    return kept


def build_key_index(pages_rows):
    """{normalised heading text: [(page, row), ...]} for the whole book.

    Lets a bookmark that misses its page be placed at the nearest real occurrence,
    however far off the outline is - offsets of three pages do occur.
    """
    idx = {}
    for pg in sorted(pages_rows):
        for i, r in enumerate(pages_rows[pg]):
            k = BL.norm_key(r["t"])
            if k:
                idx.setdefault(k, []).append((pg, i))
    return idx


def anchors_from_toc(pages_rows, toc):
    anchors = []
    key_index = build_key_index(pages_rows)
    claimed = set()    # (page, row) already assigned to some bookmark
    seen = {}          # (page, title key) -> how many times already anchored
    for it in drop_prefix_duplicates(toc):
        pg = it["page"]
        rows = pages_rows.get(pg, [])
        k = (pg, BL.norm_key(it["title"]))
        occ = seen.get(k, 0)
        seen[k] = occ + 1
        ai = find_anchor(rows, it["title"], occ)
        found_pg = pg
        # A row already handed to an earlier bookmark is not available again.  Two
        # bookmarks sharing a title but claiming different pages each start at occurrence
        # 0, so without this the nearby-page scan gives the second one the very row the
        # first is using.
        if ai is not None and (found_pg, ai) in claimed:
            ai = None
        if ai is None:
            # Bookmarks are occasionally a page off - the PDF outline points at 211 while
            # the heading is printed on 210.  Without this, the anchor silently collapsed
            # to the top of its page and neighbouring content landed in the wrong section.
            for off in (-1, 1, -2, 2):
                q = pg + off
                if q not in pages_rows:
                    continue
                cand = find_anchor(pages_rows[q], it["title"], 0)
                if cand is not None and (q, cand) not in claimed:
                    ai, found_pg = cand, q
                    break
        if ai is None:
            # Still nothing: fall back to the nearest exact occurrence anywhere in the
            # book.  The outline can be several pages out - "8.4 选择排序" is printed on
            # 362 while its bookmark claims 365.
            cands = key_index.get(BL.norm_key(it["title"]), [])
            if cands:
                same = [c for c in cands if c[0] == pg and c not in claimed]
                if occ < len(same):
                    pick = same[occ]
                else:
                    # Otherwise take the nearest occurrence not already spoken for.
                    ordered = sorted(cands, key=lambda c: (abs(c[0] - pg), c[0]))
                    free = [c for c in ordered if c not in claimed]
                    pick = free[0] if free else None
                if pick is not None:
                    ai, found_pg = pick[1], pick[0]
        if ai is not None:
            claimed.add((found_pg, ai))
        if found_pg != pg:
            pg = found_pg
            rows = pages_rows.get(pg, [])
        anchors.append({
            "level": it["level"], "title": it["title"].strip(), "page": pg,
            "pos": (pg, ai if ai is not None else 0),
            "matched": ai is not None, "rows": [],
        })
    return anchors


def assign_rows(anchors, rows, skip=()):
    """Give every row to the most recent heading at or before it."""
    ordered = sorted(anchors, key=lambda a: a["pos"])
    skip = set(skip)
    ai, cur = 0, None
    for (pg, i, r) in rows:
        pos = (pg, i)
        while ai < len(ordered) and ordered[ai]["pos"] <= pos:
            cur = ordered[ai]
            ai += 1
        if cur is None or pos in skip:
            continue
        if cur["matched"] and pos == cur["pos"]:
            continue                      # the heading itself, already emitted as a title
        cur["rows"].append((pg, i, r))


# ---------------------------------------------------------------- structure

def detect_sections(pages_rows, rows):
    """Recover a chapter/section hierarchy from body text (books with no outline).

    Two independent signals mark a heading: it is set noticeably larger than body text,
    and it is preceded by noticeably more leading than a normal line break.  Requiring
    one of them AND the size gate keeps ordinary body lines that merely start with a
    number - "1.冯·诺依曼机的基本思想", "2）第二代计算机..." - out of the outline.
    """
    if not rows:
        return [], set()
    hs = sorted(r["y1"] - r["y0"] for (_p, _i, r) in rows)
    med_h = hs[len(hs) // 2] or 26
    gaps = []
    for a, b in zip(rows, rows[1:]):
        if a[0] == b[0]:
            g = b[2]["y0"] - a[2]["y1"]
            if g > 0:
                gaps.append(g)
    gaps.sort()
    med_gap = gaps[len(gaps) // 2] if gaps else med_h * 1.4

    # A printed table of contents is a numbered list too, so it tags every entry as a
    # heading.  Real chapter openers always sit high on their page, which separates them
    # cleanly; everything before the first one is front matter and is not scanned.
    page_h = {pg: max(r["y1"] for (_p, _i, r) in rows if _p == pg) for pg, _ in
              [(p, 0) for p in sorted({x[0] for x in rows})]}
    first_chapter = None
    for (pg, i, r) in rows:
        t = r["t"].strip()
        m = CHAP_RE.match(t)
        if (m and len(t) <= 16 and (r["y1"] - r["y0"]) > 1.25 * med_h
                and r["y0"] < 0.35 * max(page_h.get(pg, 1), 1)):
            first_chapter = pg
            break

    anchors, skip, chap_no = [], set(), 0
    for idx, (pg, i, r) in enumerate(rows):
        if first_chapter is not None and pg < first_chapter:
            continue
        t = r["t"].strip()
        h = r["y1"] - r["y0"]
        # A heading that opens a page follows a page break, so treat it as spaced.
        if idx and rows[idx - 1][0] == pg:
            gap = r["y0"] - rows[idx - 1][2]["y1"]
        else:
            gap = med_gap * 2

        m = CHAP_RE.match(t)
        if (m and h > 1.25 * med_h and len(t) <= 30
                and r["y0"] < 0.35 * max(page_h.get(pg, 1), 1)):
            num = m.group(1).strip()
            rest = m.group(2).strip()
            # Prefer the number OCR actually read; fall back to counting openers.
            if num.isdigit():
                chap_no = int(num)
            elif num:
                chap_no += 1
            else:
                chap_no += 1
            # The title is often laid out as its own line below "第N章" (the decorative
            # number glyph is unreliable), so pick it up from the next row whenever the
            # heading itself carries no title - whether or not the number was read.
            if not rest and idx + 1 < len(rows) and rows[idx + 1][0] == pg:
                nxt = rows[idx + 1]
                cand = nxt[2]["t"].strip()
                if 0 < len(cand) <= 30 and not cand.startswith(("【", "（", "(")):
                    rest = cand
                    skip.add((pg, nxt[1]))
            title = (f"第{chap_no}章 {rest}".strip() if rest else f"第{chap_no}章")
            anchors.append({"level": 0, "title": title, "page": pg, "pos": (pg, i),
                            "matched": True, "rows": []})
            continue

        # The "N.N" prefix is the reliable signal: body text on these pages essentially
        # never starts that way.  The guards below reject the two shapes that do -
        # measurements ("1.5GHz的主频") and cross references ("3.1节介绍的...").
        sm = SEC_RE.match(t)
        if sm and h >= med_h and len(t) <= 42:
            rest = (sm.group(4) or "").strip()
            ok = rest and not rest[0].isascii() or (rest and not rest[0].isalnum())
            if ok and not rest.startswith(("节", "章", "小节", "所述", "介绍")):
                anchors.append({"level": 2 if sm.group(3) else 1, "title": t,
                                "page": pg, "pos": (pg, i), "matched": True, "rows": []})
    return anchors, skip


def build_tree(anchors):
    root = {"level": -1, "title": "", "children": [], "rows": []}
    stack = [root]
    for a in anchors:
        while len(stack) > 1 and stack[-1]["level"] >= a["level"]:
            stack.pop()
        node = {"level": a["level"], "title": a["title"], "page": a["page"],
                "rows": a["rows"], "children": []}
        stack[-1]["children"].append(node)
        stack.append(node)
    return root


# ---------------------------------------------------------------- emitting

def safe_name(text, idx):
    text = re.sub(r"[\\/:*?\"<>|\r\n\t]", "", text)
    text = re.sub(r"\s+", "", text)
    return f"{idx:02d}_{text[:48]}" if text else f"{idx:02d}_untitled"


def render(node, depth, out):
    if node["title"]:
        out.append("#" * min(depth, 6) + " " + node["title"])
        out.append("")
    if node.get("page") is not None:
        out.append(f"<!-- pdf_page={node['page']} -->")
        out.append("")
    for para in paragraphs(node["rows"]):
        out.append(para)
        out.append("")
    for ch in node["children"]:
        render(ch, depth + 1, out)


def build_book(key, limit=None):
    pages = load_pages(key)
    if not pages:
        return None
    rbp = rows_by_page(pages)
    rows = all_rows(rbp)
    toc = load_toc(key)
    skip = ()
    if toc:
        anchors = anchors_from_toc(rbp, toc)
        source = "bookmarks"
    else:
        anchors, skip = detect_sections(rbp, rows)
        source = "detected"
    assign_rows(anchors, rows, skip)
    tree = build_tree(anchors)

    outdir = os.path.join(SUMM, key)
    os.makedirs(outdir, exist_ok=True)
    for old in os.listdir(outdir):
        if old.endswith(".md"):
            os.remove(os.path.join(outdir, old))

    parts = tree["children"]
    if limit:
        parts = parts[:limit]
    written = []
    for idx, part in enumerate(parts):
        lines = [f"# {part['title']}", ""]
        for ch in part["children"]:
            render(ch, 2, lines)
        fname = safe_name(part["title"], idx) + ".md"
        with open(os.path.join(outdir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        written.append(fname)

    full = []
    for part in parts:
        full.append(f"# {part['title']}")
        full.append("")
        for ch in part["children"]:
            render(ch, 2, full)
    with open(os.path.join(SUMM, key + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(full).rstrip() + "\n")

    with open(os.path.join(RAGD, "structure", key + ".tree.json"), "w", encoding="utf-8") as f:
        json.dump({"key": key, "source": source, "pages": len(pages),
                   "anchors": [{k: v for k, v in a.items() if k != "rows"} for a in anchors]},
                  f, ensure_ascii=False, indent=1)

    unmatched = sum(1 for a in anchors if not a["matched"])
    print(f"[{key}] pages={len(pages)} source={source} anchors={len(anchors)} "
          f"unmatched={unmatched} parts={len(parts)}", flush=True)
    return written


if __name__ == "__main__":
    only = sys.argv[1:] or BOOK_KEYS
    for k in only:
        build_book(k)
