# -*- coding: utf-8 -*-
"""Stage 4 - cut the Markdown into retrievable chunks.

Chunking follows the book structure rather than a blind sliding window: a chunk never
spans two chapters, and every chunk carries its chapter/section breadcrumb so an answer
can cite "操作系统 第3章 3.1.2" instead of an anonymous blob.

Sizing targets bge-base-zh-v1.5's 512-token window.  Chinese runs close to one token per
character, so ~400 characters is a safe ceiling with headroom for the breadcrumb prefix.
"""
import os, sys, json, re, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import booklayout as BL

ROOT  = r"E:\DE_workSpace\lesrn_1"
BOOKS = os.path.join(ROOT, "books")
RAGD  = os.path.join(BOOKS, "rag_data")
SUMM  = os.path.join(BOOKS, "summary_books")
OUT   = os.path.join(RAGD, "chunks")

BOOK_KEYS = ["操作系统", "数据结构", "计算机组成原理", "计算机网络"]

TARGET = 400      # preferred chunk size in characters
MAXC   = 520      # hard ceiling before a paragraph is force-split
MINC   = 60       # drop fragments shorter than this
OVERLAP = 1       # paragraphs of tail context carried into the next chunk

# Front/back matter is navigation and advertising, not 408 content.  Indexing it makes
# it show up as a hit for unrelated questions, so it is excluded from the knowledge base.
SKIP_FILES = ("目录", "前言", "致读者", "王道训练营", "参考文献", "思维拓展")

# Chinese sentence enders, plus newline, used when a paragraph must be split.
_SENT = re.compile(r"(?<=[。！？；!?;])")


def iter_chapter_files(book_dir):
    for name in sorted(os.listdir(book_dir)):
        if name.endswith(".md"):
            yield name, os.path.join(book_dir, name)


def split_long(text, limit):
    """Split an over-long paragraph on sentence boundaries."""
    parts, buf = [], ""
    for s in _SENT.split(text):
        if not s:
            continue
        if buf and len(buf) + len(s) > limit:
            parts.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        parts.append(buf)
    out = []
    for p in parts:
        if len(p) <= limit:
            out.append(p)
        else:
            for i in range(0, len(p), limit):
                out.append(p[i:i + limit])
    return out


def parse_markdown(path):
    """Return a flat list of blocks: {'kind': 'h1'|'h2'|..., 'text': str}."""
    blocks = []
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            blocks.append({"kind": "h" + str(len(m.group(1))), "text": m.group(2).strip()})
        elif line.startswith("<!--"):
            m = re.search(r"pdf_page=(\d+)", line)
            if m:
                blocks.append({"kind": "page", "page": int(m.group(1))})
            continue
        else:
            blocks.append({"kind": "p", "text": line.strip()})
    return blocks


def chunk_file(book, fname, path):
    blocks = parse_markdown(path)
    chapter = ""
    trail = {}          # heading depth -> title, for the breadcrumb
    chunks, buf, pages = [], [], []
    cur_page = [None]   # boxed so the nested flush() can read it

    def flush():
        nonlocal buf, pages
        text = "\n".join(buf).strip()
        if len(text) >= MINC:
            rng = [min(pages), max(pages)] if pages else None
            trail_txt = " > ".join(trail[k] for k in sorted(trail) if k != "h1")
            chunks.append({
                "book": book,
                "chapter": chapter,
                "section": trail_txt,
                "pdf_pages": rng,
                "text": text,
            })
        buf = []
        pages = []

    for b in blocks:
        if b["kind"] == "page":
            cur_page[0] = b["page"]
            continue
        if b["kind"].startswith("h"):
            depth = int(b["kind"][1])
            title = BL.norm_plain(b["text"])
            if depth == 1:
                chapter = title
                trail = {"h1": title}
            else:
                trail = {k: v for k, v in trail.items() if int(k[1]) < depth}
                trail[b["kind"]] = title
            if buf:
                flush()
            buf = [b["text"]]
            continue
        pieces = split_long(b["text"], MAXC) if len(b["text"]) > MAXC else [b["text"]]
        for piece in pieces:
            if buf and sum(len(x) for x in buf) + len(piece) > TARGET:
                flush()
            buf.append(piece)
            if cur_page[0] is not None:
                pages.append(cur_page[0])
    if buf:
        flush()
    return chunks


def main():
    os.makedirs(OUT, exist_ok=True)
    total = 0
    summary = {}
    for book in BOOK_KEYS:
        bdir = os.path.join(SUMM, book)
        if not os.path.isdir(bdir):
            continue
        all_chunks = []
        for fname, path in iter_chapter_files(bdir):
            if any(s in fname for s in SKIP_FILES):
                continue
            all_chunks.extend(chunk_file(book, fname, path))
        # Content-addressed ids, NOT sequential ones.  Pages keep arriving while the index
        # is being rebuilt, which changes how many chunks each chapter yields; a running
        # counter would silently re-point every later id at different text and misalign
        # any embeddings already stored against the old ids.
        used = {}
        with open(os.path.join(OUT, book + ".jsonl"), "w", encoding="utf-8") as f:
            for c in all_chunks:
                h = hashlib.sha1(c["text"].encode("utf-8")).hexdigest()[:16]
                cid = f"{book}-{h}"
                if cid in used:
                    used[cid] += 1
                    cid = f"{cid}-{used[cid]}"
                else:
                    used[cid] = 1
                c["id"] = cid
                c["embed_text"] = f"《{book}》{c['chapter']} {c['section']}\n{c['text']}"
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        total += len(all_chunks)
        summary[book] = len(all_chunks)
        print(f"[{book}] chunks={len(all_chunks)}", flush=True)
    print("TOTAL chunks:", total, summary, flush=True)


if __name__ == "__main__":
    main()
