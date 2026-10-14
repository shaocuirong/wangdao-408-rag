# -*- coding: utf-8 -*-
"""Stage 5 - build the hybrid index: SQLite FTS5 (BM25) + bge-base-zh vectors.

Why hybrid: exam questions are full of exact terms ("银行家算法", "CSMA/CD", "B+树")
that a dense encoder blurs, and of paraphrases that keyword search misses.  FTS5 with
jieba tokens covers the first, the Chinese BGE model covers the second, and the two are
fused at query time with Reciprocal Rank Fusion.

The FTS column stores jieba tokens rather than raw text: FTS5's unicode61 tokenizer has
no notion of Chinese word boundaries, so pre-segmenting is what makes CJK search work.
"""
import os, sys, json, sqlite3, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.stdout.reconfigure(encoding="utf-8")

import jieba
jieba.setLogLevel(60)

ROOT  = r"E:\DE_workSpace\lesrn_1"
BOOKS = os.path.join(ROOT, "books")
RAGD  = os.path.join(BOOKS, "rag_data")
CHUNKS = os.path.join(RAGD, "chunks")
INDEX  = os.path.join(RAGD, "index")
MODEL  = os.path.join(RAGD, "models", "bge-base-zh-v1.5")

BOOK_KEYS = ["操作系统", "数据结构", "计算机组成原理", "计算机网络"]
BATCH = 16


def tokenize(text):
    return " ".join(t for t in jieba.cut_for_search(text) if t.strip())


def load_chunks():
    rows = []
    for book in BOOK_KEYS:
        path = os.path.join(CHUNKS, book + ".jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- FTS5

def sweep_stale_temps(path):
    """Delete temp files abandoned by a killed build.

    Temp names carry a pid so concurrent builds cannot collide, but a build that is
    killed cannot clean up after itself - so drop anything left over from an earlier,
    definitely-finished process (an hour is well beyond any real build).
    """
    d, base = os.path.dirname(path) or ".", os.path.basename(path)
    cutoff = time.time() - 3600
    for f in os.listdir(d):
        if f.startswith(base + ".") and ".tmp" in f:
            fp = os.path.join(d, f)
            try:
                if os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
                    print(f"  [clean] removed stale {f}", flush=True)
            except OSError:
                pass


def build_fts(rows, path):
    sweep_stale_temps(path)
    # Build beside the live file and swap it in, so a query running concurrently never
    # sees a half-written index.  The temp name carries the pid because the pipeline and
    # the periodic refresher can both be rebuilding at the same moment - a shared temp
    # path would let one process replace the file while the other is still writing it.
    tmp = f"{path}.{os.getpid()}.tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute("""CREATE TABLE chunks(
        id TEXT PRIMARY KEY, book TEXT, chapter TEXT, section TEXT,
        page_start INTEGER, page_end INTEGER, text TEXT)""")
    con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body, content='')")
    # Both tables MUST be written with the same explicit rowid.  chunks_fts is keyed by
    # rowid and joined to chunks on it, but a plain rowid table starts numbering at 1
    # while we were seeding the FTS table from 0 - which silently paired every keyword
    # hit with the *previous* chunk's text and citation.
    for i, c in enumerate(rows):
        pages = c.get("pdf_pages") or [None, None]
        con.execute("INSERT INTO chunks(rowid, id, book, chapter, section, "
                    "page_start, page_end, text) VALUES (?,?,?,?,?,?,?,?)",
                    (i, c["id"], c["book"], c["chapter"], c["section"],
                     pages[0], pages[1], c["text"]))
        con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (?,?)",
                    (i, tokenize(c["text"])))
    # Guard: a mismatch here means every join is off by a constant, which is invisible
    # in spot checks because neighbouring chunks usually share a section.
    a = con.execute("SELECT MIN(rowid), MAX(rowid) FROM chunks").fetchone()
    b = con.execute("SELECT MIN(rowid), MAX(rowid) FROM chunks_fts").fetchone()
    if a != b:
        raise RuntimeError(f"rowid mismatch between chunks {a} and chunks_fts {b}")
    con.commit()
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    con.commit()
    con.close()
    os.replace(tmp, path)


# ---------------------------------------------------------------- vectors

def build_vectors(rows, path, ids_path):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()

    # Reuse vectors we already have.  Chunk ids are content hashes, so an id that is
    # still present describes exactly the same text as before - re-embedding it would
    # burn minutes of CPU to reproduce an identical vector.  This is what lets the
    # periodic refresher keep the vector side current instead of only the keyword side.
    reused = {}
    if os.path.exists(path) and os.path.exists(ids_path):
        try:
            old_mat = np.load(path)
            old_ids = json.load(open(ids_path, encoding="utf-8"))
            if old_mat.shape[0] == len(old_ids):
                reused = {cid: old_mat[i] for i, cid in enumerate(old_ids)}
        except Exception as e:
            print(f"  [vec] ignoring unusable previous index: {e}", flush=True)

    todo = [i for i, c in enumerate(rows) if c["id"] not in reused]
    print(f"  [vec] {len(rows)} chunks: {len(rows)-len(todo)} reused, {len(todo)} to embed",
          flush=True)

    fresh = {}
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        idxs = todo[i:i + BATCH]
        enc = tok([rows[j]["embed_text"] for j in idxs], padding=True, truncation=True,
                  max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc).last_hidden_state[:, 0]          # BGE uses CLS pooling
        out = torch.nn.functional.normalize(out, p=2, dim=1)     # cosine == dot product
        arr = out.cpu().numpy().astype("float32")
        for k, j in enumerate(idxs):
            fresh[rows[j]["id"]] = arr[k]
        if (i // BATCH) % 10 == 0 and todo:
            done = min(i + BATCH, len(todo))
            el = time.time() - t0
            print(f"  [vec] {done}/{len(todo)} {done/el:.1f}/s", flush=True)

    dim = 768
    for v in list(reused.values())[:1] + list(fresh.values())[:1]:
        dim = int(v.shape[0])
        break
    if rows:
        mat = np.vstack([reused.get(c["id"], fresh.get(c["id"], np.zeros(dim, "float32")))
                         for c in rows]).astype("float32")
    else:
        mat = np.zeros((0, dim), dtype="float32")
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, mat)
    os.replace(tmp, path)
    # Record which chunk each row describes, so a later (larger) chunk set does not
    # invalidate the vectors - they simply cover a subset until the next full rebuild.
    tmpj = f"{ids_path}.{os.getpid()}.tmp"
    with open(tmpj, "w", encoding="utf-8") as f:
        json.dump([c["id"] for c in rows], f)
    os.replace(tmpj, ids_path)
    return mat.shape


def main():
    os.makedirs(INDEX, exist_ok=True)
    fts_only = "--fts-only" in sys.argv
    rows = load_chunks()
    if not rows:
        print("no chunks found - run 04_chunk.py first")
        return
    print(f"chunks: {len(rows)}", flush=True)

    fts = os.path.join(INDEX, "fts.sqlite")
    build_fts(rows, fts)
    print(f"fts built -> {fts} ({os.path.getsize(fts)/1048576:.1f} MB)", flush=True)

    vec_path = os.path.join(INDEX, "vectors.npy")
    if fts_only:
        # Keyword side only.  Re-embedding everything is minutes of CPU, which is not
        # worth stealing from a running OCR pass; the vector side is refreshed at the end.
        have = os.path.exists(vec_path)
        print(f"vectors skipped (--fts-only); existing={'yes' if have else 'no'}", flush=True)
        shape = None
    else:
        shape = build_vectors(rows, vec_path, os.path.join(INDEX, "vectors.ids.json"))
        print(f"vectors built -> {vec_path} {shape}", flush=True)

    metap = os.path.join(INDEX, "meta.json")
    with open(f"{metap}.{os.getpid()}.tmp", "w", encoding="utf-8") as f:
        json.dump({
            "count": len(rows),
            "per_book": {b: sum(1 for c in rows if c["book"] == b) for b in BOOK_KEYS},
            "model": os.path.basename(MODEL),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=1)
    os.replace(f"{metap}.{os.getpid()}.tmp", metap)
    print("done", flush=True)


if __name__ == "__main__":
    main()
