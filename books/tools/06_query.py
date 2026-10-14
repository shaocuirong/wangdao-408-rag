# -*- coding: utf-8 -*-
"""Stage 6 - hybrid retrieval over the four 王道 textbooks.

    python books/tools/06_query.py "银行家算法"
    python books/tools/06_query.py "TCP 三次握手" --book 计算机网络 --top 3
    python books/tools/06_query.py "B+树" --json

Also importable, which is how future sessions in this workspace should use it:

    import sys; sys.path.insert(0, r"E:/DE_workSpace/lesrn_1/books/tools")
    import importlib
    rag = importlib.import_module("06_query")
    hits = rag.search("进程调度算法", top=5)
"""
import os, sys, json, re, sqlite3, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jieba
jieba.setLogLevel(60)

ROOT  = r"E:\DE_workSpace\lesrn_1"
BOOKS = os.path.join(ROOT, "books")
RAGD  = os.path.join(BOOKS, "rag_data")
INDEX = os.path.join(RAGD, "index")
MODEL = os.path.join(RAGD, "models", "bge-base-zh-v1.5")

# bge-zh expects this instruction on the query side only, never on passages.
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
RRF_K = 60
POOL = 50
# Half of a rank-1 contribution: enough to lift the right past paper to the top when the
# question names a year, not enough to override a clearly better topical match.
YEAR_BONUS = 0.5 / (RRF_K + 1)

_model = None
_tok = None
_vecs = None
_ids = None


def _load_vectors():
    global _vecs, _ids
    if _vecs is not None:
        return True
    vp = os.path.join(INDEX, "vectors.npy")
    fp = os.path.join(INDEX, "fts.sqlite")
    if not os.path.exists(vp) or not os.path.exists(fp):
        return False
    import numpy as np
    _vecs = np.load(vp)
    ip = os.path.join(INDEX, "vectors.ids.json")
    if os.path.exists(ip):
        _ids = json.load(open(ip, encoding="utf-8"))
    else:
        con = sqlite3.connect(fp)
        _ids = [r[0] for r in con.execute("SELECT id FROM chunks ORDER BY rowid")]
        con.close()
    return _vecs.shape[0] == len(_ids)


def _load_model():
    global _model, _tok
    if _model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _model = AutoModel.from_pretrained(MODEL)
    _model.eval()


def _embed_query(text):
    import torch
    _load_model()
    enc = _tok([QUERY_PREFIX + text], padding=True, truncation=True,
               max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = _model(**enc).last_hidden_state[:, 0]
    out = torch.nn.functional.normalize(out, p=2, dim=1)
    return out.cpu().numpy().astype("float32")[0]


# Grammatical particles carry no retrieving power, and a plain OR over them makes long
# chunks win on raw term frequency alone.
STOPWORDS = set("""的 了 和 与 及 或 是 在 有 我 你 他 她 它 这 那 之 其 为 以 对 从 到 被 把
而 也 都 就 还 又 但 却 则 于 中 上 下 内 外 前 后 个 些 么 吗 呢 吧 啊 请 问 一下 什么 怎么
如何 为什么 哪些 哪个 介绍 说明 简述 解释 区别 联系 作用 特点 主要 一般 可以 需要
几层 几个 几种 几步 几类 几条 几项 哪几 哪几层 多少 是什么 分为 以及 下列 以下 关于 对于
更大 更小 更高 更低 更好 更快 更慢 更多 更少 更早 更晚 较大 较小 较大 很少 很多 一些 一样
重要 关键 常见 基本 简单 复杂 容易 困难 时候 情况 问题 内容 部分 方面 过程 方式 方法""".split())


def _quote(tok):
    return '"' + tok.replace('"', '""') + '"'


def _fts_search(query, book, limit):
    """Rank by rare terms first.

    A single OR over every token lets a long chunk that merely mentions the common words
    outrank the section actually about the topic.  So: drop particles, order the rest by
    document frequency, and try an AND over the rarest few, relaxing only if that is too
    narrow.  The full OR is kept as a recall backstop.
    """
    toks = [t for t in jieba.cut_for_search(query) if t.strip()]
    if not toks:
        return []
    content = [t for t in toks if t not in STOPWORDS and len(t) >= 2] or toks
    uniq, seen = [], set()
    for t in content:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    con = sqlite3.connect(os.path.join(INDEX, "fts.sqlite"))
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 1
    df = {}
    for t in uniq:
        try:
            df[t] = con.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                (_quote(t),)).fetchone()[0]
        except sqlite3.OperationalError:
            df[t] = total
    # Very common tokens are useless as filters; keep them only for the OR backstop.
    informative = sorted((t for t in uniq if df[t] <= 0.5 * total), key=lambda t: df[t])
    if not informative:
        informative = sorted(uniq, key=lambda t: df[t])

    base = ("SELECT c.rowid, c.id, c.book, c.chapter, c.section, "
            "c.page_start, c.page_end, c.text "
            "FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
            "WHERE chunks_fts MATCH ? ")
    tail = ("AND c.book = ? " if book else "") + "ORDER BY bm25(chunks_fts) LIMIT ?"

    def run(expr, n):
        args = [expr] + ([book] if book else []) + [n]
        try:
            return list(con.execute(base + tail, args))
        except sqlite3.OperationalError:
            return []

    out, seen_ids = [], set()
    for k in (3, 2, 1):
        if len(informative) < k:
            continue
        for r in run(" AND ".join(_quote(t) for t in informative[:k]), limit):
            if r[1] not in seen_ids:
                seen_ids.add(r[1])
                out.append(r)
        if len(out) >= limit:
            break
    if len(out) < limit:
        for r in run(" OR ".join(_quote(t) for t in uniq), limit):
            if r[1] not in seen_ids:
                seen_ids.add(r[1])
                out.append(r)
    con.close()
    return out[:limit]


def _vec_search(query, book, limit):
    import numpy as np
    if not _load_vectors():
        return []
    qv = _embed_query(query)
    sims = _vecs @ qv
    order = np.argsort(-sims)
    out = []
    con = sqlite3.connect(os.path.join(INDEX, "fts.sqlite"))
    for i in order:
        cid = _ids[int(i)]
        row = con.execute("SELECT id, book, chapter, section, page_start, page_end, text "
                          "FROM chunks WHERE id = ?", (cid,)).fetchone()
        if not row:
            continue
        if book and row[1] != book:
            continue
        out.append((int(i),) + row + (float(sims[int(i)]),))
        if len(out) >= limit:
            break
    con.close()
    return out


def _rare_term(query):
    """The single most discriminating query token, or None if none qualifies.

    When a learner names something specific - "银行家算法", "页面置换算法" - that rare
    word is the real question.  Chunks containing it are far better candidates than
    chunks that merely share the generic part ("算法").
    """
    _uniq, info, df, total = query_terms(query)
    if not info:
        return None
    t = info[0]
    if df.get(t, total) > 0.15 * total:
        return None
    return t


def _run_match(con, expr, book, pool):
    sql = ("SELECT c.rowid, c.id, c.book, c.chapter, c.section, "
           "c.page_start, c.page_end, c.text "
           "FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
           "WHERE chunks_fts MATCH ? ")
    args = [expr]
    if book:
        sql += "AND c.book = ? "
        args.append(book)
    sql += "ORDER BY bm25(chunks_fts) LIMIT ?"
    args.append(pool)
    try:
        return list(con.execute(sql, args))
    except sqlite3.OperationalError:
        return []


def _term_search(query, book, pool):
    """Chunks that really are about the question, ranked by BM25.

    Two shapes, in order of how specific they are:

    1. The whole query as a phrase.  Chinese compounds are often not single jieba
       tokens - "循环队列" indexes as 循环 + 队列 - so only a phrase query
       ("循环 队列", i.e. adjacent and in order) can tell the compound apart from a
       chunk that merely mentions both words somewhere.
    2. The single rarest token, which is what makes proper nouns such as
       "银行家算法" win.

    Both matter: RRF is rank-based, so without a list that keys on the exact topic, a
    chunk that is merely mediocre in all three lists can outscore the one chunk that is
    genuinely on topic.
    """
    con = sqlite3.connect(os.path.join(INDEX, "fts.sqlite"))
    try:
        uniq, _info, _df, _total = query_terms(query)
        if len(uniq) >= 2:
            phrase = '"' + " ".join(t.replace('"', '""') for t in uniq) + '"'
            rows = _run_match(con, phrase, book, pool)
            if rows:
                return rows
        t = _rare_term(query)
        if not t:
            return []
        return _run_match(con, _quote(t), book, pool)
    finally:
        con.close()


def search(query, top=5, book=None, pool=POOL):
    """Return the fused top hits with their citation metadata."""
    fts = _fts_search(query, book, pool)
    vec = _vec_search(query, book, pool)
    # A third ranked list for the discriminating term.  Fusing it alongside BM25 stops a
    # semantically-near-but-wrong chunk from winning on vector rank alone.
    key = _term_search(query, book, pool)

    # How much should the keyword lists be trusted?  They are precise when the question
    # names something specific, and close to noise when it is phrased loosely ("机器为什么
    # 能自动执行程序"); in the latter case the vector list is the only real signal, so the
    # keyword lists are down-weighted rather than allowed to bury it.
    kw_weight = 1.0 if (key or _rare_term(query)) else 0.5

    scores, payload = {}, {}
    for rank, r in enumerate(fts):
        cid = r[1]
        scores[cid] = scores.get(cid, 0.0) + kw_weight / (RRF_K + rank + 1)
        payload[cid] = {"id": cid, "book": r[2], "chapter": r[3], "section": r[4],  # noqa
                        "page_start": r[5], "page_end": r[6], "text": r[7],
                        "bm25_rank": rank + 1, "vec_rank": None}
    for rank, r in enumerate(vec):
        cid = r[1]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        payload.setdefault(cid, {"id": cid, "book": r[2], "chapter": r[3], "section": r[4],
                                 "page_start": r[5], "page_end": r[6], "text": r[7],
                                 "bm25_rank": None, "vec_rank": None})
        payload[cid]["vec_rank"] = rank + 1
        payload[cid]["vec_score"] = r[8]
    for rank, r in enumerate(key):
        cid = r[1]
        scores[cid] = scores.get(cid, 0.0) + kw_weight / (RRF_K + rank + 1)
        payload.setdefault(cid, {"id": cid, "book": r[2], "chapter": r[3], "section": r[4],
                                 "page_start": r[5], "page_end": r[6], "text": r[7],
                                 "bm25_rank": None, "vec_rank": None})
        payload[cid]["key_rank"] = rank + 1

    # If the question names an exam year, prefer chunks carrying that year's real-exam
    # marker.  The marker is matched explicitly ("【2019统考真题】" / "2019 统考真题")
    # rather than as a bare number, because hex addresses and constants are full of
    # four-digit runs - "首地址为2020FE00H" must not look like the 2020 paper.
    years = set(re.findall(r"(?:19|20)\d{2}", query))
    if years and payload:
        for cid, h in payload.items():
            flat = re.sub(r"\s+", "", h["text"])
            if any((y + "统考真题") in flat or ("【" + y) in flat for y in years):
                scores[cid] = scores.get(cid, 0.0) + YEAR_BONUS

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top]
    hits = []
    for cid, sc in ordered:
        h = payload[cid]
        h.setdefault("key_rank", None)
        h.setdefault("vec_score", None)
        h["score"] = round(sc, 6)
        hits.append(h)
    return hits


def query_terms(query):
    """Return (all_tokens, informative_tokens) for a query.

    "Informative" means the token occurs in at most half the corpus, so it can actually
    discriminate.  These are the words a relevant chunk ought to contain.
    """
    toks = [t for t in jieba.cut_for_search(query) if t.strip()]
    # Relax in stages.  Jumping straight to "use every token" - which is what a plain
    # "or toks" does - hands the stopwords back for queries like "B+树和B树的区别",
    # where B and 树 are single characters and everything else is a particle, so the
    # whole filter result is empty and 区别 ended up being treated as the key term.
    content = [t for t in toks if t not in STOPWORDS and len(t) >= 2]
    if not content:
        content = [t for t in toks if t not in STOPWORDS]
    if not content:
        content = toks
    uniq, seen = [], set()
    for t in content:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq or not os.path.exists(os.path.join(INDEX, "fts.sqlite")):
        return uniq, [], {}, 1
    con = sqlite3.connect(os.path.join(INDEX, "fts.sqlite"))
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 1
    df = {}
    for t in uniq:
        try:
            df[t] = con.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                                (_quote(t),)).fetchone()[0]
        except sqlite3.OperationalError:
            df[t] = total
    con.close()
    # df == 0 means the tokeniser split the phrase differently than the index did
    # (e.g. "冯诺依曼机" is stored as "冯·诺依曼机"), so such a term cannot discriminate.
    info = sorted((t for t in uniq if 0 < df[t] <= 0.5 * total), key=lambda t: df[t])
    return uniq, info, df, total


_NORM_RE = re.compile(r"[s　·・.,，。、；：:;()（）「」【】\-—_/？！?!'\"]+")


def _norm(s):
    """Strip punctuation so "冯诺依曼" and "冯·诺依曼" compare equal."""
    return _NORM_RE.sub("", s)


def _missing_msg(term):
    return ("检索结果可能不可靠：索引里没有任何段落包含「" + term +
            "」这个关键线索，说明该考点所在章节很可能尚未 OCR 完成。"
            "可以稍后再试，或运行 python books/tools/refresh_index.py 刷新索引。")


def diagnose(query, hits):
    """Warn when the answer is probably simply not indexed yet.

    Returning unrelated chapters for a topic that has not been OCR'd is worse than
    returning nothing: a student would study the wrong section.  If not a single
    discriminating query term occurs in any hit, say so instead.
    """
    uniq, info, df, total = query_terms(query)
    if not uniq:
        return None
    blob = _norm(" ".join(h["text"] + h.get("chapter", "") + h.get("section", "")
                          for h in hits))

    # Semantic similarity is the better coverage signal: it does not care whether the
    # learner used the book's wording.  Measured here, a question whose topic is present
    # peaks around 0.49-0.66 while a genuinely absent one peaks near 0.37.
    #
    # It is used only to SUPPRESS a warning, never to raise one.  vec_score belongs to
    # whichever chunks the vector list returned, and while the vector index is still
    # catching up it may not cover the chunk that actually answered the question - so a
    # low score can just mean "not embedded yet", not "topic missing".
    vs = [h["vec_score"] for h in hits if h.get("vec_score") is not None]
    if vs and max(vs) >= 0.46:
        return None

    # A term the tokeniser produced that the index has never seen is the clearest signal
    # that the topic is missing - but only if it is also absent from the hits after
    # punctuation is stripped, otherwise it is just a spelling variant of indexed text.
    missing = [t for t in uniq if df.get(t, 0) == 0 and len(t) >= 2 and _norm(t) not in blob]
    if missing:
        return ("检索结果可能不可靠：索引里没有任何段落包含「" + missing[0] +
                "」这个关键线索，说明该考点所在章节很可能尚未 OCR 完成。"
                "可以稍后再试，或运行 python books/tools/refresh_index.py 刷新索引。")

    if not info:
        return None
    rarest = info[0]
    # Only meaningful when the rarest term is genuinely rare: if every term is common,
    # their absence proves nothing.
    if df.get(rarest, total) > 0.15 * total:
        return None
    # Check the citation too: a chunk's own text often does not repeat its heading,
    # so a section literally titled "利用银行家算法避免死锁" would otherwise look
    # like it never mentions the term.
    if _norm(rarest) in blob:
        return None
    meta = os.path.join(INDEX, "meta.json")
    books = ""
    if os.path.exists(meta):
        m = json.load(open(meta, encoding="utf-8"))
        books = "；已索引 " + "、".join(f"{k} {v} 块" for k, v in m.get("per_book", {}).items() if v)
    return ("检索结果可能不可靠：索引里没有任何段落包含「" + rarest +
            "」这个关键线索，说明该考点所在章节很可能尚未 OCR 完成" + books +
            "。可以稍后再试，或运行 python books/tools/refresh_index.py 刷新索引。")


def cite(h):
    if h["page_start"] == h["page_end"]:
        loc = "p" + str(h["page_start"])
    else:
        loc = "p" + str(h["page_start"]) + "-" + str(h["page_end"])
    parts = [h["book"], h["chapter"]]
    if h["section"]:
        parts.append(h["section"])
    return " / ".join(p for p in parts if p) + " (" + loc + ")"


def main():
    ap = argparse.ArgumentParser(description="Hybrid retrieval over the 王道 2027 textbooks")
    ap.add_argument("query", nargs="?", default="",
                    help="question; omit to print the current index coverage")
    ap.add_argument("--book", default=None,
                    choices=["操作系统", "数据结构", "计算机组成原理", "计算机网络"])
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--chars", type=int, default=420)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if not (a.query or "").strip():
        # Report from the live database, not meta.json: the periodic refresher updates
        # the index far more often than meta.json is rewritten, so the file goes stale.
        fp = os.path.join(INDEX, "fts.sqlite")
        if not os.path.exists(fp):
            print("index not built yet - run python books/tools/05_index.py")
            return
        con = sqlite3.connect(fp)
        rows = con.execute("SELECT book, COUNT(*) FROM chunks GROUP BY book "
                           "ORDER BY COUNT(*) DESC").fetchall()
        total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        vp = os.path.join(INDEX, "vectors.npy")
        nvec = 0
        if os.path.exists(vp) and _load_vectors():
            nvec = int(_vecs.shape[0])
        print(f"知识块总数 : {total}")
        for b, n in rows:
            print(f"   {b:<10} {n:>5} 块" + ("   (尚未收录)" if n == 0 else ""))
        print(f"向量覆盖   : {nvec} 块" + ("  <- 关键词检索仍全量可用" if nvec < total else ""))
        cov = {b: n for b, n in rows}
        missing = [b for b in ("操作系统", "数据结构", "计算机组成原理", "计算机网络")
                   if cov.get(b, 0) == 0]
        if missing:
            print("尚未收录   : " + "、".join(missing) + "（OCR 可能还没跑到）")
        return

    hits = search(a.query, top=a.top, book=a.book)
    if a.json:
        print(json.dumps(hits, ensure_ascii=False, indent=1))
        return
    warn = diagnose(a.query, hits)
    if not hits:
        print("no hits (is the index built? run 05_index.py)")
        return
    if warn:
        print()
        print("[!] " + warn)
    for i, h in enumerate(hits, 1):
        print()
        print("[" + str(i) + "] " + cite(h) + "   score=" + str(h["score"])
              + "  (bm25#" + str(h["bm25_rank"]) + " vec#" + str(h["vec_rank"])
              + " key#" + str(h.get("key_rank")) + ")")
        body = h["text"].replace("\n", " ")
        print("    " + (body[:a.chars] + ("..." if len(body) > a.chars else "")))


if __name__ == "__main__":
    main()
