# -*- coding: utf-8 -*-
"""Run the whole pipeline: OCR -> Markdown -> chunks -> index.

Every stage is resumable, so this is safe to re-run after an interruption.
Stage 1 (OCR) is the only slow one: ~1432 scanned pages at ~0.13 pages/s on this
machine, i.e. two to three hours.  The later stages take a couple of minutes.

    python books/tools/00_build_all.py            # everything
    python books/tools/00_build_all.py --skip-ocr # content already extracted
"""
import os, sys, subprocess, time
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))

# Held only while the content stages run, never during the hours-long OCR pass - the
# refresher is welcome to keep the index current in the meantime.  It exists because
# 03_build_markdown.py deletes and rewrites the chapter files: if the refresher did that
# while this pipeline was reading them, the final index would be built from a partial
# chapter set, and index_is_current() could not tell, because the chunk files would be
# equally partial and the two counts would agree.
LOCK = os.path.join(os.path.dirname(HERE), "rag_data", ".build.lock")
STAGES = [
    ("01_ocr.py",            "OCR all pages"),
    ("03_build_markdown.py", "rebuild Markdown + structure"),
    ("04_chunk.py",          "chunk with chapter/page metadata"),
    ("05_index.py",          "build FTS5 + vector index"),
]


def run(script):
    print(f"\n===== {script} =====", flush=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
    print(f"===== {script} exit={p.returncode} in {time.time()-t0:.1f}s =====", flush=True)
    return p.returncode


def index_is_current():
    """Does the built index describe exactly the chunks currently on disk?

    The periodic refresher (refresh_index.py) rewrites the same Markdown and chunk files
    this pipeline reads.  If the two ever interleave, the final index could be built from
    a partially rewritten chapter set and nothing afterwards would notice.  So verify,
    and rebuild once if the counts disagree.
    """
    import json, glob
    chunk_dir = os.path.join(os.path.dirname(HERE), "rag_data", "chunks")
    meta_p = os.path.join(os.path.dirname(HERE), "rag_data", "index", "meta.json")
    if not os.path.exists(meta_p):
        return False
    total = 0
    for p in glob.glob(os.path.join(chunk_dir, "*.jsonl")):
        with open(p, encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    try:
        count = json.load(open(meta_p, encoding="utf-8"))["count"]
    except Exception:
        return False
    print(f"[verify] chunks on disk={total} index={count}", flush=True)
    return total == count


def main():
    skip = "--skip-ocr" in sys.argv
    bad = []
    for script, label in STAGES:
        if skip and script == "01_ocr.py":
            print(f"[skip] {label}", flush=True)
            continue
        if script == "03_build_markdown.py":
            os.makedirs(os.path.dirname(LOCK), exist_ok=True)
            with open(LOCK, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
        try:
            if run(script) != 0:
                bad.append(script)
                print(f"[warn] {script} failed; continuing", flush=True)
        finally:
            if script == "05_index.py" and os.path.exists(LOCK):
                os.remove(LOCK)

    if not index_is_current():
        print("[verify] index does not match the chunk files - rebuilding", flush=True)
        for script in ("03_build_markdown.py", "04_chunk.py", "05_index.py"):
            if run(script) != 0:
                bad.append(script + " (retry)")
    else:
        print("[verify] index is consistent with the chunk files", flush=True)
    print("\n================ SUMMARY ================", flush=True)
    print("failures:", bad or "none", flush=True)
    meta = os.path.join(os.path.dirname(HERE), "rag_data", "index", "meta.json")
    if os.path.exists(meta):
        print(open(meta, encoding="utf-8").read(), flush=True)


if __name__ == "__main__":
    main()
