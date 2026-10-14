# -*- coding: utf-8 -*-
"""Rebuild Markdown -> chunks -> index from whatever OCR has produced so far.

The full pipeline (00_build_all.py) only refreshes the index once, at the very end of a
multi-hour OCR run.  This helper refreshes it mid-run so the RAG stays usable and keeps
getting better while pages are still being extracted.

    python books/tools/refresh_index.py                  # refresh once
    python books/tools/refresh_index.py --every 30 --times 6   # then every 30 min, 6x
"""
import os, sys, subprocess, time, argparse
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["03_build_markdown.py", "04_chunk.py", "05_index.py"]


LOCK = os.path.join(os.path.dirname(HERE), "rag_data", ".build.lock")


def pipeline_busy():
    """True while 00_build_all.py is rewriting the shared content files.

    Running a refresh then would race it - 03_build_markdown.py deletes and rewrites the
    chapter files, so a concurrent reader can see a partial set.
    """
    try:
        if not os.path.exists(LOCK):
            return False
        if time.time() - os.path.getmtime(LOCK) > 1800:      # stale lock, ignore
            return False
        return True
    except OSError:
        return False


def refresh_once(fts_only=False):
    t0 = time.time()
    for s in STAGES:
        cmd = [sys.executable, os.path.join(HERE, s)]
        if s == "05_index.py" and fts_only:
            cmd.append("--fts-only")
        p = subprocess.run(cmd, cwd=HERE)
        if p.returncode != 0:
            print(f"[warn] {s} exit={p.returncode}", flush=True)
    # Report what the index now covers, per book.
    import json
    meta = os.path.join(os.path.dirname(HERE), "rag_data", "index", "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta, encoding="utf-8"))
        print(f"[refresh] {time.strftime('%H:%M:%S')} chunks={m['count']} "
              f"{m['per_book']} in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=0, help="minutes between refreshes; 0 = once")
    ap.add_argument("--times", type=int, default=1)
    ap.add_argument("--fts-only", action="store_true",
                    help="skip re-embedding (cheap, keeps keyword search current)")
    a = ap.parse_args()
    for i in range(a.times):
        if pipeline_busy():
            print(f"[skip] {time.strftime('%H:%M:%S')} full pipeline is rebuilding; "
                  f"leaving it alone", flush=True)
        else:
            refresh_once(fts_only=a.fts_only)
        if a.every and i + 1 < a.times:
            time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
