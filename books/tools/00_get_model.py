# -*- coding: utf-8 -*-
"""Fetch the Chinese embedding model used by the vector index.

huggingface.co is reachable directly from this machine; the hf-mirror endpoint is not
(it answers the hub API with redirects the client cannot follow).  So we try the Hub
first and fall back to ModelScope.
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
os.environ.pop("HF_ENDPOINT", None)

ROOT = r"E:\DE_workSpace\lesrn_1"
TGT = os.path.join(ROOT, "books", "rag_data", "models")
os.makedirs(TGT, exist_ok=True)

REPOS = ["BAAI/bge-base-zh-v1.5"]
PATTERNS = ["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"]


def via_hub(repo, dest):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=PATTERNS, max_workers=4)


def via_modelscope(repo, dest):
    from modelscope.hub.snapshot_download import snapshot_download as ms_download
    return ms_download(repo, cache_dir=dest)


for repo in REPOS:
    dest = os.path.join(TGT, repo.split("/")[-1])
    if os.path.isdir(dest) and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(dest)):
        print("already present:", dest, flush=True)
        continue
    for name, fn in (("huggingface", via_hub), ("modelscope", via_modelscope)):
        try:
            print(f"downloading {repo} via {name} ...", flush=True)
            p = fn(repo, dest)
            print("OK ->", p, flush=True)
            break
        except Exception as e:
            print(f"  {name} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
    else:
        print("ALL SOURCES FAILED for", repo, flush=True)
        continue

for repo in REPOS:
    dest = os.path.join(TGT, repo.split("/")[-1])
    if not os.path.isdir(dest):
        continue
    print("---", dest, flush=True)
    total = 0
    for f in sorted(os.listdir(dest)):
        fp = os.path.join(dest, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            total += sz
            print(f"   {f}  {sz/1048576:.1f} MB", flush=True)
    print(f"   TOTAL {total/1048576:.1f} MB", flush=True)
