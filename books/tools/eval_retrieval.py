# -*- coding: utf-8 -*-
"""Retrieval quality check over realistic 408 questions.

Spot checks hide systematic failures, so this runs a fixed question set and reports
whether a correct section appears in the top 1 / 3 / 5.  Each question names a keyword
that a genuinely relevant chunk must contain (in its body or in its heading).

    python books/tools/eval_retrieval.py
    python books/tools/eval_retrieval.py -v      # show the top hit for every question

Questions whose source chapter has not been OCR'd yet will fail; that is expected and
the script says so rather than pretending the retriever is broken.
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
rag = importlib.import_module("06_query")

# Two groups, because they exercise different halves of the system.
#
# LOOKUP questions name the topic, so the keyword and phrase lists can carry them.
# PARAPHRASE questions deliberately avoid the book's own wording - "换个说法问" - so they
# can only be answered by the vector side.  Without this group the semantic half of the
# hybrid retriever is never actually tested.
LOOKUP = [
    ("操作系统有哪几类用户接口",        "操作系统",       "用户接口"),
    ("什么是系统调用",                  "操作系统",       "系统调用"),
    ("进程和线程有什么区别",            "操作系统",       "线程"),
    ("死锁产生的四个必要条件",          "操作系统",       "死锁"),
    ("银行家算法是怎么避免死锁的",      "操作系统",       "银行家"),
    ("进程的三种基本状态",              "操作系统",       "状态"),
    ("时间片轮转调度算法",              "操作系统",       "时间片"),
    ("什么是管程",                      "操作系统",       "管程"),
    ("进程同步的互斥机制",              "操作系统",       "互斥"),
    ("数据结构的逻辑结构和存储结构",    "数据结构",       "逻辑结构"),
    ("算法的时间复杂度怎么计算",        "数据结构",       "时间复杂度"),
    ("顺序表和链表的区别",              "数据结构",       "链表"),
    ("栈和队列的区别",                  "数据结构",       "栈"),
    ("什么是循环队列",                  "数据结构",       "循环队列"),
    ("冯诺依曼机的主要特点",            "计算机组成原理", "诺依曼"),
    ("计算机的主要性能指标有哪些",      "计算机组成原理", "性能指标"),
    ("什么是摩尔定律",                  "计算机组成原理", "摩尔"),
    ("原码反码和补码",                  "计算机组成原理", "补码"),
    ("计算机网络由哪几部分组成",        "计算机网络",     "组成"),
    ("OSI参考模型分为哪几层",           "计算机网络",     "OSI"),
    ("什么是基带信号",                  "计算机网络",     "基带"),
    ("2019统考真题 哈夫曼编码",          "数据结构",       "2019"),
    ("2018统考真题 时钟中断",            "操作系统",       "2018"),
    ("2022统考真题 栈的入栈出栈序列",    "数据结构",       "2022"),
]

PARAPHRASE = [
    ("为什么进程切换比线程切换开销更大",  "操作系统",       "切换"),
    ("操作系统怎么避免死锁",              "操作系统",       "死锁"),
    ("一个算法好不好怎么判断",            "数据结构",       "复杂度"),
    ("数据在内存里是怎么存放的",          "计算机组成原理", "存储"),
    ("机器为什么能自动执行程序",          "计算机组成原理", "存储程序"),
    ("网络里两个主机通信要遵守什么约定",  "计算机网络",     "协议"),
    ("一堆进程抢一个资源怎么办",          "操作系统",       "互斥"),
    ("怎么衡量一台电脑快不快",            "计算机组成原理", "性能指标"),
]

CASES = [("lookup",) + c for c in LOOKUP] + [("paraphrase",) + c for c in PARAPHRASE]


def has(hits, kw):
    for h in hits:
        if kw in h["text"] or kw in h.get("chapter", "") or kw in h.get("section", ""):
            return True
    return False


def main():
    verbose = "-v" in sys.argv
    stats = {}          # group -> {1: n, 3: n, 5: n, n: n}
    warns, misses = 0, []
    vec_used = 0
    for group, q, book, kw in CASES:
        hits = rag.search(q, top=5, book=book)
        ok1, ok3, ok5 = has(hits[:1], kw), has(hits[:3], kw), has(hits[:5], kw)
        s = stats.setdefault(group, {1: 0, 3: 0, 5: 0, "n": 0})
        s["n"] += 1
        for n, ok in ((1, ok1), (3, ok3), (5, ok5)):
            s[n] += ok
        if hits and hits[0].get("vec_rank") is not None:
            vec_used += 1
        if rag.diagnose(q, hits):
            warns += 1
        mark = "ok  " if ok1 else ("ok@3" if ok3 else ("ok@5" if ok5 else "MISS"))
        if not ok5:
            misses.append((q, book, kw))
        print(f"  {mark:<5} {q:<30} [{book}]")
        if verbose and hits:
            print(f"         -> {rag.cite(hits[0])[:78]}")

    print()
    for group in ("lookup", "paraphrase"):
        s = stats.get(group)
        if not s:
            continue
        n = s["n"]
        print(f"{group:<11} n={n:<3} hit@1={s[1]}/{n}  hit@3={s[3]}/{n}  hit@5={s[5]}/{n}")
    tot, n = sum(stats[g][1] for g in stats), len(CASES)
    print(f"{'TOTAL':<11} n={n:<3} hit@1={tot}/{n}")
    print(f"top hit came from the vector list: {vec_used}/{n}")
    print(f"coverage warnings: {warns}  (topics whose chapter is probably not OCR'd yet)")
    if misses:
        print("\nno correct hit in top 5:")
        for q, book, kw in misses:
            print(f"   {q}  [{book}]  expected to contain {kw!r}")


if __name__ == "__main__":
    main()
