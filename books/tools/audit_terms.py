# -*- coding: utf-8 -*-
"""What does the rarity rule actually pick as the discriminating term?

It is supposed to lock onto the topic ("银行家算法"), but frequency alone does not imply
usefulness - it picked "更大" once and derailed the whole result list.  This prints the
chosen term for a spread of questions so junk words are visible instead of latent.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
rag = importlib.import_module("06_query")

QUERIES = [
    "什么是系统调用", "银行家算法是怎么避免死锁的", "进程和线程有什么区别",
    "什么是循环队列", "OSI参考模型分为哪几层", "什么是基带信号",
    "为什么进程切换比线程切换开销更大", "一个算法好不好怎么判断",
    "机器为什么能自动执行程序", "数据在内存里是怎么存放的",
    "网络里两个主机通信要遵守什么约定", "一堆进程抢一个资源怎么办",
    "怎么衡量一台电脑快不快", "操作系统怎么避免死锁",
    "2019统考真题 哈夫曼编码", "CSMA/CD协议的争用期", "B+树和B树的区别",
    "LRU页面置换算法", "IEEE754浮点数的表示", "TCP三次握手的过程",
    "Cache和主存的映射方式", "浮点数为什么不精确", "为什么要用虚拟内存",
    "中断和异常有什么不一样", "IP地址和MAC地址的区别",
]
for q in QUERIES:
    u, info, df, total = rag.query_terms(q)
    rt = rag._rare_term(q)
    rank = [(t, df[t]) for t in info[:3]]
    print(f"  rare={str(rt):<8} df={df.get(rt, '-'):<5} top3={rank}   <- {q}")
