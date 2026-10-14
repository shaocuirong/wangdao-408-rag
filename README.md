# 王道 2027 考研 408 本地 RAG

把四本 2027 王道考研复习指导（操作系统 / 数据结构 / 计算机组成原理 / 计算机网络）
做成一个**本地混合检索知识库**：问一个问题，返回带**书名 / 章节 / 小节 / 原书页码**的原文段落。

面向 2027 考研 083500（408 专业课）。全程本地运行，不联网、不需要 API key。

```powershell
python books\tools\06_query.py "银行家算法怎么避免死锁"
```

```
[1] 操作系统 / 第2章进程与线程 / 2.4死锁 > 2.4.3死锁避免 > 2．利用银行家算法避免死锁 (p164)
    (bm25#1 vec#1 key#1)
    2．利用银行家算法避免死锁 考点追踪银行家算法的特点（2013、2019) 银行家算法是最著名的死锁避免算法……
```

引用的页码是**书上印的页码**，不是 PDF 页码，可以直接翻书核对。

---

## 为什么要做这个

这四本 PDF 是**纯图片扫描件，完全没有文字层**——1432 页逐页抽取文本都是 0 字符。
所以不能直接搜索，必须先 OCR。同时 `计算机组成原理` 那一本**连书签都没有**，
章节结构只能从正文重建。

做完之后的效果：`books/summary_books/` 下是可直接阅读的全文 Markdown，
`books/rag_data/` 下是检索索引。

## 快速开始

需要 Python 3.10+，依赖：`numpy pandas jieba pypdfium2 rapidocr-onnxruntime torch transformers`

```powershell
pip install rapidocr-onnxruntime pypdfium2 jieba torch transformers

# 1. 下载中文嵌入模型（约 390 MB）
python books\tools\00_get_model.py

# 2. 把四本 PDF 放回 books\ 目录（见下一节"数据来源"）

# 3. 全流程：OCR → Markdown → 切块 → 建索引
python books\tools\00_build_all.py
```

如果只想用现成的索引（仓库里已经带了 OCR 结果、切块和索引）：

```powershell
python books\tools\06_query.py "你的问题"
python books\tools\06_query.py "TCP三次握手" --book 计算机网络 --top 3
python books\tools\06_query.py "B+树" --json
```

也可以在 Python 里直接用：

```python
import sys; sys.path.insert(0, r"<仓库路径>/books/tools")
import importlib
rag = importlib.import_module("06_query")
for h in rag.search("进程和线程的区别", top=5):
    print(rag.cite(h), h["text"][:200])
```

## 数据来源（重要）

仓库里**不包含**这两样东西，需要自己准备：

| 缺什么 | 大小 | 怎么办 |
|---|---|---|
| 四本原始 PDF | 815 MB | 自行获取，放进 `books/`，文件名见 `books/tools/01_ocr.py` 里的 `BOOKS_MAP` |
| 嵌入模型 bge-base-zh-v1.5 | 390 MB | 跑 `python books\tools\00_get_model.py` 自动下载 |

原因：PDF 单本 188–225 MB、模型 390 MB，都超过 GitHub 的 100 MB 单文件上限，
只能走 Git LFS，而免费额度（1 GB）装不下。

**其余全部数据都在仓库里**，克隆下来就能查：OCR 逐页结果、章节结构、4619 个知识块、
已建好的索引（关键词 + 向量）。

## 仓库结构

```
AGENTS.md                      给 AI 助手的工作区约定：408 问题必须先查本地 RAG
books/
  tools/                       全部脚本（见 tools/README.md，含完整踩坑记录）
    00_build_all.py            一键全流程
    01_ocr.py                  OCR，可断点续跑
    02_extract_toc.py          提取 PDF 书签
    03_build_markdown.py       版面还原 + 章节挂载
    04_chunk.py                按章节切块
    05_index.py                建索引（FTS5 + 向量）
    06_query.py                检索入口
    booklayout.py              版面还原库
    refresh_index.py           中途刷新索引
    audit_*.py / eval_retrieval.py   五道体检
  summary_books/               转换出的 Markdown（37 个章节文件 + 4 个整本合并）
  rag_data/
    ocr/                       逐页 OCR 结果（含坐标）
    structure/                 书签树 / 章节树
    chunks/                    知识块（带书名/章节/页码）
    index/                     fts.sqlite + vectors.npy
```

## 检索是怎么做的

三路结果用 RRF（Reciprocal Rank Fusion）融合：

1. **关键词**：SQLite FTS5 + jieba 分词，BM25 排序。
2. **向量**：bge-base-zh-v1.5（768 维），余弦相似度——换个说法提问靠它。
3. **关键线索**：整句短语查询（"循环 队列"，要求相邻同序），
   或最稀有的那个词（"银行家算法"这种专有名词靠它）。

三路缺一不可，原因都写在 `books/tools/README.md` 里，几个反直觉的点：

- **纯 OR 检索是错的**。"进程"出现在 63% 的块里，把所有词 OR 起来会让"提到过这些词的长段落"
  压过"真正讲这个知识点的段落"。
- **R RF 只看名次**，一个在三路里都平庸的块能压过只在两路里拔尖的那个块。
- **低频 ≠ 有信息量**。按词频挑关键词会选中"更大""不好"这种又少见又没用的虚词。
- **中文合成词常常不是 jieba 的一个词**。"循环队列"在索引里是 `循环` + `队列`，
  只有短语查询能把它和"碰巧两个词都出现过"区分开。

问"哪一年的真题"时会优先给那一年（例如 `2019统考真题 哈夫曼`），
但年份必须按"真题标记"匹配——组成原理里满是 `2020FE00H` 这种十六进制地址，
裸数字匹配会让它白拿加权。

## 检索质量（实测）

`books/tools/eval_retrieval.py` 跑 32 道题：

```
lookup      n=24  hit@1=24/24  hit@3=24/24  hit@5=24/24
paraphrase  n=8   hit@1=5/8    hit@3=6/8    hit@5=7/8
TOTAL       n=32  hit@1=29/32
```

- **lookup**：照书上的说法问（"什么是系统调用"）——**24 题全部第一条命中**。
- **paraphrase**：完全换成大白话问（"机器为什么能自动执行程序"）——只有 5/8 排到第一，
  但 7/8 仍在前五。**所以查不到时换用书上的原词再问一次，这是实测差异，不是客套话。**

剩下那一题（"机器为什么能自动执行程序"）是**嵌入模型的能力上限**，不是流水线问题：
实测正确答案那块已经算过向量、相似度只有 0.380，而同一章里三块无关内容拿到 0.438。
要根治只能换更强的模型。

## 质量检查

库里有五道互补的体检，跑任意一个都能看出问题：

| 脚本 | 查什么 |
|---|---|
| `verify_structure.py` | 每本书的 OCR 进度、目录来源、章节是否齐全 |
| `audit_anchors.py` | 两个书签是否抢同一行（会导致"某节空、邻近节却有两倍正文"） |
| `audit_loss.py` | 识别出的文字有多少没进 Markdown（**看"少了什么"**） |
| `audit_ocr.py` | 哪些页几乎没识别出文字（不会报错，但等于从检索里消失） |
| `eval_retrieval.py` | 检索质量 hit@1/3/5 |

当前状态：五道全绿。识别文字 **96–99%** 进入 Markdown，1432 页无空白页，
章节数 5 / 8 / 7 / 6 与书上一致。

## 免责声明

`summary_books/` 与 `rag_data/` 中是王道考研复习指导的**转换后全文**，
版权归原作者及出版方所有。本仓库仅供**个人学习检索**使用，请勿用于分发或商业用途。
建议配合自己购买的正版图书使用。

## 许可

代码部分可自由使用。教材内容版权见上。
