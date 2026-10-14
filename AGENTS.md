# 本工作区约定：408 专业课内容一律先查本地 RAG

本工作区（`E:\DE_workSpace\lesrn_1`）是一名 **2027 考研 083500 专业**考生的备考工作区。
`books\` 下放着四本 2027 王道考研复习指导，已建成**本地混合检索（RAG）索引**。

> 想知道索引现在覆盖到哪了，看 `books\rag_data\index\meta.json` 里的 `per_book`。
> 若某本书查不到内容，多半是该书 OCR 尚未跑完，
> 跑 `python books\tools\refresh_index.py` 即可把已有内容刷进索引。

## 硬性要求

**只要问题涉及 408 专业课（数据结构 / 计算机组成原理 / 操作系统 / 计算机网络），
必须先检索本地 RAG，再作答。** 不要凭记忆直接回答专业课问题，也不要用网络搜索代替它。

理由：这四本书就是本工作区的权威依据，检索结果能给出**书名 / 章节 / 页码**，
可以逐条对照原文，避免记错概念和公式。

## 怎么查

命令行（推荐，最省事）：

```powershell
cd E:\DE_workSpace\lesrn_1
python books\tools\06_query.py "银行家算法"
python books\tools\06_query.py "TCP 三次握手" --book 计算机网络 --top 3
python books\tools\06_query.py "B+树" --json          # 结构化输出
python books\tools\06_query.py "页面置换算法" --chars 800   # 看更多正文
```

在 Python 里直接用：

```python
import sys; sys.path.insert(0, r"E:\DE_workSpace\lesrn_1\books\tools")
import importlib
rag = importlib.import_module("06_query")
for h in rag.search("进程和线程的区别", top=5):
    print(rag.cite(h), h["text"][:200])
```

每条结果形如：

```
操作系统 / 第3章内存管理 / 3.1内存管理概念 > 3.1.2 覆盖与交换 (p187-188)
```

即 **书名 / 章节 / 小节 / 原书页码**，可直接引用；页码是**书上印的页码**，不是 PDF 页码。

### 看到「检索结果可能不可靠」时怎么办

如果输出里出现这一行：

> [!] 检索结果可能不可靠：索引里没有任何段落包含「×××」这个关键线索……

意思是**这个考点所在的章节还没 OCR 完**，下面的结果只是碰巧沾边的段落，**不要拿它当答案**。
正确做法是：直接说明这部分内容还没建好，或者先运行 `python books\tools\refresh_index.py` 刷新再查。
这条提示是刻意加的——返回一个无关章节比返回"没找到"更糟，考生会照着错的地方复习。

## 回答时的引用格式

引用检索到的内容时，写成：
`《操作系统》第3章 3.1.2（p187）` —— 便于回书核对。

## 索引内容范围

| 书名 | 页数 | 结构来源 |
|------|------|----------|
| 操作系统 | 372 | PDF 书签（435 条） |
| 数据结构 | 404 | PDF 书签（478 条） |
| 计算机组成原理 | 340 | 无书签，由正文标题自动重建 |
| 计算机网络 | 316 | PDF 书签（419 条） |

## 目录结构

```
books\
  *.pdf                     原始扫描件（无文字层，纯图片）
  summary_books\            转换出的 Markdown（按章一个文件，可读可改）
    <书名>\NN_<章名>.md
    <书名>.md               整本合并
  tools\                    全部脚本，见 tools\README.md
  rag_data\
    ocr\                    逐页 OCR 结果（含坐标）
    structure\              书签树 / 章节树
    chunks\                 切分后的知识块
    index\                  fts.sqlite（BM25）+ vectors.npy（向量）+ meta.json
    models\bge-base-zh-v1.5 中文嵌入模型
```

## 重建索引

```powershell
python books\tools\00_build_all.py          # OCR → Markdown → 切块 → 建索引，全流程
```

单步（正常不需要）：

```powershell
python books\tools\01_ocr.py               # 可断点续跑
python books\tools\03_build_markdown.py
python books\tools\04_chunk.py
python books\tools\05_index.py
```

### OCR 还没跑完时，想让索引跟上进度

```powershell
python books\tools\refresh_index.py --fts-only   # 只重建关键词索引，几秒钟
python books\tools\refresh_index.py             # 连向量一起重建，几分钟
```

索引是**原子替换**的，所以边重建边查询不会读到半个索引。

> 注意：OCR 是纯 CPU 密集型，1432 页在这台机器上约需 2–3 小时，且会明显占用 CPU。
> 除非书籍有更新，否则**不要重复跑 OCR**——索引已建好，直接用 `06_query.py` 即可。

## 检索原理（便于排查）

- **关键词侧**：SQLite FTS5 + jieba 分词，BM25 排序；专有名词（银行家算法、CSMA/CD、B+树）靠它命中。
  查询不是简单地把所有词 OR 起来——"进程"这类高频词在 63% 的块里都出现，
  纯 OR 会让"提到过这些词的长段落"压过"真正讲这个知识点的段落"。
  所以实现里先去掉虚词，再按**词频从低到高**用最稀有的几个词做 AND，命中不够才逐级放宽。
- **语义侧**：bge-base-zh-v1.5（768 维）向量，余弦相似度；换个说法提问靠它命中。
- **融合**：三路结果用 RRF（Reciprocal Rank Fusion）合并——关键词一路、向量一路，
  再加一路「包含那个最稀有查询词」的结果。第三路很关键：问「银行家算法」时，
  "算法效率的度量"在向量侧排第 1、关键词侧排第 17，两路相加本会压过真正讲银行家算法的段落。
- 因此出现"没查到"时，先换用书上的**原词**再试一次。
  实测（`books\tools\eval_retrieval.py`，29 题）：**照书上的词问，21 题全部第一条命中；
  完全换成大白话问，只有 4/8 排到第一**（但 7/8 仍在前五）。这个差距是实测的，不是估计。
