# longbooklib

[English](README.md) | [日本語](README.ja.md) | **中文**

为 [LongBook 系列模型](https://huggingface.co/collections/summerAIO/longbook-series)打造的智能外部记忆库。
LongBook 使用固定大小的循环状态处理长文本，显存占用很低，但远处文本中的专有名词、数字、引文等精确细节容易丢失。
`longbooklib` 把文本存放在模型之外，针对每个问题只检索相关内容并放入提示词，以此弥补这一不足。

```
 长文本 ──► 分块 ──► 向量 + BM25 索引 ──► SQLite（持久化）
                                              │
 问题 ──► 混合检索 ──► 合并相邻块 ──► 受 token 预算限制的笔记 ──► LongBook ──► 回答
              ▲                                                       │
              └──────── <recall>新的检索词</recall>（模型要求再次检索） ◄──┘
```

刻意保持精简：只有 4 个模块，唯一的硬依赖是 `numpy`，不需要向量数据库服务。

## 安装

```bash
pip install -e .                 # 核心（仅 numpy），不装 torch 也能使用 Memory
pip install -e ".[hf]"           # + transformers/torch（运行 LongBook 模型）
pip install -e ".[semantic]"     # + sentence-transformers（多语言语义检索）
```

## 快速开始

```python
from longbooklib import LongBookModel, Memory

mem = Memory("novel.db")                       # 持久化；Memory() 则只保存在内存中
lb = LongBookModel.from_pretrained("summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2", mem)

lb.remember(open("novel.txt", encoding="utf-8").read(), source="novel")   # 只需一次，重复添加不会产生重复内容
print(lb.ask("玛格丽特把指南针藏在哪里？"))
```

也可以不加载模型（无需 GPU），单独使用记忆库：

```python
mem = Memory()
mem.add(text, source="book")
print(mem.context("灯塔属于谁？", k=5, expand=1, budget=2000))
```

如果已经加载了模型，使用 `LongBookModel(model, tokenizer, mem)` 即可。

命令行：`python -m longbooklib add book.db novel.txt` · `python -m longbooklib search book.db "检索词"`

## 工作原理

| 步骤 | 说明 |
|---|---|
| 分块 | 按句子边界切分，且无损（拼接后等于原文），默认 1000 字符。 |
| 索引 | 稠密向量（numpy 矩阵）**加** BM25。中日文按字符 bigram 建索引，无需分词器。 |
| 检索 | 两路分数归一化到 0–1 后混合：`alpha·BM25 + (1-alpha)·稠密向量`。 |
| 上下文 | 命中的块再加上前后各 `expand` 个相邻块，相邻块合并为一段通顺文本，并按 **token** 预算截断（`memory_tokens`，默认 4096）。 |
| 再检索循环 | 如果笔记不够，模型可以输出 `<recall>检索词</recall>`，系统会再次检索（最多 `recall_rounds` 次，默认 2）。 |
| 存储 | SQLite 文件；完全相同的块不会重复保存；会记录所用嵌入器，重新打开时校验是否一致。 |

**关于嵌入器。** 默认的 `HashEmbedder` 不需要额外依赖，但基于词汇，能找到名字、数字和共有词，
对改写（同义表达）较弱。需要语义检索时请使用
`Memory(path, embedder=SentenceTransformerEmbedder(), alpha=0.5)`
（默认模型 `intfloat/multilingual-e5-small`，支持中文、日文、英文）。

## 针对循环（线性注意力）模型的调优

LongBook 使用固定大小的循环状态，因此提示词中的文本不能被完整回忆。`ask()` 默认启用了两项来自论文的设置。
两者都可以关闭，`lb.last_notes` 会显示实际交给模型的笔记。

- `reading="twice"` —— 把笔记和问题各给两遍（上下文、问题、上下文、问题），即 *Just Read Twice* 提示
  （[Arora 等，2024](https://arxiv.org/abs/2407.05483)）。论文报告在 16 个循环语言模型上平均提升 11.0 个点，
  且无需训练。token 预算由两份内容平分。
- `notes="evidence"` —— 只传入最相关的句子（`n_evidence=5`，通过 `Memory.evidence()`），而不是整块文本，
  以减少需要状态记住的内容。这一项是**我们的推断**，并非 LongBook 上的实验结果：依据是混合模型的精确回忆
  由注意力层承担，只剩循环状态时会丧失（[arXiv:2609.04434](https://arxiv.org/abs/2609.04434)），
  以及线性 RNN 的状态容量有限（[arXiv:2607.07386](https://arxiv.org/abs/2607.07386)）。

请在自己的任务上对比，例如 `lb.ask(q, notes="passages", reading="once")`（原来的设置）与默认值。
未实现：[上下文感知解码](https://arxiv.org/abs/2305.14739)（需要自定义解码循环）。

## 支持的模型

模型卡中提到的特殊要求已做成预设（`longbooklib.PRESETS`）并自动应用：

| 模型 | 自动应用的设置 |
|---|---|
| `summerAIO/Qwen3.8-27B-Longbook27-v2` | `trust_remote_code=True` |
| `summerAIO/Qwen3.8-27B-Longbook27-VL-GDN64` | `trust_remote_code=True`，使用 `AutoModelForMultimodalLM` 加载（此处仅用于文本） |
| `summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2` | `trust_remote_code=True` |
| `j-llm/Qwen3.5-2B-Longbook2` | `use_cache=False`（模型卡报告了缓存 `get_seq_length` 报错） |

`enable_thinking` 默认关闭（若开启，回答中的 `<think>` 块会被自动去除）。
解码默认为 greedy，可以把任意 `generate` 参数直接传给 `ask(...)`。

## API

- `Memory(path=None, embedder=None, chunk_chars=1000, alpha=0.7)` —— `add`、`add_file`、`search`、`passages`、
  `context`、`delete(source)`、`clear`、`sources`、`len(mem)`。
- `LongBookModel(model, tokenizer, memory)` / `.from_pretrained(name, memory)` —— `remember`、`remember_file`、
  `ask(question, k=6, expand=1, recall_rounds=2, history=None, remember_turns=False)`、`generate(messages)`。

## 已验证与未验证

- 合成的“大海捞针”测试（200 篇文档、约 5,300 个块，每篇含一条带罕见名字的事实）：默认设置下
  recall@1 = 200/200，CPU 上每次查询约 1 ms。名字是唯一的，对 BM25 有利，
  因此它只是功能自检，并不代表回答质量的基准。
- 测试（`pytest`，23 项）无需 torch，模型封装部分用桩（stub）模型验证。
  **尚未用真实的 LongBook 权重运行过。** 模型卡中的注意事项：加载时会执行远程代码（建议固定 `revision`）；
  27B 系列需要把整个模型放进 GPU（不支持 CPU 卸载）；9B 的模型卡写明直接使用原生 `transformers` 的路径尚未验证。

## 测试

```bash
pip install -e ".[test]" && pytest
```

许可证：Apache-2.0（与模型相同）
