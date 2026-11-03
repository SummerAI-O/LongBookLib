# longbooklib

**English** | [日本語](README.ja.md) | [中文](README.zh.md)

Smart external memory for the [LongBook model series](https://huggingface.co/collections/summerAIO/longbook-series).
LongBook models keep a constant-size recurrent state, which is great for memory use but means exact details
(names, numbers, quotes) from far back are easy to lose. `longbooklib` stores the text outside the model,
finds what matters for each question, and puts only that into the prompt.

```
 long text ──► chunk ──► embed + BM25 index ──► SQLite (persistent)
                                                    │
 question ──► hybrid search ──► neighbours merged ──► token-budgeted notes ──► LongBook ──► answer
                  ▲                                                              │
                  └──────────── <recall>new query</recall> (model asks again) ◄──┘
```

Small on purpose: 4 modules, `numpy` is the only hard dependency, no vector DB server.

## Install

```bash
pip install -e .                 # core (numpy only) – Memory works without torch
pip install -e ".[hf]"           # + transformers/torch to run LongBook models
pip install -e ".[semantic]"     # + sentence-transformers for semantic (multilingual) search
```

## Quick start

```python
from longbooklib import LongBookModel, Memory

mem = Memory("novel.db")                       # persistent; Memory() keeps it in RAM
lb = LongBookModel.from_pretrained("summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2", mem)

lb.remember(open("novel.txt", encoding="utf-8").read(), source="novel")   # once; re-adding is a no-op
print(lb.ask("Where did Marguerite hide the compass?"))
```

Use the memory alone (no model, no GPU):

```python
mem = Memory()
mem.add(text, source="book")
print(mem.context("who owns the lighthouse?", k=5, expand=1, budget=2000))
```

Already have a model loaded? `LongBookModel(model, tokenizer, mem)`.

CLI: `python -m longbooklib add book.db novel.txt` · `python -m longbooklib search book.db "query"`.

## How it works

| Step | What it does |
|---|---|
| Chunking | Sentence-aware, lossless (`"".join(chunks) == text`), default 1000 chars. |
| Index | Dense vectors (numpy matrix) **and** BM25. CJK is indexed as character bigrams, so ja/zh need no segmenter. |
| Search | Both score lists are normalised to 0–1 and blended: `alpha·BM25 + (1-alpha)·dense`. |
| Context | Top hits get ±`expand` neighbouring chunks, adjacent chunks are merged into one readable passage, and everything is cut to a **token** budget (`memory_tokens`, default 4096). |
| Recall loop | If the notes are not enough, the model may answer `<recall>query</recall>`; the memory is searched again (up to `recall_rounds`, default 2). |
| Storage | SQLite file; exact-duplicate chunks are skipped; the embedder name is recorded and checked on reopen. |

**Embedders.** The default `HashEmbedder` needs nothing and is lexical, so it finds names, numbers and
shared words but not paraphrases. For semantic search use
`Memory(path, embedder=SentenceTransformerEmbedder(), alpha=0.5)` (default model:
`intfloat/multilingual-e5-small`, covers English/Japanese/Chinese).

## Supported models

Quirks come from the model cards and are applied automatically (`longbooklib.PRESETS`):

| Model | Applied automatically |
|---|---|
| `summerAIO/Qwen3.8-27B-Longbook27-v2` | `trust_remote_code=True` |
| `summerAIO/Qwen3.8-27B-Longbook27-VL-GDN64` | `trust_remote_code=True`, loaded with `AutoModelForMultimodalLM` (text-only use here) |
| `summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2` | `trust_remote_code=True` |
| `j-llm/Qwen3.5-2B-Longbook2` | `use_cache=False` (the card reports a cache `get_seq_length` error) |

`enable_thinking` is off by default (`<think>` blocks are stripped from answers if you turn it on).
Decoding is greedy; pass any `generate` kwarg to `ask(...)`.

## API

- `Memory(path=None, embedder=None, chunk_chars=1000, alpha=0.7)` – `add`, `add_file`, `search`, `passages`,
  `context`, `delete(source)`, `clear`, `sources`, `len(mem)`.
- `LongBookModel(model, tokenizer, memory)` / `.from_pretrained(name, memory)` – `remember`, `remember_file`,
  `ask(question, k=6, expand=1, recall_rounds=2, history=None, remember_turns=False)`, `generate(messages)`.

## Measured, and not measured

- Synthetic needle test (200 documents, ~5.3k chunks, one rare-name fact each): recall@1 = 200/200 with the
  default setup, ~1 ms per query on CPU. The names are unique, which favours BM25 – treat it as a sanity check,
  not a benchmark of answer quality.
- The test-suite (`pytest`, 23 tests) runs without torch; the model wrapper is tested with stub models.
  **It has not been run against the real LongBook weights yet.** Notes from the cards: remote code is executed
  on load (pin a `revision`); the 27B models need the whole model on GPU (no CPU offload); the 9B card says the
  plain-`transformers` path is unverified.

## Tests

```bash
pip install -e ".[test]" && pytest
```

License: Apache-2.0 (same as the models).
