# longbooklib

[English](README.md) | **日本語** | [中文](README.zh.md)

[LongBook シリーズ](https://huggingface.co/collections/summerAIO/longbook-series)のための、賢い外部メモリ・ライブラリです。
LongBook は一定サイズの再帰状態で長文を処理するためメモリ効率に優れますが、遠い位置にある固有名詞・数値・引用などの
正確な情報は落ちやすくなります。`longbooklib` はテキストをモデルの外に保存し、質問ごとに必要な部分だけを探して
プロンプトに入れることで、これを補います。

```
 長文 ──► チャンク化 ──► 埋め込み + BM25 索引 ──► SQLite（永続化）
                                                      │
 質問 ──► ハイブリッド検索 ──► 隣接チャンクを結合 ──► トークン予算内のノート ──► LongBook ──► 回答
                ▲                                                                   │
                └──────────── <recall>新しい検索語</recall>（モデルが再検索を要求） ◄──┘
```

意図的に小さく作っています。モジュールは 4 つ、必須依存は `numpy` だけで、ベクトル DB サーバーも不要です。

## インストール

```bash
pip install -e .                 # コア（numpy のみ）。torch なしで Memory が使えます
pip install -e ".[hf]"           # + transformers/torch（LongBook モデルを動かす場合）
pip install -e ".[semantic]"     # + sentence-transformers（多言語の意味検索）
```

## クイックスタート

```python
from longbooklib import LongBookModel, Memory

mem = Memory("novel.db")                       # 永続化。Memory() なら RAM 上のみ
lb = LongBookModel.from_pretrained("summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2", mem)

lb.remember(open("novel.txt", encoding="utf-8").read(), source="novel")   # 初回のみ。再登録しても重複しません
print(lb.ask("マルグリットはコンパスをどこに隠しましたか？"))
```

モデルなし（GPU 不要）でメモリ単体でも使えます。

```python
mem = Memory()
mem.add(text, source="book")
print(mem.context("灯台は誰のものですか？", k=5, expand=1, budget=2000))
```

すでにモデルを読み込んである場合は `LongBookModel(model, tokenizer, mem)` を使います。

CLI: `python -m longbooklib add book.db novel.txt` ・ `python -m longbooklib search book.db "検索語"`

## 仕組み

| 段階 | 内容 |
|---|---|
| チャンク化 | 文境界を考慮し、元の文字列を欠落なく復元できます（既定 1000 文字）。 |
| 索引 | 密ベクトル（numpy 行列）と BM25 の両方。日本語・中国語は文字 bigram で索引するため形態素解析は不要です。 |
| 検索 | 2 つのスコアを 0〜1 に正規化して `alpha·BM25 + (1-alpha)·密ベクトル` で合成します。 |
| コンテキスト | 上位ヒットの前後 `expand` チャンクを追加し、連続するチャンクは 1 つの読みやすい文章に結合、**トークン**予算（`memory_tokens`、既定 4096）に収めます。 |
| 再検索ループ | ノートが足りなければモデルが `<recall>検索語</recall>` を返し、再検索します（`recall_rounds` 回まで、既定 2）。 |
| 保存 | SQLite ファイル。完全一致のチャンクは重複登録されません。埋め込みの種類を記録し、再オープン時に照合します。 |

**埋め込みについて。** 既定の `HashEmbedder` は追加依存なしですが語彙ベースなので、固有名詞・数値・共通語は見つけられても
言い換えには弱いです。意味検索には
`Memory(path, embedder=SentenceTransformerEmbedder(), alpha=0.5)` を使ってください
（既定モデルは `intfloat/multilingual-e5-small`。日本語・英語・中国語に対応）。

## 再帰型（リニアアテンション）モデル向けの調整

LongBook は一定サイズの再帰状態を持つため、プロンプト内の文章を完全には想起できません。そこで `ask()` は、
論文に基づく次の 2 つを既定で有効にしています。どちらも無効にでき、`lb.last_notes` でモデルに実際に渡した
ノートを確認できます。

- `reading="twice"` — ノートと質問を 2 回与えます（文脈・質問・文脈・質問）。*Just Read Twice* のプロンプト
  （[Arora ほか, 2024](https://arxiv.org/abs/2407.05483)）で、16 種類の再帰型 LM で平均 +11.0 ポイントの改善が
  報告されています（追加学習なし）。トークン予算は 2 つのコピーで分け合います。
- `notes="evidence"` — チャンク全体ではなく、質問に最も合う文だけ（`n_evidence=5`、`Memory.evidence()`）を渡し、
  状態に保持させる量を減らします。この部分は LongBook での結果ではなく**こちらの推測**です。ハイブリッドモデルでは
  正確な想起をアテンション層が担い、再帰状態だけになると失われること
  （[arXiv:2609.04434](https://arxiv.org/abs/2609.04434)）、線形 RNN の状態容量が有限であること
  （[arXiv:2607.07386](https://arxiv.org/abs/2607.07386)）を根拠にしています。

自分のタスクで比較してください。例: `lb.ask(q, notes="passages", reading="once")`（従来の設定）と既定値の比較。
未実装: [文脈認識デコーディング](https://arxiv.org/abs/2305.14739)（独自のデコードループが必要）。

## 対応モデル

モデルカードに書かれた癖はプリセット（`longbooklib.PRESETS`）として自動適用されます。

| モデル | 自動適用される設定 |
|---|---|
| `summerAIO/Qwen3.8-27B-Longbook27-v2` | `trust_remote_code=True` |
| `summerAIO/Qwen3.8-27B-Longbook27-VL-GDN64` | `trust_remote_code=True`、`AutoModelForMultimodalLM` で読み込み（ここではテキストのみ使用） |
| `summerAIO/Qwen3.5-9B-Longbook9-GDN32-x2` | `trust_remote_code=True` |
| `j-llm/Qwen3.5-2B-Longbook2` | `use_cache=False`（カードにキャッシュの `get_seq_length` エラーが報告されているため） |

`enable_thinking` は既定でオフです（オンにした場合、回答から `<think>` ブロックは取り除かれます）。
デコードは greedy で、`ask(...)` に `generate` の引数をそのまま渡せます。

## API

- `Memory(path=None, embedder=None, chunk_chars=1000, alpha=0.7)` — `add`、`add_file`、`search`、`passages`、
  `context`、`delete(source)`、`clear`、`sources`、`len(mem)`。
- `LongBookModel(model, tokenizer, memory)` / `.from_pretrained(name, memory)` — `remember`、`remember_file`、
  `ask(question, k=6, expand=1, recall_rounds=2, history=None, remember_turns=False)`、`generate(messages)`。

## 検証できたこと・できていないこと

- 合成の「針探し」テスト（200 文書・約 5,300 チャンク、各文書にまれな名前を含む事実を 1 件）: 既定設定で
  recall@1 = 200/200、CPU で 1 クエリあたり約 1 ms。名前が一意なので BM25 に有利な条件です。回答品質のベンチマークではなく、
  動作確認として見てください。
- テスト（`pytest`、23 件）は torch なしで動き、モデル部分はスタブで検証しています。
  **実際の LongBook の重みではまだ実行していません。** モデルカードの注意点: 読み込み時にリモートコードが実行されます
  （`revision` の固定を推奨）。27B 系はモデル全体を GPU に載せる必要があります（CPU オフロード非対応）。
  9B のカードには、素の `transformers` 経路は未検証と書かれています。

## テスト

```bash
pip install -e ".[test]" && pytest
```

ライセンス: Apache-2.0（モデルと同じ）
