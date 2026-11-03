"""Tokenizer and embedders.

``HashEmbedder`` needs nothing but numpy and is deterministic, so a memory is
usable out of the box. It is lexical (word + char-trigram hashing). For real
semantic matching use ``SentenceTransformerEmbedder`` (multilingual by default).
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import List, Sequence

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9_À-ɏ]+|[぀-ヿ㐀-鿿가-힯]+")


def tokenize(text: str) -> List[str]:
    """Lowercased words; CJK runs become character bigrams (no segmenter needed)."""
    toks: List[str] = []
    for m in _WORD.finditer(text.lower()):
        w = m.group()
        if ord(w[0]) > 0x2FFF:
            if len(w) == 1:
                toks.append(w)
            else:
                toks.extend(w[i : i + 2] for i in range(len(w) - 1))
        else:
            toks.append(w)
    return toks


class HashEmbedder:
    """Dependency-free signed feature-hashing embedder (L2-normalised)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            toks = tokenize(text)
            feats = Counter(toks)
            feats = {f: float(c) for f, c in feats.items()}
            for t in toks:  # char trigrams make long Latin words robust to inflection
                if len(t) >= 5 and t.isascii():
                    for j in range(len(t) - 2):
                        key = "#" + t[j : j + 3]
                        feats[key] = feats.get(key, 0.0) + 0.3
            for f, c in feats.items():
                h = zlib.crc32(f.encode("utf-8"))
                out[r, h % self.dim] += (1.0 if (h >> 16) & 1 else -1.0) * math.log1p(c)
            n = float(np.linalg.norm(out[r]))
            if n > 0:
                out[r] /= n
        return out

    encode_query = encode


class SentenceTransformerEmbedder:
    """Semantic embedder (en/ja/zh/...). Needs ``pip install sentence-transformers``."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        query_prefix: str = "query: ",
        doc_prefix: str = "passage: ",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model_name}"
        self._qp, self._dp, self._bs = query_prefix, doc_prefix, batch_size

    def _enc(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        v = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self._bs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._enc(texts, self._dp)

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._enc(texts, self._qp)
