"""External memory: SQLite persistence + in-RAM hybrid (BM25 + dense) retrieval.

Design in one paragraph: text is cut into contiguous chunks, each chunk is
embedded and also indexed for BM25. A query is scored by both, the two score
lists are normalised and blended, and the winning chunks are expanded with
their neighbours (same source, adjacent position) and merged back into
readable passages that fit a token budget.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .chunker import chunk_text
from .embed import HashEmbedder, tokenize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
    id     INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    seq    INTEGER NOT NULL,
    text   TEXT NOT NULL,
    hash   TEXT NOT NULL UNIQUE,
    meta   TEXT,
    ts     REAL NOT NULL,
    vec    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source, seq);
CREATE TABLE IF NOT EXISTS info(key TEXT PRIMARY KEY, value TEXT);
"""

@dataclass
class Hit:
    index: int  # position inside this Memory instance (not stable across delete)
    id: int
    text: str
    score: float
    source: str
    seq: int
    meta: dict = field(default_factory=dict)
    ts: float = 0.0


@dataclass
class Passage:
    source: str
    start: int  # first chunk seq
    end: int  # last chunk seq
    text: str
    score: float
    rank: int


class Memory:
    """A persistent, searchable external store for long texts and chat turns.

    >>> mem = Memory()                      # in RAM; Memory("book.db") persists
    >>> mem.add(long_text, source="book")   # -> number of new chunks
    >>> print(mem.context("who killed the butler?"))
    """

    def __init__(
        self,
        path: Optional[str] = None,
        embedder=None,
        chunk_chars: int = 1000,
        k1: float = 1.5,
        b: float = 0.75,
        alpha: float = 0.7,
    ):
        """``alpha`` is the BM25 weight in hybrid scoring (0 = dense only, 1 = BM25 only).

        0.7 suits the default HashEmbedder (itself lexical). With a semantic
        embedder such as SentenceTransformerEmbedder, 0.5 is a good starting point.
        """
        self.embedder = embedder or HashEmbedder()
        self.chunk_chars = chunk_chars
        self.k1, self.b = k1, b
        self.alpha = alpha
        self._db = sqlite3.connect(path or ":memory:")
        if path:
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._check_embedder()
        self._load()

    # ------------------------------------------------------------------ state
    def _check_embedder(self) -> None:
        cur = dict(self._db.execute("SELECT key, value FROM info"))
        name = getattr(self.embedder, "name", type(self.embedder).__name__)
        if "embedder" in cur and cur["embedder"] != name:
            raise ValueError(
                f"memory was built with embedder {cur['embedder']!r} but {name!r} was given; "
                "open it with the same embedder or re-add the text to a new file"
            )
        self._db.executemany(
            "INSERT OR REPLACE INTO info VALUES (?, ?)",
            [("embedder", name), ("dim", str(self.embedder.dim))],
        )
        self._db.commit()

    def _reset(self) -> None:
        self._ids: List[int] = []
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._seqs: List[int] = []
        self._metas: List[dict] = []
        self._ts: List[float] = []
        self._dl: List[int] = []
        self._post: Dict[str, Dict[int, int]] = {}
        self._pos: Dict[Tuple[str, int], int] = {}
        self._by_source: Dict[str, List[int]] = {}
        self._next_seq: Dict[str, int] = {}
        self._hashes: set = set()
        self._total_len = 0
        self._next_id = 1
        self._mat = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._n = 0

    def _load(self) -> None:
        self._reset()
        rows = self._db.execute(
            "SELECT id, source, seq, text, hash, meta, ts, vec FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            return
        vecs = np.vstack([np.frombuffer(r[7], dtype=np.float32) for r in rows])
        self._grow(len(rows))
        self._mat[: len(rows)] = vecs
        for r in rows:
            self._register(r[0], r[1], r[2], r[3], r[4], json.loads(r[5] or "{}"), r[6])
        self._next_id = rows[-1][0] + 1

    def _grow(self, extra: int) -> None:
        need = self._n + extra
        if need > self._mat.shape[0]:
            cap = max(need, int(self._mat.shape[0] * 1.5) + 64)
            new = np.zeros((cap, self.embedder.dim), dtype=np.float32)
            new[: self._n] = self._mat[: self._n]
            self._mat = new

    def _register(self, id_, source, seq, text, hash_, meta, ts) -> None:
        i = self._n
        self._n += 1
        self._ids.append(id_)
        self._texts.append(text)
        self._sources.append(source)
        self._seqs.append(seq)
        self._metas.append(meta)
        self._ts.append(ts)
        self._pos[(source, seq)] = i
        self._by_source.setdefault(source, []).append(i)
        self._next_seq[source] = max(self._next_seq.get(source, 0), seq + 1)
        self._hashes.add(hash_)
        toks = tokenize(text)
        self._dl.append(len(toks))
        self._total_len += len(toks)
        for t, c in Counter(toks).items():
            self._post.setdefault(t, {})[i] = c

    # -------------------------------------------------------------- mutation
    def add(self, text: str, source: str = "doc", meta: Optional[dict] = None) -> int:
        """Chunk, embed and store ``text``. Returns the number of NEW chunks.

        Exact-duplicate chunks are skipped, so re-adding a document is cheap.
        """
        new_chunks, new_hashes, seen = [], [], set()
        for c in chunk_text(text, self.chunk_chars):
            h = hashlib.sha1(c.encode("utf-8")).hexdigest()
            if h in self._hashes or h in seen:
                continue
            seen.add(h)
            new_chunks.append(c)
            new_hashes.append(h)
        if not new_chunks:
            return 0
        vecs = np.asarray(self.embedder.encode(new_chunks), dtype=np.float32)
        now = time.time()
        meta = meta or {}
        meta_json = json.dumps(meta, ensure_ascii=False)
        seq0 = self._next_seq.get(source, 0)
        rows = []
        self._grow(len(new_chunks))
        for j, (c, h) in enumerate(zip(new_chunks, new_hashes)):
            id_ = self._next_id + j
            rows.append((id_, source, seq0 + j, c, h, meta_json, now, vecs[j].tobytes()))
            self._mat[self._n] = vecs[j]
            self._register(id_, source, seq0 + j, c, h, meta, now)
        self._next_id += len(new_chunks)
        self._db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)", rows)
        self._db.commit()
        return len(new_chunks)

    def add_file(self, path: str, source: Optional[str] = None, encoding: str = "utf-8") -> int:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            return self.add(f.read(), source=source or path)

    def delete(self, source: str) -> int:
        """Remove every chunk of ``source``. Returns how many were removed."""
        cur = self._db.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._db.commit()
        self._load()
        return cur.rowcount

    def clear(self) -> None:
        self._db.execute("DELETE FROM chunks")
        self._db.commit()
        self._load()

    # ------------------------------------------------------------- inspection
    def __len__(self) -> int:
        return self._n

    def sources(self) -> Dict[str, int]:
        return {s: len(v) for s, v in self._by_source.items()}

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- search
    @staticmethod
    def _top(scores: np.ndarray, m: int) -> np.ndarray:
        m = min(m, len(scores))
        if m <= 0:
            return np.zeros(0, dtype=np.int64)
        idx = np.argpartition(-scores, m - 1)[:m]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return idx[np.isfinite(scores[idx])]

    def _bm25(self, query: str) -> Dict[int, float]:
        n = self._n
        avg = self._total_len / n if n else 1.0
        avg = avg or 1.0
        scores: Dict[int, float] = {}
        for t in set(tokenize(query)):
            post = self._post.get(t)
            if not post:
                continue
            df = len(post)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, tf in post.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self._dl[i] / avg)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return scores

    def search(self, query: str, k: int = 5, source: Optional[str] = None) -> List[Hit]:
        """Hybrid retrieval: dense cosine + BM25, blended by normalised score."""
        n = self._n
        if n == 0 or not query.strip():
            return []
        pool = min(n, max(32, k * 8))

        enc = getattr(self.embedder, "encode_query", self.embedder.encode)
        q = np.asarray(enc([query]), dtype=np.float32)[0]
        dense = self._mat[:n] @ q
        sparse = self._bm25(query)

        if source is not None:
            allowed = self._by_source.get(source, [])
            mask = np.full(n, -np.inf, dtype=np.float32)
            mask[allowed] = 0.0
            dense = dense + mask
            sparse = {i: s for i, s in sparse.items() if self._sources[i] == source}

        # Score-aware fusion (not RRF): a rank-only fusion throws away how decisively
        # BM25 prefers a chunk that contains a rare name/number. Each list is
        # normalised to [0, 1] and combined as (1-alpha)*dense + alpha*bm25.
        fused: Dict[int, float] = {}
        top = self._top(dense, pool)
        if len(top):
            d = dense[top]
            lo, hi = float(d.min()), float(d.max())
            span = hi - lo
            for i, s in zip(top, d):
                dn = (float(s) - lo) / span if span > 1e-9 else 1.0
                fused[int(i)] = (1.0 - self.alpha) * dn
        ranked_sparse = sorted(sparse.items(), key=lambda kv: -kv[1])[:pool]
        if ranked_sparse:
            smax = ranked_sparse[0][1] or 1.0
            for i, s in ranked_sparse:
                fused[i] = fused.get(i, 0.0) + self.alpha * (s / smax)

        best = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self._hit(i, s) for i, s in best]

    def _hit(self, i: int, score: float) -> Hit:
        return Hit(
            index=i,
            id=self._ids[i],
            text=self._texts[i],
            score=score,
            source=self._sources[i],
            seq=self._seqs[i],
            meta=self._metas[i],
            ts=self._ts[i],
        )

    # -------------------------------------------------------------- passages
    def passages(
        self,
        query: str,
        k: int = 5,
        expand: int = 1,
        budget: Optional[int] = None,
        count: Callable[[str], int] = len,
        source: Optional[str] = None,
    ) -> List[Passage]:
        """Top hits + ``expand`` neighbours each side, merged into passages.

        ``budget`` is measured with ``count`` (pass a tokenizer-based counter to
        get a token budget). It is a soft cap: the best hit is always included.
        Hits are admitted first, neighbours are added afterwards by distance.
        """
        hits = self.search(query, k=k, source=source)
        chosen: Dict[int, int] = {}  # position -> rank of the hit that pulled it in
        used = 0

        def fits(i: int) -> bool:
            return budget is None or not chosen or used + count(self._texts[i]) <= budget

        for rank, h in enumerate(hits):
            if fits(h.index):
                chosen[h.index] = rank
                used += count(h.text)
        for d in range(1, expand + 1):
            for rank, h in enumerate(hits):
                if h.index not in chosen:
                    continue
                for seq in (h.seq - d, h.seq + d):
                    j = self._pos.get((h.source, seq))
                    if j is None or j in chosen or not fits(j):
                        continue
                    chosen[j] = rank
                    used += count(self._texts[j])

        # merge contiguous chunks of the same source into one passage
        order = sorted(chosen, key=lambda i: (self._sources[i], self._seqs[i]))
        groups: List[List[int]] = []
        for i in order:
            if (
                groups
                and self._sources[groups[-1][-1]] == self._sources[i]
                and self._seqs[groups[-1][-1]] + 1 == self._seqs[i]
            ):
                groups[-1].append(i)
            else:
                groups.append([i])
        out = []
        for g in groups:
            rank = min(chosen[i] for i in g)
            out.append(
                Passage(
                    source=self._sources[g[0]],
                    start=self._seqs[g[0]],
                    end=self._seqs[g[-1]],
                    text="".join(self._texts[i] for i in g).strip(),
                    score=hits[rank].score,
                    rank=rank,
                )
            )
        out.sort(key=lambda p: p.rank)
        return out

    def context(self, query: str, **kw) -> str:
        """Retrieve and format notes ready to paste into a prompt."""
        return format_passages(self.passages(query, **kw))


def format_passages(passages: List[Passage]) -> str:
    blocks = []
    for n, p in enumerate(passages, 1):
        where = f"{p.source}#{p.start}" if p.start == p.end else f"{p.source}#{p.start}-{p.end}"
        blocks.append(f"[{n}] ({where})\n{p.text}")
    return "\n\n".join(blocks)
