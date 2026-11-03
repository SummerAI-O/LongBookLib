"""Lossless sentence-aware chunking.

Chunks are exact, contiguous slices of the input, so ``"".join(chunks)`` gives
back the original text (whitespace-only chunks aside). That lets the memory
re-assemble neighbouring chunks into one clean passage later.
"""
from __future__ import annotations

import re
from typing import List

# A piece ends after CJK/Latin sentence punctuation, a ". " boundary, or newlines.
_END = re.compile(r"[。！？!?]+\s*|\.\s+|\n+")


def _pieces(text: str) -> List[str]:
    out, last = [], 0
    for m in _END.finditer(text):
        out.append(text[last : m.end()])
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return out


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Sentences are packed greedily; a single sentence longer than ``max_chars``
    is hard-split.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    chunks: List[str] = []
    cur: List[str] = []
    size = 0

    def flush() -> None:
        nonlocal cur, size
        if cur:
            chunks.append("".join(cur))
            cur, size = [], 0

    for piece in _pieces(text):
        while len(piece) > max_chars:
            flush()
            chunks.append(piece[:max_chars])
            piece = piece[max_chars:]
        if size + len(piece) > max_chars:
            flush()
        cur.append(piece)
        size += len(piece)
    flush()
    return [c for c in chunks if c.strip()]
