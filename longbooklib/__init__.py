"""longbooklib - smart external memory for the LongBook model series."""
from .chunker import chunk_text
from .embed import HashEmbedder, SentenceTransformerEmbedder, tokenize
from .memory import Evidence, Hit, Memory, Passage, format_evidence, format_passages
from .model import PRESETS, LongBookModel

__version__ = "0.2.0"
__all__ = [
    "Memory",
    "Hit",
    "Passage",
    "Evidence",
    "format_passages",
    "format_evidence",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "LongBookModel",
    "PRESETS",
    "chunk_text",
    "tokenize",
    "__version__",
]
