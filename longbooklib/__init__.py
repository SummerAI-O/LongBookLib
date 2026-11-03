"""longbooklib - smart external memory for the LongBook model series."""
from .chunker import chunk_text
from .embed import HashEmbedder, SentenceTransformerEmbedder, tokenize
from .memory import Hit, Memory, Passage, format_passages
from .model import PRESETS, LongBookModel

__version__ = "0.1.0"
__all__ = [
    "Memory",
    "Hit",
    "Passage",
    "format_passages",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "LongBookModel",
    "PRESETS",
    "chunk_text",
    "tokenize",
    "__version__",
]
