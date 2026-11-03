"""Tiny CLI:  python -m longbooklib add|search|ask DB ..."""
from __future__ import annotations

import argparse
import sys

from .embed import HashEmbedder
from .memory import Memory


def _memory(args) -> Memory:
    emb = None
    if args.embedder == "e5":
        from .embed import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
    return Memory(args.db, embedder=emb or HashEmbedder())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="longbook")
    p.add_argument("--embedder", choices=["hash", "e5"], default="hash")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add text files to a memory db")
    a.add_argument("db")
    a.add_argument("files", nargs="+")

    s = sub.add_parser("search", help="print the notes retrieved for a query")
    s.add_argument("db")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)

    q = sub.add_parser("ask", help="answer a question with a LongBook model")
    q.add_argument("db")
    q.add_argument("query")
    q.add_argument("--model", required=True)

    args = p.parse_args(argv)
    mem = _memory(args)
    if args.cmd == "add":
        for f in args.files:
            print(f"{f}: +{mem.add_file(f)} chunks")
        print(f"total chunks: {len(mem)}")
    elif args.cmd == "search":
        print(mem.context(args.query, k=args.k) or "(no results)")
    else:
        from .model import LongBookModel

        print(LongBookModel.from_pretrained(args.model, mem).ask(args.query))
    return 0


if __name__ == "__main__":
    sys.exit(main())
