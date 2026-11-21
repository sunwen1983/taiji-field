"""Build (or rebuild) the cached 64 hexagram document embeddings.

Usage: python scripts/build_embeddings.py [--model NAME] [--force]

Downloads the embedding model on first run (BAAI/bge-small-zh-v1.5 by
default), encodes the 64 hexagram documents, and writes
taiji_field/data/embeddings.npz. The cache is keyed by model name and a
hash of the document texts; the encoder rebuilds it automatically if the
model or the documents change, so this script is only needed to
pre-build it (e.g. before committing the cache).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taiji_field.grounding import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    GroundingEncoder,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if a valid cache exists")
    parser.add_argument("--output", default=str(DEFAULT_CACHE))
    args = parser.parse_args()
    cache = Path(args.output)
    if args.force and cache.exists():
        cache.unlink()
    encoder = GroundingEncoder(model_name=args.model, cache_path=cache)
    print(f"backend: {encoder.backend}")
    emb = encoder.doc_embeddings()
    print(f"cached {emb.shape[0]} document embeddings "
          f"(dim {emb.shape[1]}) -> {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
