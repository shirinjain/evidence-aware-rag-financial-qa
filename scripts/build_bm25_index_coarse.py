"""
Same as build_bm25_index.py, but indexes coarse_chunks.parquet (table-
level units) instead of child_chunks.parquet (row-level).
"""
from pathlib import Path

import bm25s
import pandas as pd

from build_bm25_index import STOPWORDS, _build_index_text

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "coarse_chunks.parquet"
INDEX_DIR = ROOT / "data" / "processed" / "bm25_index_coarse"


def main():
    df = pd.read_parquet(CHUNKS_PATH)
    print(f"indexing {len(df)} coarse units")

    corpus_texts = df.apply(_build_index_text, axis=1).tolist()
    chunk_ids = df["chunk_id"].tolist()

    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=STOPWORDS)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(str(INDEX_DIR), corpus=chunk_ids)
    print(f"saved index to {INDEX_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
