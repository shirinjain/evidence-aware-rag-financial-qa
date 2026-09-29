"""
Builds a BM25 index over child_chunks.parquet.

Indexed on a contextual prefix (entity + fiscal_period + section_title)
concatenated with raw_text, not raw_text alone - so a query mentioning
a company name or year can match even if the chunk's own text doesn't
repeat it (e.g. a bare "Net sales | $32,765 | ..." row has no company
name in it otherwise). This corpus doesn't have a pre-built
embedding_text column (unlike the original project's contextual_prefix.py
pipeline), so it's built inline here instead.

STOPWORDS uses nltk's 198-word list, not bm25s's default 33-word "en"
list - the original project found the default list too small (missing
common words like "what"/"how"/"does"/"from" that show up constantly in
financial questions), and applied the fix at both index-build and
query time consistently. Same fix applied here from the start.

Output: data/processed/bm25_index/ (loadable via bm25s.BM25.load)
"""
from pathlib import Path

import bm25s
import nltk
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
INDEX_DIR = ROOT / "data" / "processed" / "bm25_index"

try:
    STOPWORDS = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = nltk.corpus.stopwords.words("english")


def _build_index_text(row) -> str:
    parts = [str(row["entity"]), f"FY{row['fiscal_period']}", str(row["section_title"] or "")]
    return f"{', '.join(p for p in parts if p)}: {row['raw_text']}"


def main():
    df = pd.read_parquet(CHUNKS_PATH)
    print(f"indexing {len(df)} chunks")

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
