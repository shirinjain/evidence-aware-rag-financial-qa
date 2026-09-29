"""
Builds the ChromaDB vector store from mpnet_embeddings.npy (computed on
Colab - see colab_embed_mpnet.py) joined with child_chunks.parquet's
metadata. Row-order alignment verified directly beforehand (146,821/
146,821 chunk_ids matched in order) - positional indexing is safe.

Output: data/processed/dense_index/ (a Chroma persistent database dir)
"""
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "mpnet_embeddings.npy"
DB_DIR = ROOT / "data" / "processed" / "dense_index"
COLLECTION_NAME = "chunks"
BATCH_SIZE = 5000  # under Chroma's 5,461 max_batch_size


def build_metadata(row) -> dict:
    return {
        "entity": row["entity"],
        "fiscal_period": int(row["fiscal_period"]),
        "chunk_type": row["chunk_type"],
        "section_title": row["section_title"] or "",
        "subsection_title": row["subsection_title"] if pd.notna(row["subsection_title"]) else "",
    }


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    assert len(chunks) == len(embeddings), f"row count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
    print(f"{len(chunks)} chunks to insert")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    n = len(chunks)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = chunks.iloc[start:end]
        collection.add(
            ids=batch["chunk_id"].tolist(),
            embeddings=embeddings[start:end].tolist(),
            metadatas=[build_metadata(row) for _, row in batch.iterrows()],
            documents=batch["raw_text"].tolist(),
        )
        print(f"  inserted {end}/{n}", flush=True)

    print(f"\ndone. collection count: {collection.count()}")


if __name__ == "__main__":
    main()
