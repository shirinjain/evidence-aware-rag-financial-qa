"""
Loads the ChromaDB dense index and provides retrieve(query_text, k)
returning a ranked list of chunk_ids - same shape as bm25_retrieve.py.
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "processed" / "dense_index"
COLLECTION_NAME = "chunks"

_model = None
_collection = None


def _load():
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="mps")
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def retrieve(query_text: str, k: int = 20) -> list:
    return [cid for cid, _ in retrieve_with_scores(query_text, k)]


def retrieve_with_scores(query_text: str, k: int = 20) -> list:
    """Returns [(chunk_id, cosine_similarity), ...], best match first -
    Chroma returns cosine DISTANCE (lower=closer), converted here to
    similarity (higher=closer) so both retrievers use the same
    "higher is better" convention for fusion."""
    model, collection = _load()
    query_emb = model.encode([query_text], show_progress_bar=False)[0]
    result = collection.query(query_embeddings=[query_emb.tolist()], n_results=k)
    ids = result["ids"][0]
    similarities = [1 - d for d in result["distances"][0]]
    return list(zip(ids, similarities))


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, str(ROOT))
    from eval.metrics import evaluate_ranking, load_gold_relevance

    gold = load_gold_relevance()
    questions = [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]
    q = next(q for q in questions if q["financebench_id"] == "financebench_id_03029")
    gold_chunks = gold[q["financebench_id"]]

    print("QUESTION:", q["question"])
    print("gold chunks:", len(gold_chunks))

    retrieved = retrieve(q["question"], k=20)
    print("top 10 retrieved:")
    for i, cid in enumerate(retrieved[:10], 1):
        marker = " <-- GOLD" if cid in gold_chunks else ""
        print(f"  {i:2d}. {cid}{marker}")

    scores = evaluate_ranking(retrieved, gold_chunks, ks=[5, 10, 20])
    print("scores:", scores)
