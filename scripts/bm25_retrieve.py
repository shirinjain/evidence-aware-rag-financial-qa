"""
Loads the saved BM25 index and provides a simple retrieve(query_text, k)
function returning a ranked list of chunk_ids (best first) - the exact
list[str] shape eval/metrics.py's precision_at_k/recall_at_k/ndcg_at_k
expect as their `retrieved` argument.
"""
import sys
from pathlib import Path

import bm25s

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "processed" / "bm25_index"

sys.path.insert(0, str(ROOT))
from scripts.build_bm25_index import STOPWORDS  # same list used to build the index - must match

_retriever = None
_corpus = None


def _load():
    global _retriever, _corpus
    if _retriever is None:
        _retriever = bm25s.BM25.load(str(INDEX_DIR), load_corpus=True, show_progress=False)
        _corpus = _retriever.corpus
    return _retriever, _corpus


def retrieve(query_text: str, k: int = 20) -> list:
    """Returns a ranked list of chunk_id strings, best match first."""
    return [cid for cid, _ in retrieve_with_scores(query_text, k)]


def retrieve_with_scores(query_text: str, k: int = 20) -> list:
    """Returns [(chunk_id, bm25_score), ...], best match first - needed
    for score-based fusion (convex combination), not just rank-based
    fusion (RRF)."""
    retriever, corpus = _load()
    query_tokens = bm25s.tokenize([query_text], stopwords=STOPWORDS, return_ids=False, show_progress=False)
    results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=k, show_progress=False)
    return [(r["text"], float(s)) for r, s in zip(results[0], scores[0])]


if __name__ == "__main__":
    import json

    from eval.metrics import evaluate_ranking, load_gold_relevance

    gold = load_gold_relevance()
    questions = [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]
    q = next(q for q in questions if q["financebench_id"] == "financebench_id_03029")  # 3M capex example
    gold_chunks = gold[q["financebench_id"]]

    print("QUESTION:", q["question"])
    print("gold chunks:", len(gold_chunks))
    print()

    retrieved = retrieve(q["question"], k=20)
    print("top 10 retrieved:")
    for i, cid in enumerate(retrieved[:10], 1):
        marker = " <-- GOLD" if cid in gold_chunks else ""
        print(f"  {i:2d}. {cid}{marker}")

    scores = evaluate_ranking(retrieved, gold_chunks, ks=[5, 10, 20])
    print()
    print("scores:", scores)
