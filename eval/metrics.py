"""
Shared retrieval evaluation metrics: NDCG@k, Precision@k, Recall@k.

Same hand-verified formulas as the original evidence-aware-rag
project's eval/metrics.py - reused as-is, only load_gold_relevance()
changes, since gold evidence here is stored as gold_relevance.json (a
list of {financebench_id, gold_chunk_ids, hop_groups, ...} dicts from
align_gold_evidence.py) rather than a flat parquet of (question_id,
chunk_id) pairs.

Relevance is binary here - a chunk is either gold for a question or it
isn't, no graded relevance.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_RELEVANCE_PATH = ROOT / "data" / "processed" / "gold_relevance.json"


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of the top-k retrieved chunks that are actually gold."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    n_relevant_in_top_k = sum(1 for c in top_k if c in relevant)
    return n_relevant_in_top_k / k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of ALL gold chunks for this query that appear in the
    top-k retrieved. 0.0 if the query has no gold chunks at all (can't
    divide by zero; there's nothing to recall)."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    n_found = sum(1 for c in relevant if c in top_k)
    return n_found / len(relevant)


def _dcg_at_k(relevance_sequence: list, k: int) -> float:
    """relevance_sequence[i] is 1 if the i-th (0-indexed) retrieved item
    is gold, else 0. Position discount uses 1-indexed rank: item at
    rank i (1-indexed) contributes relevance / log2(i + 1) - the "+1"
    avoids log2(1) = 0 for the very first position."""
    dcg = 0.0
    for i, rel in enumerate(relevance_sequence[:k]):
        rank = i + 1
        dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """DCG@k of the actual ranking, divided by the best-possible DCG@k
    (IDCG@k) - the score if all the query's gold chunks had been ranked
    first. 0.0 if the query has no gold chunks (IDCG would be 0)."""
    if not relevant:
        return 0.0
    actual_relevance = [1 if c in relevant else 0 for c in retrieved[:k]]
    dcg = _dcg_at_k(actual_relevance, k)

    n_ideal = min(k, len(relevant))
    ideal_relevance = [1] * n_ideal
    idcg = _dcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0


def load_gold_relevance(path: Path = GOLD_RELEVANCE_PATH) -> dict:
    """Returns {financebench_id: set(chunk_id)} - every gold chunk_id
    for that question (the tightened set where available, else the
    broader evidence_text match - see align_gold_evidence.py)."""
    records = json.load(open(path))
    return {r["financebench_id"]: set(r["gold_chunk_ids"]) for r in records}


def load_hop_groups(path: Path = GOLD_RELEVANCE_PATH) -> dict:
    """Returns {financebench_id: {named_item: [chunk_ids]}} - only
    populated for questions where gold was tightened via explicit
    justification line-item names (see coverage_metrics.py)."""
    records = json.load(open(path))
    return {r["financebench_id"]: r.get("hop_groups", {}) for r in records}


def evaluate_ranking(retrieved: list, relevant: set, ks: list = (5, 10, 20)) -> dict:
    """Convenience wrapper: all three metrics at each k, for one query."""
    return {
        f"{metric_name}@{k}": metric_fn(retrieved, relevant, k)
        for k in ks
        for metric_name, metric_fn in [
            ("precision", precision_at_k), ("recall", recall_at_k), ("ndcg", ndcg_at_k),
        ]
    }


if __name__ == "__main__":
    # Hand-built toy example, verified by hand before trusting this on
    # real data - 2 relevant chunks total ("A", "C") out of 5 retrieved.
    retrieved = ["B", "A", "D", "C", "E"]
    relevant = {"A", "C"}

    p3 = precision_at_k(retrieved, relevant, 3)
    r3 = recall_at_k(retrieved, relevant, 3)
    n3 = ndcg_at_k(retrieved, relevant, 3)
    print(f"Precision@3: {p3:.4f}  (expected 1/3 = 0.3333 - only 'A' is in top 3: [B,A,D])")
    print(f"Recall@3:    {r3:.4f}  (expected 1/2 = 0.5000 - found 1 of 2 relevant chunks)")
    print(f"NDCG@3:      {n3:.4f}  (expected ~0.3868 - see hand calc below)")
    assert abs(p3 - 1 / 3) < 1e-9
    assert abs(r3 - 0.5) < 1e-9
    # hand calc: DCG@3 = 0/log2(2) + 1/log2(3) + 0/log2(4) = 0.63093
    #            IDCG@3 = min(3,2)=2 ideal positions: 1/log2(2) + 1/log2(3) = 1.63093
    #            NDCG@3 = 0.63093 / 1.63093 = 0.38686
    assert abs(n3 - 0.38686) < 1e-4

    p5 = precision_at_k(retrieved, relevant, 5)
    r5 = recall_at_k(retrieved, relevant, 5)
    n5 = ndcg_at_k(retrieved, relevant, 5)
    print(f"\nPrecision@5: {p5:.4f}  (expected 2/5 = 0.4000 - both relevant now in top 5)")
    print(f"Recall@5:    {r5:.4f}  (expected 2/2 = 1.0000 - found all relevant chunks)")
    print(f"NDCG@5:      {n5:.4f}  (expected ~0.6509 - see hand calc below)")
    assert abs(p5 - 0.4) < 1e-9
    assert abs(r5 - 1.0) < 1e-9
    # hand calc: DCG@5 = 0.63093 (from A at rank2) + 1/log2(5) (C at rank4) = 0.63093+0.43068 = 1.06161
    #            IDCG@5 = same as IDCG@3 (only 2 relevant chunks exist): 1.63093
    #            NDCG@5 = 1.06161 / 1.63093 = 0.65094
    assert abs(n5 - 0.65094) < 1e-4

    # edge case: no relevant chunks at all for this query
    assert precision_at_k(retrieved, set(), 3) == 0.0
    assert recall_at_k(retrieved, set(), 3) == 0.0
    assert ndcg_at_k(retrieved, set(), 3) == 0.0

    # edge case: perfect ranking (all relevant chunks ranked first) -> NDCG should be exactly 1.0
    perfect_retrieved = ["A", "C", "B", "D", "E"]
    assert abs(ndcg_at_k(perfect_retrieved, relevant, 5) - 1.0) < 1e-9

    print("\nAll hand-verified sanity checks passed.")

    print("\n=== spot check against real gold_relevance.json ===")
    gold = load_gold_relevance()
    print(f"loaded gold relevance for {len(gold)} questions")
    fbid, chunk_set = next(iter(gold.items()))
    print(f"question {fbid}: {len(chunk_set)} gold chunks")
    # sanity: retrieving the actual gold chunks first should give perfect scores
    fake_retrieval = list(chunk_set) + ["not_a_real_chunk_1", "not_a_real_chunk_2"]
    scores = evaluate_ranking(fake_retrieval, chunk_set, ks=[5, 10])
    print("scores when gold chunks are ranked first (sanity check):", scores)
