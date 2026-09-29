"""
Reciprocal Rank Fusion and Convex Combination fusion of two ranked
lists - same formulas the original project used, and independently
confirmed by arxiv 2604.01733's ablation (RRF k=60 got Recall@5=0.695
vs CC alpha=0.5's 0.726 on their benchmark - testing both here rather
than assuming which wins on ours).
"""
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k_const: int = 60) -> list[str]:
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] += 1.0 / (k_const + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def convex_combination_fusion(ranked_lists_with_scores: list[list[tuple]], alpha: float = 0.5) -> list[str]:
    """ranked_lists_with_scores: list of [(chunk_id, score), ...] pairs,
    one list per method - scores are min-max normalized to [0,1] within
    each method before combining (so BM25's and cosine similarity's
    different scales don't just let one method dominate)."""
    normalized = []
    for pairs in ranked_lists_with_scores:
        if not pairs:
            normalized.append({})
            continue
        scores = [s for _, s in pairs]
        lo, hi = min(scores), max(scores)
        rng = hi - lo if hi > lo else 1.0
        normalized.append({cid: (s - lo) / rng for cid, s in pairs})

    weights = [alpha, 1 - alpha] if len(normalized) == 2 else [1 / len(normalized)] * len(normalized)
    combined = defaultdict(float)
    for norm_scores, w in zip(normalized, weights):
        for cid, s in norm_scores.items():
            combined[cid] += w * s
    return sorted(combined, key=combined.get, reverse=True)
