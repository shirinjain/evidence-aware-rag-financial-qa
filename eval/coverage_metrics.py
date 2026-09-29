"""
coverage@k: fraction of a question's DISTINCT required facts (hop
groups), not raw gold chunks, that appear at least once in the top-k
ranking.

Motivation, concretely: the AES ROA question needs 2 distinct facts
(total assets, net income) but has 18 gold chunks total (~9 restated
copies of each). Precision@10 would score 1.0 for a ranking that
returns 10 different "net income" restatements and zero "total assets"
chunks - a ranking that's actually useless for answering the question,
since ROA can't be computed without both facts. coverage@k is designed
to catch exactly this blind spot, since standard NDCG/precision treat
every gold chunk as interchangeable regardless of which fact it
represents.
"""


def coverage_at_k(ranked_chunk_ids: list[str], hop_groups: dict[str, list[str]], k: int) -> float:
    """hop_groups: {named_item: [chunk_ids satisfying that fact]}.
    Returns the fraction of hop groups with >=1 member in the top-k."""
    if not hop_groups:
        return None
    top_k = set(ranked_chunk_ids[:k])
    covered = sum(1 for chunk_ids in hop_groups.values() if top_k & set(chunk_ids))
    return covered / len(hop_groups)


if __name__ == "__main__":
    # AES ROA-style example: 2 hops, 9 restated chunks each
    hop_groups = {
        "total assets": [f"ta_{i}" for i in range(9)],
        "net income attributable to the aes corporation": [f"ni_{i}" for i in range(9)],
    }

    # ranking A: 10 "net income" chunks, zero "total assets" - the
    # deceptive high-precision-but-useless case
    ranking_a = [f"ni_{i}" for i in range(9)] + ["unrelated_chunk"]
    # ranking B: 5 of each - genuinely covers both needed facts
    ranking_b = [f"ta_{i}" for i in range(5)] + [f"ni_{i}" for i in range(5)]

    for name, ranking in [("A (all net income, no total assets)", ranking_a), ("B (both facts covered)", ranking_b)]:
        gold_hits = len(set(ranking[:10]) & (set(hop_groups["total assets"]) | set(hop_groups["net income attributable to the aes corporation"])))
        precision_at_10 = gold_hits / 10
        coverage = coverage_at_k(ranking, hop_groups, k=10)
        print(f"ranking {name}: precision@10={precision_at_10:.2f}  coverage@10={coverage:.2f}")
