"""
The final validated retrieval + reranking pipeline for this project.

Stage 1 (retrieve_fusion): BM25 + dense, each filtered to the detected
company entity and fiscal year(s), fused via convex combination
(alpha=0.4, min-max normalized scores per method).

Stage 2 (rerank, applied by default in retrieve()): cross-encoder
reranking of stage 1's top-k candidates, fine-tuned via RankNetLoss on
domain-informed hard negatives (see finetune_cross_encoder.py /
colab_finetune_cross_encoder.py and build_finetune_data.py). Blended
with the stage-1 ranking via beta:
    final_score = beta * reranker_score_norm + (1 - beta) * stage1_rank_score
beta=1.0 (full trust in the reranker) is the default here - validated
as the best choice specifically on same-distribution held-out data
(single/multi-metric factual-lookup questions: +64% P@5, +72% NDCG@5
on 185 genuinely unseen facts). NOTE: beta=1.0 was found to actively
HURT on cross-distribution data (natural FinanceBench's analytical/
computed-ratio questions: -21% P@5 on 30 held-out val questions) - see
SESSION_SUMMARY.md's RQ1 results section for the full beta sweep and
the reasoning behind this choice. If the pipeline is expected to serve
that question style too, beta=0.3 is the safer, dual-distribution-safe
choice (positive on both, just smaller on the single/multi-metric side).

Checkpoint used: data/processed/cross_encoder_checkpoints_v3/checkpoint-160
(final epoch of a 5-epoch RankNetLoss fine-tune on 591 question-rows /
3682 docs built from this project's own single-metric/multi-metric/
narrative precise question set - see build_finetune_data.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from sentence_transformers import CrossEncoder

from bm25_retrieve import retrieve_with_scores as bm25_scores
from dense_retrieve import retrieve_with_scores as dense_scores
from entity_detection import detect_entity
from year_detection import detect_years
from rrf_fusion import convex_combination_fusion
from mine_redundant_hops import TRAIN_METRIC_LABELS, VAL_METRIC_LABELS

_ALL_METRIC_LABELS = {**TRAIN_METRIC_LABELS, **VAL_METRIC_LABELS}
TOP_PER_SUBQUERY = 3  # validated via eval_query_decomposition.py: coverage@10 +20% held-out, 0 regressions

RERANKER_PATH = ROOT / "data" / "processed" / "cross_encoder_checkpoints_v3" / "checkpoint-160"
DEFAULT_BETA = 1.0  # see module docstring - dataset-dependent, 0.3 is the dual-distribution-safe alternative

_id_to_entity = None
_id_to_year = None
_id_to_text = None
_reranker = None


def _load_metadata():
    global _id_to_entity, _id_to_year, _id_to_text
    if _id_to_entity is None:
        chunks = pd.read_parquet(ROOT / "data" / "processed" / "child_chunks.parquet")
        _id_to_entity = dict(zip(chunks["chunk_id"], chunks["entity"]))
        _id_to_year = dict(zip(chunks["chunk_id"], chunks["fiscal_period"]))
        _id_to_text = dict(zip(chunks["chunk_id"], chunks["raw_text"]))
    return _id_to_entity, _id_to_year, _id_to_text


def _load_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(str(RERANKER_PATH))
    return _reranker


def retrieve_fusion(
    question_text: str,
    k: int = 50,
    use_entity_filter: bool = True,
    use_year_filter: bool = True,
    alpha: float = 0.4,
) -> list[str]:
    """Stage 1 only: BM25+dense fusion with entity/year filtering, no
    reranking. Kept as a standalone building block for comparison
    scripts that need the pre-reranking baseline."""
    id_to_entity, id_to_year, _ = _load_metadata()
    bm25_pairs = bm25_scores(question_text, k=k)
    dense_pairs = dense_scores(question_text, k=k)

    entity = detect_entity(question_text) if use_entity_filter else None
    years = detect_years(question_text) if use_year_filter else []

    def _filter(pairs):
        out = pairs
        if entity:
            out = [(c, s) for c, s in out if id_to_entity.get(c) == entity]
        if years:
            out = [(c, s) for c, s in out if id_to_year.get(c) in years]
        return out

    return convex_combination_fusion([_filter(bm25_pairs), _filter(dense_pairs)], alpha=alpha)


def rerank(question_text: str, candidates: list[str], beta: float = DEFAULT_BETA) -> list[str]:
    """Blends the cross-encoder reranker's score with stage 1's rank
    position: final_score = beta*reranker_norm + (1-beta)*stage1_rank.
    beta=1.0 ignores stage 1's order entirely (pure reranker order);
    beta=0.0 returns candidates unchanged."""
    if not candidates or beta <= 0.0:
        return candidates
    _, _, id_to_text = _load_metadata()
    model = _load_reranker()

    pairs = [(question_text, id_to_text.get(cid, "")) for cid in candidates]
    ce_scores = model.predict(pairs, show_progress_bar=False)
    if beta >= 1.0:
        return [cid for _, cid in sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)]

    n = len(candidates)
    orig_rank_map = {cid: (n - i) / n for i, cid in enumerate(candidates)}
    lo, hi = min(ce_scores), max(ce_scores)
    rng = (hi - lo) if hi > lo else 1.0
    ce_norm_map = {cid: (s - lo) / rng for cid, s in zip(candidates, ce_scores)}
    blended = {cid: beta * ce_norm_map[cid] + (1 - beta) * orig_rank_map[cid] for cid in candidates}
    return sorted(candidates, key=lambda c: blended[c], reverse=True)


def retrieve(
    question_text: str,
    k: int = 50,
    use_entity_filter: bool = True,
    use_year_filter: bool = True,
    alpha: float = 0.4,
    use_reranker: bool = True,
    beta: float = DEFAULT_BETA,
) -> list[str]:
    """Full pipeline: stage 1 fusion, then stage 2 reranking (on by
    default, beta=1.0). Set use_reranker=False to get the pre-rerank
    baseline via this same function signature."""
    candidates = retrieve_fusion(question_text, k=k, use_entity_filter=use_entity_filter, use_year_filter=use_year_filter, alpha=alpha)
    if not use_reranker:
        return candidates
    return rerank(question_text, candidates, beta=beta)


def detect_metrics(question_text: str) -> list[str]:
    """Which of the 24 known metric names (17 train + 7 held-out val,
    from mine_redundant_hops.py) appear in this question - used to
    decide whether a question is multi-metric and, if so, what to
    decompose it into. Reuses the same label lists the corpus mining
    and precise-question construction already use, rather than a new
    ad-hoc list, so detection stays consistent with how those metrics
    are actually phrased in this project's data."""
    q_lower = question_text.lower()
    return [name for name, (labels, _) in _ALL_METRIC_LABELS.items() if any(label in q_lower for label in labels)]


def retrieve_decomposed(question_text: str, k: int = 50, beta: float = DEFAULT_BETA, top_per_subquery: int = TOP_PER_SUBQUERY) -> list[str]:
    """Splits a multi-metric question into one single-metric sub-question
    per detected metric ("What was {entity}'s {metric} in FY{year}?"),
    retrieves+reranks each independently (full pipeline: stage-1 fusion
    + cross-encoder), then interleaves (round-robin, NOT concatenated
    block-by-block - see eval_query_decomposition.py's module notes on
    why that ordering matters) their top-N results into one combined
    ranking.

    Why this exists: a bundled multi-metric query dilutes BM25/dense
    term-overlap scoring for each individual metric against the OTHER
    metrics also named in the same query - a real, diagnosed failure
    mode (one metric's gold chunk missing from the top-100 entirely
    when bundled with 3 others, found at rank 1 when asked alone).
    Validated on held-out multi-metric questions, both arms through the
    full CE-reranked pipeline: coverage@10 0.806 -> 0.969 (+20.2%), 0
    questions regressed, 13/40 improved.

    Falls back to the ordinary retrieve() if fewer than 2 metrics are
    detected (nothing to decompose) or entity/year can't be detected
    (can't construct valid sub-questions)."""
    entity = detect_entity(question_text)
    years = detect_years(question_text)
    metrics = detect_metrics(question_text)
    if len(metrics) < 2 or not entity or not years:
        return retrieve(question_text, k=k, beta=beta)

    year = years[0]
    per_metric_ranked = []
    for metric in metrics:
        subq = f"What was {entity}'s {metric} in FY{year}?"
        ranked = retrieve(subq, k=k, beta=beta)
        per_metric_ranked.append(ranked[:top_per_subquery])

    combined = []
    for slot in range(top_per_subquery):
        for metric_ranked in per_metric_ranked:
            if slot < len(metric_ranked) and metric_ranked[slot] not in combined:
                combined.append(metric_ranked[slot])
    return combined


def retrieve_auto(question_text: str, k: int = 50, beta: float = DEFAULT_BETA) -> list[str]:
    """Recommended entry point: routes to retrieve_decomposed() when the
    question looks multi-metric (2+ detected metrics), otherwise the
    ordinary retrieve(). retrieve() itself is left unchanged (still the
    direct/bundled approach) so existing evaluation scripts that call it
    directly keep their exact prior behavior."""
    if len(detect_metrics(question_text)) >= 2:
        return retrieve_decomposed(question_text, k=k, beta=beta)
    return retrieve(question_text, k=k, beta=beta)


if __name__ == "__main__":
    import json

    from eval.metrics import evaluate_ranking, load_gold_relevance

    gold = load_gold_relevance()
    questions = [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]
    q = next(q for q in questions if q["financebench_id"] == "financebench_id_03029")
    gold_chunks = gold[q["financebench_id"]]

    print("QUESTION:", q["question"])
    print("gold chunks:", len(gold_chunks))

    before = retrieve_fusion(q["question"], k=50)
    after = retrieve(q["question"], k=50)
    print("\nbefore rerank scores:", evaluate_ranking(before, gold_chunks, ks=[5, 10, 20]))
    print("after rerank scores: ", evaluate_ranking(after, gold_chunks, ks=[5, 10, 20]))
