"""
Safety-net evaluation for cross-encoder checkpoints: never adopt a
reranker without first confirming it beats the no-rerank baseline on
the actual downstream task. A converging loss curve does NOT guarantee
this (verified directly: a previous checkpoint had a normal-looking
loss curve but still made retrieval worse, with inverted score
preferences on spot-check) - only measuring real P@5/NDCG@5 does.

For each checkpoint found under CHECKPOINTS_DIR:
  1. Rerank pipeline_best.py's top-50 candidates, two ways:
     a. Pure reranker order (100% reranker score)
     b. Blended: beta*reranker_score + (1-beta)*original_fusion_rank_score,
        swept over a few beta values - a low beta limits how much a
        noisy/unreliable reranker can hurt the ranking even if it's
        not fully trustworthy, while still letting it help when it's
        confident and correct.
  2. Compare against the no-rerank baseline on the SAME question set.

Only report a checkpoint as "adopt" if it beats baseline at some beta -
never silently deploy a reranker that doesn't demonstrably help.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from sentence_transformers import CrossEncoder

from pipeline_best import retrieve
from eval.metrics import evaluate_ranking, load_gold_relevance

CONTAMINATED_FBIDS = {
    "financebench_id_00995", "financebench_id_00917", "financebench_id_00494",
    "financebench_id_01474", "financebench_id_00283", "financebench_id_00603",
}
BETAS = [0.3, 0.5, 0.7, 1.0]  # 1.0 = pure reranker order, no blending


def _rank_score_map(chunk_ids: list[str]) -> dict[str, float]:
    """Converts a ranked list into a [0,1] score by rank position, so
    it can be blended with the reranker's score on a comparable scale."""
    n = len(chunk_ids)
    return {cid: (n - i) / n for i, cid in enumerate(chunk_ids)}


def evaluate_checkpoint(checkpoint_path: str, eval_questions: list[dict], id_to_text: dict) -> dict:
    model = CrossEncoder(checkpoint_path)
    results_by_beta = {beta: [] for beta in BETAS}
    baseline_results = []

    for q in eval_questions:
        gold_ids = set(q["gold_chunk_ids"])
        candidates = retrieve(q["question"], k=50)
        baseline_results.append(evaluate_ranking(candidates, gold_ids, ks=[5, 10, 20]))

        if not candidates:
            # entity+year filtering can occasionally leave zero
            # candidates for a question - nothing to rerank, and the
            # baseline score above already correctly reflects 0 for it
            for beta in BETAS:
                results_by_beta[beta].append(evaluate_ranking([], gold_ids, ks=[5, 10, 20]))
            continue

        pairs = [(q["question"], id_to_text.get(cid, "")) for cid in candidates]
        ce_scores = model.predict(pairs, show_progress_bar=False)
        ce_score_map = dict(zip(candidates, ce_scores))
        orig_rank_map = _rank_score_map(candidates)

        # normalize ce scores to [0,1] within this candidate set before blending
        lo, hi = min(ce_scores), max(ce_scores)
        rng = (hi - lo) if hi > lo else 1.0
        ce_norm_map = {cid: (s - lo) / rng for cid, s in ce_score_map.items()}

        for beta in BETAS:
            blended = {cid: beta * ce_norm_map[cid] + (1 - beta) * orig_rank_map[cid] for cid in candidates}
            reranked = sorted(candidates, key=lambda c: blended[c], reverse=True)
            results_by_beta[beta].append(evaluate_ranking(reranked, gold_ids, ks=[5, 10, 20]))

    summary = {"baseline": pd.DataFrame(baseline_results).mean().to_dict()}
    for beta in BETAS:
        summary[f"beta={beta}"] = pd.DataFrame(results_by_beta[beta]).mean().to_dict()
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", help="Directory containing one or more checkpoint-N subfolders (or the model dir itself)")
    args = parser.parse_args()

    chunks = pd.read_parquet(ROOT / "data" / "processed" / "child_chunks.parquet")
    id_to_text = dict(zip(chunks["chunk_id"], chunks["raw_text"]))

    splits = json.load(open(ROOT / "data" / "processed" / "question_splits.json"))
    gold = load_gold_relevance()
    natural_qs = {q["financebench_id"]: q for q in [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]}

    # clean train-only eval set, excluding any question that could have
    # touched training data - never evaluate a reranker on data it or
    # any earlier contaminated version may have seen
    eval_questions = [
        {"question": natural_qs[qid]["question"], "gold_chunk_ids": list(gold[qid])}
        for qid, split in splits.items()
        if split == "train" and qid in natural_qs and len(gold.get(qid, [])) > 0 and qid not in CONTAMINATED_FBIDS
    ]
    print(f"evaluating on {len(eval_questions)} clean train questions", flush=True)

    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoint_paths = sorted(checkpoints_dir.glob("checkpoint-*")) or [checkpoints_dir]

    for ckpt in checkpoint_paths:
        print(f"\n=== {ckpt.name} ===", flush=True)
        summary = evaluate_checkpoint(str(ckpt), eval_questions, id_to_text)
        for config, scores in summary.items():
            print(f"  {config}: P@5={scores['precision@5']:.4f} NDCG@5={scores['ndcg@5']:.4f}")


if __name__ == "__main__":
    main()
