"""
Builds (query, docs, labels) rows for cross-encoder fine-tuning with
RankNetLoss - one row per question, docs = positives + hard negatives,
labels = 1/0.

Uses our OWN precise question set (single-metric, multi-metric,
narrative - all built/selected this session with exact or verified-
clean gold) as the positive source, instead of the original 84 noisy
FinanceBench train questions - much better ground truth to train
against. Hard negatives: top-ranked NON-gold results from
pipeline_best.py's retrieve() (BM25+dense CC fusion, alpha=0.4, entity+
year filtered) - these are candidates that already fooled our current
best pipeline, the same "domain-informed hard negative" principle used
throughout this project.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from pipeline_best import retrieve

MAX_POSITIVES_PER_QUESTION = 6
NEGATIVES_PER_POSITIVE = 3  # richer contrastive signal per question than the previous ~1:1 ratio
RETRIEVE_K = 50
OUT_PATH = ROOT / "data" / "processed" / "finetune_pairwise.jsonl"


def _load_precise_questions() -> list[dict]:
    single = json.load(open(ROOT / "data" / "processed" / "precise_single_metric_questions.json"))
    multi = json.load(open(ROOT / "data" / "processed" / "precise_questions.json"))
    narrative = json.load(open(ROOT / "data" / "processed" / "precise_narrative_questions.json"))
    return (
        [{"question": q["question"], "gold_chunk_ids": q["gold_chunk_ids"]} for q in single]
        + [{"question": q["question"], "gold_chunk_ids": q["gold_chunk_ids"]} for q in multi]
        + [{"question": q["question"], "gold_chunk_ids": q["gold_chunk_ids"]} for q in narrative]
    )


def main():
    chunks = pd.read_parquet(ROOT / "data" / "processed" / "child_chunks.parquet")
    id_to_text = dict(zip(chunks["chunk_id"], chunks["raw_text"]))

    questions = _load_precise_questions()
    print(f"{len(questions)} precise questions (single+multi+narrative)", flush=True)

    random.seed(0)
    rows = []
    for i, q in enumerate(questions, 1):
        gold_ids = [cid for cid in q["gold_chunk_ids"] if cid in id_to_text]
        sampled_positives = random.sample(gold_ids, min(MAX_POSITIVES_PER_QUESTION, len(gold_ids)))
        if not sampled_positives:
            continue

        ranked = retrieve(q["question"], k=RETRIEVE_K)
        hard_negative_ids = [cid for cid in ranked if cid not in gold_ids and cid in id_to_text]
        n_negatives = min(len(sampled_positives) * NEGATIVES_PER_POSITIVE, len(hard_negative_ids))
        sampled_negatives = random.sample(hard_negative_ids, n_negatives) if n_negatives else []
        if not sampled_negatives:
            continue

        docs = [id_to_text[c] for c in sampled_positives] + [id_to_text[c] for c in sampled_negatives]
        labels = [1] * len(sampled_positives) + [0] * len(sampled_negatives)
        rows.append({"query": q["question"], "docs": docs, "labels": labels})

        if i % 50 == 0 or i == len(questions):
            print(f"[{i}/{len(questions)}] {len(rows)} question-rows so far", flush=True)

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    total_docs = sum(len(r["docs"]) for r in rows)
    print(f"\nwrote {len(rows)} question-rows ({total_docs} total docs) to {OUT_PATH}")


if __name__ == "__main__":
    main()
