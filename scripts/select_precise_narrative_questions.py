"""
Companion to compose_precise_questions.py, but for narrative (prose)
content instead of tables. Prose doesn't have a clean "label | value"
structure to construct fresh questions from the way a table row does,
so instead of generating new questions, this SELECTS real FinanceBench
questions whose (already-cleaned) gold set is small and narrative-
dominant - a real annotator already wrote a focused question and cited
1-2 specific paragraphs, so these are naturally precise once alignment
is clean; no new construction needed, just verified selection.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
GOLD_PATH = ROOT / "data" / "processed" / "gold_relevance.json"
QUESTIONS_PATH = ROOT / "data" / "raw" / "financebench_merged.jsonl"
SPLITS_PATH = ROOT / "data" / "processed" / "question_splits.json"
OUT_PATH = ROOT / "data" / "processed" / "precise_narrative_questions.json"

MAX_GOLD_CHUNKS = 5
MIN_NARRATIVE_FRAC = 0.8


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    chunk_type_map = dict(zip(chunks["chunk_id"], chunks["chunk_type"]))
    gold_records = {r["financebench_id"]: r for r in json.load(open(GOLD_PATH))}
    questions = {q["financebench_id"]: q for q in [json.loads(l) for l in open(QUESTIONS_PATH)]}
    splits = json.load(open(SPLITS_PATH))

    selected = []
    for fbid, record in gold_records.items():
        # train split only - this file feeds cross-encoder fine-tuning
        # data, so val/test must never appear here (caught a real bug:
        # 3 val + 3 test questions were leaking into training data
        # before this filter existed)
        if splits.get(fbid) != "train":
            continue
        gold_ids = record["gold_chunk_ids"]
        if not gold_ids or len(gold_ids) > MAX_GOLD_CHUNKS:
            continue
        types = [chunk_type_map.get(cid) for cid in gold_ids]
        if types.count("narrative") / len(types) < MIN_NARRATIVE_FRAC:
            continue
        q = questions[fbid]
        selected.append({
            "financebench_id": fbid,
            "question": q["question"],
            "answer": q["answer"],
            "company": q["company"],
            "gold_chunk_ids": gold_ids,
            "gold_tightened": record.get("gold_tightened", False),
        })

    with open(OUT_PATH, "w") as f:
        json.dump(selected, f, indent=2)

    n_gold = [len(q["gold_chunk_ids"]) for q in selected]
    print(f"selected {len(selected)} precise narrative questions -> {OUT_PATH}")
    if n_gold:
        print(f"gold chunks per question: min={min(n_gold)} max={max(n_gold)} mean={sum(n_gold)/len(n_gold):.1f}")


if __name__ == "__main__":
    main()
