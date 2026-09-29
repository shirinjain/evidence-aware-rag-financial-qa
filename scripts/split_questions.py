"""
Stratified 60/20/20 train/val/test split across all three question
sources: the 150 natural FinanceBench questions (stratified on
question_type: metrics-generated/domain-relevant/novel-generated, 50
each) plus the 22 table-based and 7 narrative-based synthetic compound
questions (stratified on their own "type" field).

Same round-based sizing as the original evidence-aware-rag project's
split_questions.py (avoids floor/ceil rounding losing or duplicating a
question). financebench_id and synthetic_id are separate namespaces
(no collision risk), so all three sources can share one output file.

Honest caveat: the synthetic strata are small (13/9/7 questions per
type) - a 60/20/20 split leaves as few as 1 example in val or test for
some types. Printed explicitly below rather than hidden, same as the
original project's sparse-cross-company-cell caveat.

Output: data/processed/question_splits.json
    {question_id: "train" | "val" | "test"}
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NATURAL_QUESTIONS_PATH = ROOT / "data" / "raw" / "financebench_merged.jsonl"
SYNTHETIC_TABLE_PATH = ROOT / "data" / "processed" / "synthetic_2hop_questions.json"
SYNTHETIC_NARRATIVE_PATH = ROOT / "data" / "processed" / "synthetic_2hop_narrative_questions.json"
OUT_PATH = ROOT / "data" / "processed" / "question_splits.json"

SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
SEED = 42


def split_group(question_ids: list, rng: random.Random) -> dict:
    """60/20/20 split of one stratum, sized by count (not floor/ceil
    rounding that could drop or duplicate a question)."""
    ids = list(question_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_val = round(n * SPLIT_RATIOS["val"])
    assignment = {}
    for qid in ids[:n_train]:
        assignment[qid] = "train"
    for qid in ids[n_train:n_train + n_val]:
        assignment[qid] = "val"
    for qid in ids[n_train + n_val:]:
        assignment[qid] = "test"
    return assignment


def main():
    natural = [json.loads(l) for l in open(NATURAL_QUESTIONS_PATH)]
    synthetic_table = json.load(open(SYNTHETIC_TABLE_PATH))
    synthetic_narrative = json.load(open(SYNTHETIC_NARRATIVE_PATH))

    records = []
    for q in natural:
        records.append({"id": q["financebench_id"], "source": "natural", "stratify_key": q["question_type"]})
    for q in synthetic_table:
        records.append({"id": q["synthetic_id"], "source": "synthetic_table", "stratify_key": q["type"]})
    for q in synthetic_narrative:
        records.append({"id": q["synthetic_id"], "source": "synthetic_narrative", "stratify_key": q["type"]})

    rng = random.Random(SEED)
    by_stratum = defaultdict(list)
    for r in records:
        by_stratum[r["stratify_key"]].append(r["id"])

    assignment = {}
    for stratum, ids in by_stratum.items():
        assignment.update(split_group(ids, rng))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(assignment, indent=2))

    df = pd.DataFrame(records)
    df["split"] = df["id"].map(assignment)

    print("=== split sizes (all sources combined) ===")
    print(df["split"].value_counts())
    print()
    print("=== stratify_key balance per split ===")
    print(pd.crosstab(df["stratify_key"], df["split"], margins=True))
    print()
    print("=== source balance per split (transparency check) ===")
    print(pd.crosstab(df["source"], df["split"], margins=True))
    print()
    print("CAVEAT: synthetic strata are small (13/9/7 per type) - some")
    print("val/test cells above have only 1-2 questions. Treat synthetic")
    print("val/test metrics as illustrative, not statistically robust.")

    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
