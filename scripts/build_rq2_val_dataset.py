"""
Held-out counterpart to build_rq2_dataset.py (RQ2 train set, 56
questions). Mirrors its two-source structure exactly, but drawing from
data never touched during training, for a genuine same-distribution
held-out coverage-aware ranking evaluation:

1. val_precise_questions.json (40 multi-metric questions, built from 7
   metrics - goodwill, accounts payable, other assets, total equity,
   accrued liabilities, SG&A, total operating expenses - never used in
   the 591-question training set). Each hop maps to exactly 1 chunk, so
   coverage@k == recall@k here, same caveat as the train set's synthetic
   half.

2. Natural FinanceBench questions from the VAL split (untouched by any
   training) whose hop_groups have real redundancy.

3. Mined redundant-hop questions (mine_redundant_hops.py), restricted to
   the 7 held-out VAL_METRIC_LABELS so this stays a genuine unseen-metric
   test, not reused training metrics - see build_rq2_dataset.py and
   mine_redundant_hops.py for the mining/verification methodology.

Output: data/processed/rq2_val_dataset.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT_PATH = ROOT / "data" / "processed" / "rq2_val_dataset.json"

from mine_redundant_hops import VAL_METRIC_LABELS, mine_redundant_questions


def main():
    val_precise = json.load(open(ROOT / "data" / "processed" / "val_precise_questions.json"))
    natural_gold = json.load(open(ROOT / "data" / "processed" / "gold_relevance.json"))
    splits = json.load(open(ROOT / "data" / "processed" / "question_splits.json"))

    records = []
    for i, q in enumerate(val_precise, 1):
        hop_groups = {m: [info["chunk_id"]] for m, info in q["metrics"].items()}
        records.append({
            "id": q.get("val_precise_id", f"val_precise_multi_{i:03d}"),
            "question": q["question"],
            "source": "synthetic_multi_metric_val",
            "gold_chunk_ids": q["gold_chunk_ids"],
            "hop_groups": hop_groups,
            "has_redundant_hops": False,
        })

    natural_redundant_val = [
        r for r in natural_gold
        if len(r.get("hop_groups", {})) >= 2 and any(len(v) >= 2 for v in r["hop_groups"].values())
        and splits.get(r["financebench_id"]) == "val"
    ]
    for r in natural_redundant_val:
        records.append({
            "id": r["financebench_id"],
            "question": r["question"],
            "source": "natural_tightened_val",
            "gold_chunk_ids": r["gold_chunk_ids"],
            "hop_groups": r["hop_groups"],
            "has_redundant_hops": True,
        })

    mined_val = mine_redundant_questions(VAL_METRIC_LABELS, "redundant_mined_val")
    records.extend(mined_val)

    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    n_synth = sum(1 for r in records if r["source"] == "synthetic_multi_metric_val")
    n_nat = sum(1 for r in records if r["source"] == "natural_tightened_val")
    n_mined = sum(1 for r in records if r["source"] == "redundant_mined")
    print(f"{len(records)} total RQ2 VAL questions ({n_synth} synthetic-val, {n_nat} natural-redundant-val, {n_mined} mined-redundant-val)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
