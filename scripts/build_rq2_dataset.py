"""
Combines three complementary sources into one RQ2 (coverage-aware
ranking) training/eval set:

1. Synthetic multi-metric precise questions (compose_precise_questions.py) -
   exact, zero-noise hop_groups, but each hop maps to exactly 1 chunk
   (no redundancy), so coverage@k == recall@k here mathematically.
   Good for clean, reliable training signal.

2. Natural FinanceBench questions (TRAIN split only - test/val untouched)
   whose hop_groups have real redundancy (2+ hops, at least one hop
   with 2+ restated chunks, like the AES ROA example: "net income"
   restated ~9 times across sections). This is where coverage@k
   actually diverges from recall@k and demonstrates the failure mode
   that motivates coverage-aware ranking in the first place.

3. Mined redundant-hop questions (mine_redundant_hops.py) - genuine,
   value-verified redundancy found directly in the corpus (earnings
   releases restated in 10-Ks, MD&A summary tables restating financial
   statements, narrative prose corroborating a table row), searched
   across ALL filing types, not just 10-Ks. This is the main source of
   volume for the redundant-hop tier (25 questions vs. 12 from natural
   FinanceBench alone) - see mine_redundant_hops.py's docstring for the
   verification discipline (exact value match, not just label match)
   and the false-positive it caught and fixed along the way.

Output: data/processed/rq2_dataset.json - list of
{question, source, gold_chunk_ids, hop_groups}
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT_PATH = ROOT / "data" / "processed" / "rq2_dataset.json"

from mine_redundant_hops import TRAIN_METRIC_LABELS, mine_redundant_questions


def main():
    synthetic = json.load(open(ROOT / "data" / "processed" / "precise_questions.json"))
    natural_gold = json.load(open(ROOT / "data" / "processed" / "gold_relevance.json"))
    splits = json.load(open(ROOT / "data" / "processed" / "question_splits.json"))

    records = []
    for q in synthetic:
        hop_groups = {m: [info["chunk_id"]] for m, info in q["metrics"].items()}
        records.append({
            "id": q["precise_id"],
            "question": q["question"],
            "source": "synthetic_multi_metric",
            "gold_chunk_ids": q["gold_chunk_ids"],
            "hop_groups": hop_groups,
            "has_redundant_hops": False,
        })

    natural_redundant = [
        r for r in natural_gold
        if len(r.get("hop_groups", {})) >= 2 and any(len(v) >= 2 for v in r["hop_groups"].values())
        and splits.get(r["financebench_id"]) == "train"
    ]
    for r in natural_redundant:
        records.append({
            "id": r["financebench_id"],
            "question": r["question"],
            "source": "natural_tightened",
            "gold_chunk_ids": r["gold_chunk_ids"],
            "hop_groups": r["hop_groups"],
            "has_redundant_hops": True,
        })

    mined = mine_redundant_questions(TRAIN_METRIC_LABELS, "redundant_mined_train")
    records.extend(mined)

    with open(OUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    n_synthetic = sum(1 for r in records if r["source"] == "synthetic_multi_metric")
    n_natural = sum(1 for r in records if r["source"] == "natural_tightened")
    n_mined = sum(1 for r in records if r["source"] == "redundant_mined")
    print(f"{len(records)} total RQ2 questions ({n_synthetic} synthetic, {n_natural} natural-redundant, {n_mined} mined-redundant, train-only)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
