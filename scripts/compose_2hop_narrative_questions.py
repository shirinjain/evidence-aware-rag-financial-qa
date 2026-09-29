"""
Composes synthetic 2-hop questions from PAIRS of existing FinanceBench
narrative-answer questions (different companies each), instead of
extracting new facts from tables - reuses the gold_chunk_ids we already
verified via align_gold_evidence.py, rather than a new fact-extraction
pipeline (which is what caused the quarterly-table/segment-table bugs
in the table-based composer).

Candidate pool: domain-relevant/novel-generated questions whose gold
evidence is >80% narrative chunks and small (<=5 gold chunks) - both
conditions picked to keep each half a clean, well-scoped single fact
rather than a sprawling multi-paragraph justification.
"""
import json
import random
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
GOLD_PATH = ROOT / "data" / "processed" / "gold_relevance.json"
QUESTIONS_PATH = ROOT / "data" / "raw" / "financebench_merged.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "synthetic_2hop_narrative_questions.json"

MAX_GOLD_CHUNKS = 5
MIN_NARRATIVE_FRAC = 0.8


def _find_candidates():
    chunks = pd.read_parquet(CHUNKS_PATH)
    chunk_type_map = dict(zip(chunks["chunk_id"], chunks["chunk_type"]))
    results = {r["financebench_id"]: r for r in json.load(open(GOLD_PATH))}
    questions = [json.loads(l) for l in open(QUESTIONS_PATH)]

    candidates = []
    for q in questions:
        if q["question_type"] not in ("domain-relevant", "novel-generated"):
            continue
        gold_ids = results.get(q["financebench_id"], {}).get("gold_chunk_ids", [])
        if not gold_ids or len(gold_ids) > MAX_GOLD_CHUNKS:
            continue
        types = [chunk_type_map.get(cid) for cid in gold_ids]
        if types.count("narrative") / len(types) >= MIN_NARRATIVE_FRAC:
            candidates.append({"company": q["company"], "question": q["question"], "gold_chunk_ids": gold_ids})
    return candidates


def main():
    candidates = _find_candidates()
    print(f"{len(candidates)} narrative single-hop candidates found")

    random.seed(0)
    pairs = [(a, b) for a, b in combinations(candidates, 2) if a["company"] != b["company"]]
    random.shuffle(pairs)

    questions = []
    used_companies = set()
    for a, b in pairs:
        # avoid reusing the same company across multiple synthetic
        # questions where possible, to keep the small pool diverse
        if a["company"] in used_companies and b["company"] in used_companies:
            continue
        q_a = a["question"].rstrip("?")
        q_b = b["question"][0].lower() + b["question"][1:]
        composed = f"{q_a}, and {q_b}"
        questions.append({
            "question": composed,
            "type": "narrative_2hop",
            "hop_groups": {
                f"{a['company']}: {a['question']}": a["gold_chunk_ids"],
                f"{b['company']}: {b['question']}": b["gold_chunk_ids"],
            },
        })
        used_companies.update([a["company"], b["company"]])

    for i, q in enumerate(questions, 1):
        q["synthetic_id"] = f"synthetic_narrative_2hop_{i:03d}"

    with open(OUT_PATH, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"generated {len(questions)} narrative synthetic 2-hop questions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
