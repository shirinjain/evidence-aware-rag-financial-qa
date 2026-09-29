"""
Same construction method as compose_precise_questions.py, but using
ONLY metrics that were never used to build the 591-question training
set - genuinely new facts the reranker has never seen, giving an honest
same-distribution held-out check without needing to retrain.
"""
import json
import re
from pathlib import Path

import pandas as pd

from compose_precise_questions import _matches_statement_type, _label, _collect_facts, _load_annual_10k_docs

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
OUT_MULTI_PATH = ROOT / "data" / "processed" / "val_precise_questions.json"
OUT_SINGLE_PATH = ROOT / "data" / "processed" / "val_precise_single_metric_questions.json"
MAX_METRICS_PER_QUESTION = 4

# metrics NOT present in METRIC_LABELS (compose_precise_questions.py) -
# verified via direct coverage check before adding these
NEW_METRIC_LABELS = {
    "goodwill": (["goodwill"], "balance_sheet"),
    "accounts payable": (["accounts payable"], "balance_sheet"),
    "other assets": (["other assets"], "balance_sheet"),
    "total equity": (["total equity"], "balance_sheet"),
    "accrued liabilities": (["accrued liabilities"], "balance_sheet"),
    "selling, general and administrative expenses": (["selling, general and administrative expenses"], "income_statement"),
    "total operating expenses": (["total operating expenses"], "income_statement"),
}


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    table_rows = chunks[chunks["chunk_type"] == "table_row"].copy()
    table_rows["label"] = table_rows["raw_text"].apply(_label)
    table_rows["doc_name"] = table_rows["chunk_id"].str.split("__block").str[0]

    annual_10k_docs = _load_annual_10k_docs()
    table_rows = table_rows[table_rows["doc_name"].isin(annual_10k_docs)]

    facts_by_metric = {}
    for metric, (labels, statement_type) in NEW_METRIC_LABELS.items():
        valid_section = table_rows["section_title"].apply(lambda s: _matches_statement_type(s, statement_type))
        restricted = table_rows[valid_section]
        facts_by_metric[metric] = _collect_facts(restricted, labels)
    for metric, facts in facts_by_metric.items():
        print(f"{metric}: {len(facts)} (company, year) pairs")

    all_keys = set()
    for facts in facts_by_metric.values():
        all_keys.update(facts.keys())

    import random
    random.seed(1)  # different seed from training generation, though metrics don't overlap anyway
    single_questions = []
    multi_questions = []
    for company, year in sorted(all_keys):
        available = {m: facts[(company, year)] for m, facts in facts_by_metric.items() if (company, year) in facts}

        for metric, fact in available.items():
            single_questions.append({
                "question": f"What was {company}'s {metric} in FY{year}?",
                "company": company,
                "fiscal_period": year,
                "metric": metric,
                "value": fact["value"],
                "gold_chunk_ids": [fact["chunk_id"]],
            })

        metric_names = sorted(available.keys())
        if len(metric_names) >= 2:
            random.shuffle(metric_names)
            for start in range(0, len(metric_names), MAX_METRICS_PER_QUESTION):
                group = sorted(metric_names[start:start + MAX_METRICS_PER_QUESTION])
                if len(group) < 2:
                    continue
                metric_phrase = ", ".join(group[:-1]) + (" and " + group[-1] if len(group) > 1 else group[0])
                multi_questions.append({
                    "question": f"What was {company}'s {metric_phrase} in FY{year}?",
                    "company": company,
                    "fiscal_period": year,
                    "metrics": {m: {"value": available[m]["value"], "chunk_id": available[m]["chunk_id"]} for m in group},
                    "gold_chunk_ids": [available[m]["chunk_id"] for m in group],
                })

    for i, q in enumerate(single_questions, 1):
        q["val_precise_id"] = f"val_precise_single_{i:03d}"
    for i, q in enumerate(multi_questions, 1):
        q["val_precise_id"] = f"val_precise_multi_{i:03d}"

    with open(OUT_SINGLE_PATH, "w") as f:
        json.dump(single_questions, f, indent=2)
    with open(OUT_MULTI_PATH, "w") as f:
        json.dump(multi_questions, f, indent=2)

    print(f"\ngenerated {len(single_questions)} single-metric val questions -> {OUT_SINGLE_PATH}")
    print(f"generated {len(multi_questions)} multi-metric val questions -> {OUT_MULTI_PATH}")


if __name__ == "__main__":
    main()
