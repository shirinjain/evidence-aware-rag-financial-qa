"""
Builds precise questions with EXACT, small (2-5 chunk) gold sets by
construction - instead of aligning gold to FinanceBench's own recorded
evidence (which, after 3+ rounds of cleanup, still has residual noise
for ~110/150 questions), we pick the chunk FIRST and write the question
around it, so gold is exact by definition, no fuzzy matching involved.

Same validated extraction method as compose_2hop_questions.py: 10-K
documents only, rows restricted to sections literally titled
"Consolidated Statement(s) of ..." or "Consolidated Balance Sheet(s)"
(the two real, distinct heading conventions used for the income
statement and balance sheet respectively - checked directly, since
"consolidated statement" alone misses balance sheets, which don't use
the word "Statement" in their heading).

Each question asks for ALL metrics available for one (company, year)
pair in a single sentence - gold_chunk_ids is exactly one chunk per
metric asked about, so a 4-metric question has exactly 4 gold chunks,
a 2-metric question has exactly 2, etc. No inflation possible.
"""

from __future__ import annotations
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
QUESTIONS_PATH = ROOT / "data" / "raw" / "financebench_merged.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "precise_questions.json"
MAX_METRICS_PER_QUESTION = 4

# each metric needs a SPECIFIC section-heading pattern, not the generic
# "consolidated statement" substring - found via a real bug: a 3M "Net
# income" row got pulled from the "Consolidated Statement of Changes in
# Equity" (an equity-allocation breakdown, not the income statement),
# because that heading also contains "consolidated statement". Income
# statement and balance sheet are structurally distinct statements with
# distinct heading conventions and must be matched separately.
METRIC_LABELS = {
    "revenue": (["net sales", "total revenues", "net revenues", "total net sales", "total net revenue"], "income_statement"),
    "net income": (["net income"], "income_statement"),
    "gross profit": (["gross profit"], "income_statement"),
    "operating income": (["operating income"], "income_statement"),
    "cost of sales": (["cost of sales", "cost of goods sold"], "income_statement"),
    "income tax expense": (["income tax expense"], "income_statement"),
    "total assets": (["total assets"], "balance_sheet"),
    "total liabilities": (["total liabilities"], "balance_sheet"),
    "cash and cash equivalents": (["cash and cash equivalents"], "balance_sheet"),
    "total current assets": (["total current assets"], "balance_sheet"),
    "total current liabilities": (["total current liabilities"], "balance_sheet"),
    "long-term debt": (["long-term debt"], "balance_sheet"),
    "inventories": (["inventories"], "balance_sheet"),
    "capital expenditures": (["purchases of property, plant and equipment", "capital expenditures"], "cash_flow_statement"),
    "operating cash flow": (["net cash provided by operating activities"], "cash_flow_statement"),
    "depreciation and amortization": (["depreciation and amortization"], "cash_flow_statement"),
    "dividends paid": (["dividends paid"], "cash_flow_statement"),
}

def _matches_statement_type(section_title: str, statement_type: str) -> bool:
    """Whitespace-stripped match (Docling sometimes splits a word across
    a line break mid-token, e.g. "Consolidated Statement of Incom e" for
    "Income" - a literal substring/regex match on the raw text misses
    this, same issue found earlier when locating income-statement pages
    from raw PDF text). "Comprehensive Income" is a distinct statement
    from the main income statement (reports other comprehensive income,
    not net income) and must be excluded explicitly, not just matched
    loosely on "income"."""
    stripped = re.sub(r"\s+", "", (section_title or "").lower())
    if statement_type == "income_statement":
        if "comprehensiveincom" in stripped:
            return False
        return "consolidatedstatement" in stripped and ("ofincom" in stripped or "ofoperations" in stripped)
    if statement_type == "balance_sheet":
        return "consolidatedbalancesheet" in stripped
    if statement_type == "cash_flow_statement":
        return "consolidatedstatement" in stripped and "ofcashflow" in stripped
    return False


def _label(raw_text: str) -> str:
    label = raw_text.split("|")[0].strip().lower()
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def _first_value(raw_text: str) -> str | None:
    nums = re.findall(r"\$?\s?-?\(?[\d,]{2,}(?:\.\d+)?\)?", raw_text.split("|", 1)[1]) if "|" in raw_text else []
    return nums[0].strip() if nums else None


def _load_annual_10k_docs() -> set[str]:
    rows = [json.loads(l) for l in open(QUESTIONS_PATH)]
    return {r["doc_name"] for r in rows if r["doc_type"].lower() == "10k"}


def _collect_facts(table_rows: pd.DataFrame, labels: list[str]) -> dict[tuple[str, int], dict]:
    facts = {}
    for _, row in table_rows.iterrows():
        if row["label"] not in labels:
            continue
        value = _first_value(row["raw_text"])
        if value is None:
            continue
        key = (row["entity"], row["fiscal_period"])
        if key not in facts:
            facts[key] = {"chunk_id": row["chunk_id"], "raw_text": row["raw_text"], "value": value}
    return facts


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    table_rows = chunks[chunks["chunk_type"] == "table_row"].copy()
    table_rows["label"] = table_rows["raw_text"].apply(_label)
    table_rows["doc_name"] = table_rows["chunk_id"].str.split("__block").str[0]

    annual_10k_docs = _load_annual_10k_docs()
    table_rows = table_rows[table_rows["doc_name"].isin(annual_10k_docs)]

    facts_by_metric = {}
    for metric, (labels, statement_type) in METRIC_LABELS.items():
        valid_section = table_rows["section_title"].apply(lambda s: _matches_statement_type(s, statement_type))
        restricted = table_rows[valid_section]
        facts_by_metric[metric] = _collect_facts(restricted, labels)
    for metric, facts in facts_by_metric.items():
        print(f"{metric}: {len(facts)} (company, year) pairs")

    all_keys = set()
    for facts in facts_by_metric.values():
        all_keys.update(facts.keys())

    random.seed(0)
    single_questions = []
    multi_questions = []
    for company, year in sorted(all_keys):
        available = {m: facts[(company, year)] for m, facts in facts_by_metric.items() if (company, year) in facts}

        # single-metric questions - one per available metric, gold=1
        # chunk. This is the "3M capex" pattern: a sharp, unambiguous
        # single-fact lookup, which scored 0.8 precision in the natural
        # FinanceBench set - reproducing that pattern deliberately here.
        for metric, fact in available.items():
            single_questions.append({
                "question": f"What was {company}'s {metric} in FY{year}?",
                "company": company,
                "fiscal_period": year,
                "metric": metric,
                "value": fact["value"],
                "gold_chunk_ids": [fact["chunk_id"]],
            })

        # multi-metric compound questions - grouped into chunks of at
        # most MAX_METRICS_PER_QUESTION so each question stays a
        # realistic thing someone would actually ask (bundling all 17
        # available metrics into one sentence, as an earlier version of
        # this script did, produces an absurd, unnatural question - a
        # company/year with many available metrics instead gets several
        # separate, reasonably-sized compound questions)
        metric_names = sorted(available.keys())
        if len(metric_names) >= 2:
            random.shuffle(metric_names)
            for start in range(0, len(metric_names), MAX_METRICS_PER_QUESTION):
                group = sorted(metric_names[start:start + MAX_METRICS_PER_QUESTION])
                if len(group) < 2:
                    continue  # a lone leftover metric isn't worth a compound question
                metric_phrase = ", ".join(group[:-1]) + (" and " + group[-1] if len(group) > 1 else group[0])
                multi_questions.append({
                    "question": f"What was {company}'s {metric_phrase} in FY{year}?",
                    "company": company,
                    "fiscal_period": year,
                    "metrics": {m: {"value": available[m]["value"], "chunk_id": available[m]["chunk_id"]} for m in group},
                    "gold_chunk_ids": [available[m]["chunk_id"] for m in group],
                })

    for i, q in enumerate(single_questions, 1):
        q["precise_id"] = f"precise_single_{i:03d}"
    for i, q in enumerate(multi_questions, 1):
        q["precise_id"] = f"precise_multi_{i:03d}"

    with open(OUT_PATH, "w") as f:
        json.dump(multi_questions, f, indent=2)
    single_out_path = OUT_PATH.parent / "precise_single_metric_questions.json"
    with open(single_out_path, "w") as f:
        json.dump(single_questions, f, indent=2)

    n_gold_multi = [len(q["gold_chunk_ids"]) for q in multi_questions]
    print(f"\ngenerated {len(multi_questions)} multi-metric questions -> {OUT_PATH}")
    if n_gold_multi:
        print(f"  gold chunks per question: min={min(n_gold_multi)} max={max(n_gold_multi)} mean={sum(n_gold_multi)/len(n_gold_multi):.1f}")
    print(f"generated {len(single_questions)} single-metric questions -> {single_out_path}")
    print("  gold chunks per question: 1 (always, by construction)")


if __name__ == "__main__":
    main()
