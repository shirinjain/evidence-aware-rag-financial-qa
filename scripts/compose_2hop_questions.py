"""
Composes synthetic 2-hop questions by pairing two atomic facts (each
from a different company) - either two revenue facts (comparison
framing) or a revenue+expense pair (additive framing).

Since we construct these ourselves, we know exactly which chunk_id
answers which half - giving free, exact hop_groups (no fuzzy alignment
needed, unlike the natural 150-question set).

This is a SYNTHETIC layer on top of a real single-hop benchmark
(FinanceBench), not a naturally-occurring multi-hop benchmark - must be
disclosed as such in any writeup, same as agreed earlier.
"""

from __future__ import annotations
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
OUT_PATH = ROOT / "data" / "processed" / "synthetic_2hop_questions.json"

N_QUESTIONS = 40

# end-anchored ($) so a label like "Net revenues generated from the
# U.K." (a geographic revenue-breakdown row, caught as a false positive
# during testing - it starts with "Net revenues" but isn't the total
# revenue line) doesn't match just because it shares a prefix
REVENUE_PATTERNS = [r"^total revenues?$", r"^total net (?:revenue|sales)s?$", r"^net sales$", r"^net revenues?$"]
EXPENSE_PATTERNS = [
    r"^total costs? and expenses?$",
    r"^total operating expenses?$",
    r"^total cost of revenues?$",
    r"^total noninterest expense$",
]


def _label(raw_text: str) -> str:
    label = raw_text.split("|")[0].strip().lower()
    # strip trailing annotations like "(millions)", "(unaudited)" so
    # "Net sales (millions)" still exact-matches "net sales"
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def _first_value(raw_text: str) -> str | None:
    nums = re.findall(r"\$?\s?-?\(?[\d,]{2,}(?:\.\d+)?\)?", raw_text.split("|", 1)[1]) if "|" in raw_text else []
    return nums[0].strip() if nums else None


def _doc_name_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("__block")[0]


def _load_annual_10k_docs() -> set[str]:
    """Documents that are actual annual 10-Ks - excludes 10-Q/8-K/
    earnings releases, which report partial-period figures that would
    be wrong if labeled as "FY{year}" (found via MGM's 10-Q Q2 revenue
    getting mislabeled as its full FY2023 total)."""
    rows = [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]
    return {r["doc_name"] for r in rows if r["doc_type"].lower() == "10k"}


def _collect_facts(chunks: pd.DataFrame, patterns: list[str], annual_10k_docs: set[str]) -> dict[tuple[str, int], dict]:
    table_rows = chunks[chunks["chunk_type"] == "table_row"].copy()
    table_rows["doc_name"] = table_rows["chunk_id"].apply(_doc_name_from_chunk_id)
    facts = {}
    for _, row in table_rows.iterrows():
        if row["doc_name"] not in annual_10k_docs:
            continue
        label = _label(row["raw_text"])
        if not any(re.match(p, label) for p in patterns):
            continue
        section = (row["section_title"] or "").lower()
        # only trust rows from the actual primary consolidated
        # statement - same heading pattern used earlier to locate
        # income-statement pages. Rejects: quarterly-breakdown tables
        # (Netflix: $1.82B single-quarter figure mislabeled as annual)
        # and geographic/business segment sub-totals (Nike: "NORTH
        # AMERICA" segment's $17.2B mislabeled as the ~$44.5B global
        # total) - both reuse the same line-item label outside the
        # actual consolidated statement.
        if "consolidated statement" not in section:
            continue
        value = _first_value(row["raw_text"])
        if value is None:
            continue
        key = (row["entity"], row["fiscal_period"])
        if key not in facts:  # keep first qualifying occurrence per company/year
            facts[key] = {"chunk_id": row["chunk_id"], "raw_text": row["raw_text"], "value": value}
    return facts


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    annual_10k_docs = _load_annual_10k_docs()
    revenue_facts = _collect_facts(chunks, REVENUE_PATTERNS, annual_10k_docs)
    expense_facts = _collect_facts(chunks, EXPENSE_PATTERNS, annual_10k_docs)

    print(f"revenue facts: {len(revenue_facts)} (company, year) pairs")
    print(f"expense facts: {len(expense_facts)} (company, year) pairs")

    random.seed(0)
    revenue_keys = list(revenue_facts.keys())
    expense_keys = list(expense_facts.keys())

    questions = []
    # type 1: revenue vs revenue, two different companies (comparison framing)
    random.shuffle(revenue_keys)
    for (company_a, year_a), (company_b, year_b) in zip(revenue_keys[0::2], revenue_keys[1::2]):
        if company_a == company_b or len(questions) >= N_QUESTIONS // 2:
            continue
        fact_a, fact_b = revenue_facts[(company_a, year_a)], revenue_facts[(company_b, year_b)]
        question = f"What was {company_a}'s total revenue in FY{year_a}, and what was {company_b}'s total revenue in FY{year_b}?"
        questions.append({
            "question": question,
            "type": "revenue_vs_revenue",
            "hop_groups": {
                f"{company_a} FY{year_a} revenue": [fact_a["chunk_id"]],
                f"{company_b} FY{year_b} revenue": [fact_b["chunk_id"]],
            },
            "hop_facts": [
                {"company": company_a, "year": year_a, "metric": "revenue", "value": fact_a["value"], "raw_text": fact_a["raw_text"]},
                {"company": company_b, "year": year_b, "metric": "revenue", "value": fact_b["value"], "raw_text": fact_b["raw_text"]},
            ],
        })

    # type 2: revenue + expense, two different companies (additive framing)
    random.shuffle(expense_keys)
    for (exp_company, exp_year) in expense_keys:
        if len(questions) >= N_QUESTIONS:
            break
        candidates = [k for k in revenue_keys if k[0] != exp_company]
        if not candidates:
            continue
        rev_company, rev_year = random.choice(candidates)
        fact_rev, fact_exp = revenue_facts[(rev_company, rev_year)], expense_facts[(exp_company, exp_year)]
        question = f"What was {rev_company}'s total revenue in FY{rev_year}, and what was {exp_company}'s total operating expense in FY{exp_year}?"
        questions.append({
            "question": question,
            "type": "revenue_vs_expense",
            "hop_groups": {
                f"{rev_company} FY{rev_year} revenue": [fact_rev["chunk_id"]],
                f"{exp_company} FY{exp_year} expense": [fact_exp["chunk_id"]],
            },
            "hop_facts": [
                {"company": rev_company, "year": rev_year, "metric": "revenue", "value": fact_rev["value"], "raw_text": fact_rev["raw_text"]},
                {"company": exp_company, "year": exp_year, "metric": "expense", "value": fact_exp["value"], "raw_text": fact_exp["raw_text"]},
            ],
        })

    for i, q in enumerate(questions, 1):
        q["synthetic_id"] = f"synthetic_2hop_{i:03d}"

    with open(OUT_PATH, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\ngenerated {len(questions)} synthetic 2-hop questions -> {OUT_PATH}")
    by_type = pd.Series([q["type"] for q in questions]).value_counts()
    print(by_type.to_string())


if __name__ == "__main__":
    main()
