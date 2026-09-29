"""
Mines GENUINE redundant hop groups from the corpus - cases where the
same fact (same company, same fiscal period, same metric, same VALUE)
is reported in more than one chunk. This is real, naturally-occurring
redundancy (an earnings-release table restated in the subsequent 10-K,
an MD&A "Selected Financial Data" summary table restating the main
financial statements, a number stated in prose AND in a table), not
fabricated - every group here is verified by exact value match, the
same discipline used for RQ1's gold-alignment cleanup.

Two things this deliberately does NOT do (found the hard way, see
below), to avoid the exact false-positive trap that made an earlier,
naive version of this audit useless:

1. Never treats "same label" as "same fact" - e.g. a "net sales" row
   worth $14,694 and one worth $377 both match the label but are
   clearly different line items (segment vs total). Redundancy is only
   claimed when the extracted VALUE also matches (sign-agnostic).

2. Fixes a real bug in the value-extraction regex used elsewhere in
   this project (compose_precise_questions.py::_first_value): matching
   "[\\d,]{2,}" against "$ 0.382" incorrectly extracts "382" (or, with
   a naive lookbehind patch, "82") by grabbing digits AFTER the decimal
   point, silently corrupting any row whose value is a small decimal
   (per-share/EPS rows sharing a label like "Net income"). Fixed here
   via a negative lookbehind that refuses to start a match adjacent to
   an existing digit or decimal point, so "0.382" is captured whole
   and never collides with an unrelated "382" or "487" elsewhere.

Search space is deliberately widened past compose_precise_questions.py's
10-K-only restriction to ALL filing types (10-K, 10-Q, 8-K, earnings
releases) and past table-only matching to include narrative-vs-table
cross-matches (the same number stated in prose AND in a table row) -
this is what took genuine matches from ~19 to ~55+ for the train
metrics and ~2 to ~10 for the held-out val metrics.

Output schema matches rq2_dataset.json / rq2_val_dataset.json exactly:
{id, question, source, gold_chunk_ids, hop_groups, has_redundant_hops}
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"

MAX_METRICS_PER_QUESTION = 4

TRAIN_METRIC_LABELS = {
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

VAL_METRIC_LABELS = {
    "goodwill": (["goodwill"], "balance_sheet"),
    "accounts payable": (["accounts payable"], "balance_sheet"),
    "other assets": (["other assets"], "balance_sheet"),
    "total equity": (["total equity"], "balance_sheet"),
    "accrued liabilities": (["accrued liabilities"], "balance_sheet"),
    "selling, general and administrative expenses": (["selling, general and administrative expenses"], "income_statement"),
    "total operating expenses": (["total operating expenses"], "income_statement"),
}

# known false positive caught by manual inspection: an EPS-calculation
# row mislabeled "Net income" (reports per-share values like $0.382,
# not the aggregate figure) that happens to collide in value between
# its own basic/diluted columns. Excluded upstream, before either
# mining function runs, so it can't reappear via table-table OR
# narrative-table matching independently (it did the latter on a first
# pass - narrative matching doesn't do pairwise grouping, so it isn't
# covered by a pair-level exclusion applied only to the table-table path).
EXCLUDE_CHUNK_IDS = {
    "AMCOR_2020_10K__block298__row23",
    "AMCOR_2020_10K__block298__row27",
}


def _label(raw_text: str) -> str:
    label = raw_text.split("|")[0].strip().lower()
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def _first_value(raw_text: str) -> str | None:
    body = raw_text.split("|", 1)[1] if "|" in raw_text else ""
    # negative lookbehind: never start a match adjacent to a digit or
    # "." already consumed by a previous match attempt, so "0.382"
    # extracts whole instead of leaking "382" or "82" from inside it
    nums = re.findall(r"(?<![\d.])\$?\s?-?\(?\d[\d,]*(?:\.\d+)?\)?", body)
    for n in nums:
        digits_only = re.sub(r"[^\d.]", "", n)
        int_part = digits_only.split(".")[0]
        if len(int_part) >= 2 or "." in digits_only:
            return n.strip()
    return None


def _normalize_value(v: str | None) -> float | None:
    if v is None:
        return None
    v = v.replace("$", "").replace(",", "").strip().strip("()")
    try:
        return round(abs(float(v)), 2)
    except ValueError:
        return None


def _matches_statement_type(section_title: str, statement_type: str) -> bool:
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


def _mine_table_table(table_rows: pd.DataFrame, metric_labels: dict) -> dict:
    """Returns {(metric, entity, fiscal_period): {chunk_id: (value, doc_name)}}
    restricted to groups with >=2 distinct chunks sharing the same
    normalized value."""
    out = {}
    for metric, (labels, statement_type) in metric_labels.items():
        valid_section = table_rows["section_title"].apply(lambda s: _matches_statement_type(s, statement_type))
        restricted = table_rows[valid_section]
        facts_all = {}
        for _, row in restricted.iterrows():
            if row["label"] not in labels:
                continue
            if row["chunk_id"] in EXCLUDE_CHUNK_IDS:
                continue
            value = _first_value(row["raw_text"])
            if value is None:
                continue
            key = (row["entity"], row["fiscal_period"])
            facts_all.setdefault(key, {})[row["chunk_id"]] = (value, row["doc_name"])

        for key, uniq in facts_all.items():
            by_norm = {}
            for cid, (val, doc_name) in uniq.items():
                norm = _normalize_value(val)
                if norm is None:
                    continue
                by_norm.setdefault(norm, {})[cid] = doc_name
            for norm_val, cid_to_doc in by_norm.items():
                if len(cid_to_doc) < 2:
                    continue
                out[(metric, key, norm_val)] = cid_to_doc
    return out


def _mine_narrative_hits(table_rows: pd.DataFrame, narrative_rows: pd.DataFrame, metric_labels: dict) -> dict:
    """Returns {(metric, entity, fiscal_period, norm_value): {chunk_id: doc_name}}
    for values found in a table AND corroborated by narrative text."""
    out = {}
    for metric, (labels, statement_type) in metric_labels.items():
        valid_section = table_rows["section_title"].apply(lambda s: _matches_statement_type(s, statement_type))
        restricted = table_rows[valid_section]
        for _, row in restricted.iterrows():
            if row["label"] not in labels:
                continue
            if row["chunk_id"] in EXCLUDE_CHUNK_IDS:
                continue
            value = _first_value(row["raw_text"])
            norm_val = _normalize_value(value)
            if norm_val is None:
                continue
            formatted = f"{int(norm_val):,}" if norm_val == int(norm_val) else f"{norm_val:,}"
            if len(formatted) < 4:
                continue
            candidates = narrative_rows[
                (narrative_rows["entity"] == row["entity"]) & (narrative_rows["fiscal_period"] == row["fiscal_period"])
            ]
            matches = candidates[candidates["raw_text"].str.contains(re.escape(formatted), na=False)]
            if len(matches) == 0:
                continue
            key = (metric, (row["entity"], row["fiscal_period"]), norm_val)
            group = out.setdefault(key, {})
            group[row["chunk_id"]] = row["doc_name"]
            for _, m in matches.iterrows():
                group[m["chunk_id"]] = m["doc_name"]
    return out


def _merge(table_table: dict, narrative: dict) -> dict:
    merged = {k: dict(v) for k, v in table_table.items()}
    for k, v in narrative.items():
        merged.setdefault(k, {}).update(v)
    return merged


def mine_redundant_questions(metric_labels: dict, id_prefix: str) -> list[dict]:
    chunks = pd.read_parquet(CHUNKS_PATH)
    table_rows = chunks[chunks["chunk_type"] == "table_row"].copy()
    table_rows["label"] = table_rows["raw_text"].apply(_label)
    table_rows["doc_name"] = table_rows["chunk_id"].str.split("__block").str[0]
    narrative_rows = chunks[chunks["chunk_type"] != "table_row"].copy()
    narrative_rows["doc_name"] = narrative_rows["chunk_id"].str.split("__block").str[0]

    table_table = _mine_table_table(table_rows, metric_labels)
    narrative = _mine_narrative_hits(table_rows, narrative_rows, metric_labels)
    merged = _merge(table_table, narrative)  # {(metric, key, norm_val): {chunk_id: doc_name}}

    # group facts into questions by shared document context, not just
    # (entity, fiscal_period) - a 10-K's and a same-year 10-Q's figures
    # for "total assets" are DIFFERENT facts (different values, already
    # kept separate by norm_val) and must not be bundled into one
    # question just because they nominally share a fiscal_period label
    groups = {}  # group_key -> {metric: {chunk_id: doc_name}}
    for (metric, key, norm_val), chunk_to_doc in merged.items():
        entity, fiscal_period = key
        doc_set = frozenset(chunk_to_doc.values())
        group_key = (entity, fiscal_period, doc_set)
        groups.setdefault(group_key, {})[metric] = chunk_to_doc

    records = []
    counter = 0
    for (entity, fiscal_period, doc_set), metrics in groups.items():
        metric_names = sorted(metrics.keys())
        # split into chunks of MAX_METRICS_PER_QUESTION, same convention as compose_precise_questions.py
        for i in range(0, len(metric_names), MAX_METRICS_PER_QUESTION):
            batch = metric_names[i:i + MAX_METRICS_PER_QUESTION]
            hop_groups = {m: sorted(metrics[m].keys()) for m in batch}
            gold_chunk_ids = [cid for m in batch for cid in hop_groups[m]]
            counter += 1
            if len(batch) == 1:
                question = f"What was {entity}'s {batch[0]} in FY{fiscal_period}?"
            else:
                question = f"What was {entity}'s {', '.join(batch[:-1])} and {batch[-1]} in FY{fiscal_period}?"
            records.append({
                "id": f"{id_prefix}_{counter:03d}",
                "question": question,
                "source": "redundant_mined",
                "gold_chunk_ids": gold_chunk_ids,
                "hop_groups": hop_groups,
                "has_redundant_hops": True,
            })
    return records


if __name__ == "__main__":
    train_records = mine_redundant_questions(TRAIN_METRIC_LABELS, "redundant_mined_train")
    val_records = mine_redundant_questions(VAL_METRIC_LABELS, "redundant_mined_val")
    print(f"mined {len(train_records)} train-metric redundant questions")
    print(f"mined {len(val_records)} val-metric redundant questions")
    for r in train_records[:3]:
        print(r["question"], "->", list(r["hop_groups"].keys()))
